import os
import sqlite3
import json
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, timezone
from contextlib import contextmanager

try:
    from detector.data_models import PotholeRecord
    from detector.config import QUEUE_SENT_RETENTION_DAYS, QUEUE_REJECTED_RETENTION_DAYS
except ImportError:
    from data_models import PotholeRecord
    from config import QUEUE_SENT_RETENTION_DAYS, QUEUE_REJECTED_RETENTION_DAYS

logger = logging.getLogger(__name__)


class QueueStatus(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    REJECTED_BY_SERVER = "REJECTED_BY_SERVER"


class LocalQueueManager:
    """
    SQLite-backed local queue manager for offline buffering, retry scheduling,
    and retention purging of PotholeRecord payloads.
    
    Database features:
      - Four-value status CHECK constraint (PENDING, SENT, FAILED, REJECTED_BY_SERVER).
      - PRAGMA auto_vacuum = INCREMENTAL at DB initialization.
      - purge_old_records() removes expired SENT and REJECTED_BY_SERVER records,
        followed by PRAGMA incremental_vacuum.
    """

    def __init__(self, db_path: str = "queue.db"):
        """
        Initializes the queue manager and ensures SQLite table schema exists.

        :param db_path: Filepath to SQLite database.
        """
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager yielding a SQLite connection that cleanly commits and closes."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initializes database schema and sets PRAGMA auto_vacuum = INCREMENTAL."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._connection() as conn:
            # Set incremental auto_vacuum prior to table creation
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL;")
            conn.execute("PRAGMA journal_mode = WAL;")

            conn.execute("""
            CREATE TABLE IF NOT EXISTS queue (
                pothole_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'SENT', 'FAILED', 'REJECTED_BY_SERVER')),
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_retry_at TIMESTAMP,
                last_error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_status_retry
            ON queue (status, next_retry_at);
            """)

        logger.info(f"LocalQueueManager initialized database at {self.db_path}")

    def _serialize_record(self, record: Union[PotholeRecord, Dict[str, Any]]) -> Tuple[str, str]:
        """Extracts (pothole_id, payload_json_string) from PotholeRecord or dict."""
        if isinstance(record, dict):
            pothole_id = str(record["pothole_id"])
            payload_str = json.dumps(record)
        else:
            pothole_id = str(record.pothole_id)
            if hasattr(record, "__dict__"):
                payload_str = json.dumps(record.__dict__)
            else:
                payload_str = json.dumps({
                    "pothole_id": record.pothole_id,
                    "timestamp": record.timestamp,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "depth_cm": record.depth_cm,
                    "confidence": record.confidence,
                    "gps_fix_quality": record.gps_fix_quality,
                    "depth_quality_flag": record.depth_quality_flag,
                    "bounding_box": record.bounding_box,
                    "metadata": record.metadata,
                })
        return pothole_id, payload_str

    def enqueue(
        self,
        record: Union[PotholeRecord, Dict[str, Any]],
        status: QueueStatus = QueueStatus.PENDING,
        last_error: str = "",
        next_retry_at: Optional[datetime] = None,
        increment_retry: bool = False,
    ):
        """
        Inserts or updates a record in the local queue.

        :param record: PotholeRecord or dictionary payload.
        :param status: QueueStatus enum value.
        :param last_error: Diagnostic error message from transmission attempt.
        :param next_retry_at: Timestamp for next retry attempt (None if immediate/none).
        :param increment_retry: If True, increments the retry count on conflict.
        """
        pothole_id, payload_str = self._serialize_record(record)
        next_retry_str = next_retry_at.strftime("%Y-%m-%d %H:%M:%S") if next_retry_at else None

        with self._connection() as conn:
            if increment_retry:
                conn.execute("""
                INSERT INTO queue (pothole_id, payload, status, retry_count, next_retry_at, last_error, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(pothole_id) DO UPDATE SET
                    payload = excluded.payload,
                    status = excluded.status,
                    retry_count = queue.retry_count + 1,
                    next_retry_at = excluded.next_retry_at,
                    last_error = excluded.last_error,
                    updated_at = CURRENT_TIMESTAMP;
                """, (pothole_id, payload_str, status.value, next_retry_str, last_error))
            else:
                conn.execute("""
                INSERT INTO queue (pothole_id, payload, status, retry_count, next_retry_at, last_error, updated_at)
                VALUES (?, ?, ?, 0, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(pothole_id) DO UPDATE SET
                    payload = excluded.payload,
                    status = excluded.status,
                    next_retry_at = excluded.next_retry_at,
                    last_error = excluded.last_error,
                    updated_at = CURRENT_TIMESTAMP;
                """, (pothole_id, payload_str, status.value, next_retry_str, last_error))

        logger.debug(f"Queued record {pothole_id} with status={status.value}")

    def mark_status(
        self,
        pothole_id: str,
        status: QueueStatus,
        last_error: str = "",
        next_retry_at: Optional[datetime] = None,
        increment_retry: bool = False,
    ):
        """
        Updates the status, error, and retry timing for an existing record.

        :param pothole_id: Unique record ID.
        :param status: Target QueueStatus.
        :param last_error: Error description string.
        :param next_retry_at: Optional next retry datetime.
        :param increment_retry: If True, increments retry_count by 1.
        """
        next_retry_str = next_retry_at.strftime("%Y-%m-%d %H:%M:%S") if next_retry_at else None

        with self._connection() as conn:
            if increment_retry:
                conn.execute("""
                UPDATE queue
                SET status = ?,
                    last_error = ?,
                    next_retry_at = ?,
                    retry_count = retry_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE pothole_id = ?;
                """, (status.value, last_error, next_retry_str, pothole_id))
            else:
                conn.execute("""
                UPDATE queue
                SET status = ?,
                    last_error = ?,
                    next_retry_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE pothole_id = ?;
                """, (status.value, last_error, next_retry_str, pothole_id))

    def get_retry_count(self, pothole_id: str) -> int:
        """Returns the current retry count for a pothole_id (or 0 if not found)."""
        with self._connection() as conn:
            row = conn.execute("SELECT retry_count FROM queue WHERE pothole_id = ?", (pothole_id,)).fetchone()
            return int(row["retry_count"]) if row else 0

    def get_pending_records(self, limit: int = 50) -> List[Tuple[str, Dict[str, Any], int]]:
        """
        Retrieves records eligible for retry:
        status == 'PENDING' AND (next_retry_at IS NULL OR next_retry_at <= CURRENT_TIMESTAMP).

        :param limit: Maximum batch size.
        :return: List of tuples (pothole_id, payload_dict, retry_count).
        """
        with self._connection() as conn:
            rows = conn.execute("""
            SELECT pothole_id, payload, retry_count
            FROM queue
            WHERE status = 'PENDING'
              AND (next_retry_at IS NULL OR next_retry_at <= CURRENT_TIMESTAMP)
            ORDER BY created_at ASC
            LIMIT ?;
            """, (limit,)).fetchall()

            results = []
            for row in rows:
                try:
                    payload = json.loads(row["payload"])
                    results.append((row["pothole_id"], payload, int(row["retry_count"])))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse payload for {row['pothole_id']}: {e}")
            return results

    def get_record(self, pothole_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single record's database row as a dictionary."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM queue WHERE pothole_id = ?", (pothole_id,)).fetchone()
            if row:
                d = dict(row)
                try:
                    d["payload"] = json.loads(d["payload"])
                except Exception:
                    pass
                return d
            return None

    def purge_old_records(
        self,
        sent_retention_days: int = QUEUE_SENT_RETENTION_DAYS,
        rejected_retention_days: int = QUEUE_REJECTED_RETENTION_DAYS,
    ) -> int:
        """
        Deletes SENT rows older than sent_retention_days (30 days) and
        REJECTED_BY_SERVER rows older than rejected_retention_days (60 days),
        then executes PRAGMA incremental_vacuum.

        :param sent_retention_days: Days to retain SENT records (default: 30).
        :param rejected_retention_days: Days to retain REJECTED_BY_SERVER records (default: 60).
        :return: Total number of purged rows.
        """
        total_purged = 0
        with self._connection() as conn:
            cursor_sent = conn.execute(
                "DELETE FROM queue WHERE status = 'SENT' AND updated_at < datetime('now', ?);",
                (f"-{int(sent_retention_days)} days",)
            )
            sent_deleted = cursor_sent.rowcount

            cursor_rejected = conn.execute(
                "DELETE FROM queue WHERE status = 'REJECTED_BY_SERVER' AND updated_at < datetime('now', ?);",
                (f"-{int(rejected_retention_days)} days",)
            )
            rejected_deleted = cursor_rejected.rowcount

            total_purged = sent_deleted + rejected_deleted

            # Execute incremental vacuum to reclaim deleted database pages
            conn.execute("PRAGMA incremental_vacuum;")

        logger.info(
            f"Purged {total_purged} expired records ({sent_deleted} SENT, {rejected_deleted} REJECTED_BY_SERVER). "
            f"Executed incremental vacuum."
        )
        return total_purged
