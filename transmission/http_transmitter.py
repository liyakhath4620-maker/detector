import logging
import requests
import json
from typing import Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta, timezone

try:
    from detector.transmission.base import BaseTransmitter
    from detector.transmission.local_queue import LocalQueueManager, QueueStatus
    from detector.data_models import PotholeRecord
    from detector.config import DEPTH_NULL_STRATEGY, DepthNullStrategy
except ImportError:
    from transmission.base import BaseTransmitter
    from transmission.local_queue import LocalQueueManager, QueueStatus
    from data_models import PotholeRecord
    from config import DEPTH_NULL_STRATEGY, DepthNullStrategy

logger = logging.getLogger(__name__)


class HttpTransmitter(BaseTransmitter):
    """
    HTTP Transmitter for dispatching PotholeRecord payloads to the OmniSight backend.
    
    Features:
      - Uses persistent requests.Session with configurable connect/read timeouts.
      - On transient failures (connection error, timeout, or 5xx HTTP response):
        enqueues record as PENDING in SQLite queue.db with exponential backoff.
      - On client error (HTTP 400 or 422): marks record as REJECTED_BY_SERVER,
        records the server error response in last_error, and halts further retries.
      - On success (HTTP 2xx): marks record as SENT if previously queued.
    """

    def __init__(
        self,
        server_url: str,
        queue_manager: Optional[LocalQueueManager] = None,
        timeout: Union[float, Tuple[float, float]] = (5.0, 10.0),
        base_backoff_sec: float = 2.0,
        max_backoff_sec: float = 300.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the HttpTransmitter.

        :param server_url: Target endpoint URL for POST requests.
        :param queue_manager: LocalQueueManager instance (defaults to new queue.db instance).
        :param timeout: Timeout tuple (connect_timeout, read_timeout) or float in seconds.
        :param base_backoff_sec: Base delay in seconds for exponential backoff (default: 2.0).
        :param max_backoff_sec: Cap on backoff delay in seconds (default: 300.0).
        :param headers: Optional HTTP request headers (e.g. auth tokens).
        """
        self.server_url = server_url
        self.queue_manager = queue_manager if queue_manager is not None else LocalQueueManager("queue.db")
        self.timeout = timeout
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec

        self.session = requests.Session()
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OmniSight-Detector-Edge/1.0",
        }
        if headers:
            default_headers.update(headers)
        self.session.headers.update(default_headers)

    def _prepare_payload(self, record: Union[PotholeRecord, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Converts PotholeRecord or dict into JSON-ready dictionary and extracts pothole_id."""
        if isinstance(record, dict):
            pothole_id = str(record["pothole_id"])
            payload = dict(record)
        else:
            pothole_id = str(record.pothole_id)
            payload = {
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
            }

        # Apply DEPTH_NULL_STRATEGY if rejected outlier
        if payload.get("depth_quality_flag") == "rejected_outlier" or payload.get("depth_cm") is None:
            if DEPTH_NULL_STRATEGY == DepthNullStrategy.OMIT:
                payload.pop("depth_cm", None)
            elif DEPTH_NULL_STRATEGY == DepthNullStrategy.SENTINEL:
                payload["depth_cm"] = -1.0
            else:
                payload["depth_cm"] = None

        return pothole_id, payload

    def _calculate_backoff(self, retry_count: int) -> timedelta:
        """Calculates exponential backoff delay based on retry_count."""
        delay = min(self.max_backoff_sec, self.base_backoff_sec * (2 ** retry_count))
        return timedelta(seconds=delay)

    def transmit(self, record: Union[PotholeRecord, Dict[str, Any]]) -> bool:
        """
        Transmits a single pothole record to the server via HTTP POST.

        - Success (2xx): marks SENT in queue, returns True.
        - Client error (400, 422): marks REJECTED_BY_SERVER, stops retrying, returns False.
        - Server error (5xx) or Network error (timeout, connection failure):
          enqueues as PENDING with exponential backoff, returns False.

        :param record: PotholeRecord or payload dictionary.
        :return: True if successfully delivered to server, False otherwise.
        """
        pothole_id, payload = self._prepare_payload(record)
        now_utc = datetime.now(timezone.utc)
        current_retry_count = self.queue_manager.get_retry_count(pothole_id)

        try:
            response = self.session.post(
                self.server_url,
                json=payload,
                timeout=self.timeout,
            )

            # Success: 200, 201, 204
            if 200 <= response.status_code < 300:
                logger.info(f"Record {pothole_id} transmitted successfully (HTTP {response.status_code})")
                self.queue_manager.mark_status(pothole_id, QueueStatus.SENT)
                return True

            # Client Error: 400 or 422 (Schema rejection / bad request)
            if response.status_code in (400, 422):
                err_msg = f"HTTP {response.status_code} Rejected by server: {response.text[:500]}"
                logger.warning(f"Record {pothole_id} rejected by server: {err_msg}")
                self.queue_manager.enqueue(
                    payload,
                    status=QueueStatus.REJECTED_BY_SERVER,
                    last_error=err_msg,
                    next_retry_at=None,
                )
                return False

            # Server Error: 5xx (Internal error, Bad Gateway, Service Unavailable)
            if response.status_code >= 500:
                err_msg = f"HTTP {response.status_code} Server Error: {response.text[:300]}"
                logger.warning(f"Transient server error for {pothole_id}: {err_msg}")
                next_retry = now_utc + self._calculate_backoff(current_retry_count)
                self.queue_manager.enqueue(
                    payload,
                    status=QueueStatus.PENDING,
                    last_error=err_msg,
                    next_retry_at=next_retry,
                    increment_retry=True,
                )
                return False

            # Other unexpected 4xx errors
            err_msg = f"HTTP {response.status_code}: {response.text[:300]}"
            next_retry = now_utc + self._calculate_backoff(current_retry_count)
            self.queue_manager.enqueue(
                payload,
                status=QueueStatus.PENDING,
                last_error=err_msg,
                next_retry_at=next_retry,
                increment_retry=True,
            )
            return False

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            err_msg = f"Network failure ({type(e).__name__}): {e}"
            logger.warning(f"Connection failure for record {pothole_id}: {err_msg}")
            next_retry = now_utc + self._calculate_backoff(current_retry_count)
            self.queue_manager.enqueue(
                payload,
                status=QueueStatus.PENDING,
                last_error=err_msg,
                next_retry_at=next_retry,
                increment_retry=True,
            )
            return False

        except requests.exceptions.RequestException as e:
            err_msg = f"Request error ({type(e).__name__}): {e}"
            logger.error(f"Unexpected transmission error for record {pothole_id}: {err_msg}")
            next_retry = now_utc + self._calculate_backoff(current_retry_count)
            self.queue_manager.enqueue(
                payload,
                status=QueueStatus.PENDING,
                last_error=err_msg,
                next_retry_at=next_retry,
                increment_retry=True,
            )
            return False

    def process_pending_queue(self, batch_size: int = 50) -> int:
        """
        Processes pending records in the local queue eligible for retry.

        :param batch_size: Maximum records to attempt in this cycle.
        :return: Number of records successfully transmitted.
        """
        pending = self.queue_manager.get_pending_records(limit=batch_size)
        success_count = 0

        for pothole_id, payload, retry_count in pending:
            logger.debug(f"Retrying pending record {pothole_id} (attempt #{retry_count + 1})...")
            if self.transmit(payload):
                success_count += 1

        return success_count

    def close(self):
        """Closes the underlying requests.Session."""
        self.session.close()
