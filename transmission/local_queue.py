from enum import Enum
from detector.data_models import PotholeRecord

class QueueStatus(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    REJECTED_BY_SERVER = "REJECTED_BY_SERVER"

class LocalQueueManager:
    def __init__(self, db_path: str):
        pass

    def enqueue(self, record: PotholeRecord, status: QueueStatus = QueueStatus.PENDING, last_error: str = ""):
        pass

    def get_pending_records(self):
        pass

    def purge_old_records(self, retention_days: int = 30):
        pass
