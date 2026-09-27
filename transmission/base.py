from abc import ABC, abstractmethod
from detector.data_models import PotholeRecord

class BaseTransmitter(ABC):
    @abstractmethod
    def transmit(self, record: PotholeRecord) -> bool:
        """Transmits a single pothole record. Returns True if successful, False otherwise."""
        pass
