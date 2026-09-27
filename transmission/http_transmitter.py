from detector.transmission.base import BaseTransmitter
from detector.data_models import PotholeRecord

class HttpTransmitter(BaseTransmitter):
    def __init__(self, server_url: str):
        pass

    def transmit(self, record: PotholeRecord) -> bool:
        pass
