from detector.inputs.base import BaseGPSSource
from detector.data_models import GPSCoordinate
from typing import Optional
from datetime import datetime

class GPSFileSource(BaseGPSSource):
    def __init__(self, filepath: str):
        pass

    def get_coordinate_at_time(self, timestamp: datetime) -> Optional[GPSCoordinate]:
        pass
