from detector.data_models import PotholeRecord, FrameData, Detection, GPSCoordinate
from typing import Optional

class RecordBuilder:
    def __init__(self, video_filename: str):
        pass
        
    def build_record(self, frame_data: FrameData, detection: Detection, gps_coord: Optional[GPSCoordinate], depth_cm: Optional[float]) -> PotholeRecord:
        pass
