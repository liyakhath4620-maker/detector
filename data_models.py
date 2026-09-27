from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class FrameData:
    frame_id: int
    timestamp: datetime
    bgr_image: Any  # numpy array (cv2 image)

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_label: str

@dataclass
class GPSCoordinate:
    timestamp: datetime
    latitude: float
    longitude: float
    speed: Optional[float] = None
    altitude: Optional[float] = None
    accuracy: Optional[float] = None

@dataclass
class PotholeRecord:
    pothole_id: str
    timestamp: str
    latitude: Optional[float]
    longitude: Optional[float]
    depth_cm: Optional[float]
    confidence: float
    gps_fix_quality: str
    depth_quality_flag: str
    bounding_box: Dict[str, int]
    metadata: Dict[str, Any]
