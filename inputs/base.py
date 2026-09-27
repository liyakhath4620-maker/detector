from abc import ABC, abstractmethod
from typing import Generator, Optional
from datetime import datetime
try:
    from detector.data_models import FrameData, GPSCoordinate
except ImportError:
    from data_models import FrameData, GPSCoordinate

class BaseVideoSource(ABC):
    @abstractmethod
    def get_frames(self) -> Generator[FrameData, None, None]:
        """Yields frames from the video source."""
        pass

class BaseGPSSource(ABC):
    @abstractmethod
    def get_coordinate_at_time(self, timestamp: datetime) -> Optional[GPSCoordinate]:
        """Returns the GPS coordinate for a given timestamp, using nearest-neighbor or interpolation."""
        pass
