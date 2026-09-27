import cv2
import logging
from datetime import datetime, timedelta
from typing import Generator, Optional
try:
    from detector.inputs.base import BaseVideoSource
    from detector.data_models import FrameData
except ImportError:
    from inputs.base import BaseVideoSource
    from data_models import FrameData

logger = logging.getLogger(__name__)

class VideoFileSource(BaseVideoSource):
    def __init__(self, filepath: str, sample_fps: float = 2.0, start_time: Optional[datetime] = None):
        self.filepath = filepath
        self.sample_fps = sample_fps
        self.start_time = start_time or datetime.utcnow()
        
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {filepath}")
            
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps <= 0:
            logger.warning(f"Could not determine source FPS for {filepath}. Defaulting to 30.0")
            self.source_fps = 30.0
            
        # Calculate how many frames to skip to achieve the target sample_fps
        # E.g. source is 30 FPS, target is 2 FPS -> process every 15 frames
        self.frame_step = max(1, int(round(self.source_fps / self.sample_fps)))
        logger.info(f"Initialized VideoFileSource: {filepath} (Source FPS: {self.source_fps}, Target FPS: {self.sample_fps}, Step: {self.frame_step})")

    def get_frames(self) -> Generator[FrameData, None, None]:
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_idx % self.frame_step == 0:
                # Calculate absolute timestamp based on start_time and frame milliseconds
                msec_offset = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                current_timestamp = self.start_time + timedelta(milliseconds=msec_offset)
                
                yield FrameData(
                    frame_id=frame_idx,
                    timestamp=current_timestamp,
                    bgr_image=frame
                )
                processed_count += 1
                
            frame_idx += 1
            
        self.cap.release()
        logger.info(f"Video reading complete. Processed {processed_count} frames out of {frame_idx} total.")
