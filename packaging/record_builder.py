import os
import json
import uuid
import logging
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime

try:
    from detector.data_models import PotholeRecord, FrameData, Detection, GPSCoordinate
    from detector.config import DEPTH_NULL_STRATEGY, DepthNullStrategy
except ImportError:
    from data_models import PotholeRecord, FrameData, Detection, GPSCoordinate
    from config import DEPTH_NULL_STRATEGY, DepthNullStrategy

logger = logging.getLogger(__name__)


class RecordBuilder:
    """
    Combines outputs from Stage 3 (Detector), Stage 4 (Depth Estimator), and
    Stage 5 (GPS Source) into standard PotholeRecord objects conforming to
    the OmniSight backend schema.
    
    Applies the DEPTH_NULL_STRATEGY config when depth_quality_flag is 'rejected_outlier'.
    Routes records with gps_fix_quality == 'no_fix' to data/quarantine_{session_id}.jsonl.
    """

    def __init__(
        self,
        video_filename: str,
        session_id: Optional[str] = None,
        depth_null_strategy: Optional[DepthNullStrategy] = None,
        quarantine_dir: str = "data",
    ):
        """
        Initializes the RecordBuilder.

        :param video_filename: Base filename of the video being processed.
        :param session_id: Unique session identifier (defaults to generated UUID).
        :param depth_null_strategy: DepthNullStrategy enum (defaults to config.DEPTH_NULL_STRATEGY).
        :param quarantine_dir: Directory where unlocalized records are quarantined.
        """
        self.video_filename = video_filename
        self.session_id = session_id if session_id else str(uuid.uuid4())[:8]
        self.depth_null_strategy = depth_null_strategy if depth_null_strategy is not None else DEPTH_NULL_STRATEGY
        self.quarantine_dir = quarantine_dir
        self.quarantine_path = os.path.join(self.quarantine_dir, f"quarantine_{self.session_id}.jsonl")

    def build_record(
        self,
        frame_data: FrameData,
        detection: Detection,
        gps_coord: Optional[Any] = None,
        depth_cm: Optional[Any] = None,
        depth_quality_flag: Optional[str] = None,
        gps_fix_quality: Optional[str] = None,
    ) -> PotholeRecord:
        """
        Constructs a PotholeRecord from frame, detection, GPS, and depth information.

        :param frame_data: FrameData instance containing frame_id and timestamp.
        :param detection: Detection instance with bounding box and confidence.
        :param gps_coord: GPSCoordinate instance, GPSMatchResult tuple, or None.
        :param depth_cm: Calculated depth (float), DepthResult tuple, or None.
        :param depth_quality_flag: Explicit depth quality flag ('ok' or 'rejected_outlier').
        :param gps_fix_quality: Explicit GPS fix quality ('matched', 'interpolated', 'no_fix').
        :return: Populated PotholeRecord dataclass instance.
        """
        # 1. Resolve GPS coordinate and fix quality
        gps_obj: Optional[GPSCoordinate] = None
        resolved_gps_quality = "no_fix"

        if gps_coord is not None:
            # Handle tuple/GPSMatchResult: (coord, fix_quality)
            if isinstance(gps_coord, tuple) and len(gps_coord) == 2:
                gps_obj = gps_coord[0]
                resolved_gps_quality = str(gps_coord[1])
            elif isinstance(gps_coord, GPSCoordinate):
                gps_obj = gps_coord
                resolved_gps_quality = getattr(gps_coord, "gps_fix_quality", getattr(gps_coord, "fix_quality", "matched"))

        if gps_fix_quality is not None:
            resolved_gps_quality = gps_fix_quality
        elif gps_obj is None:
            resolved_gps_quality = "no_fix"

        lat = gps_obj.latitude if gps_obj is not None else None
        lon = gps_obj.longitude if gps_obj is not None else None

        # 2. Resolve depth value and depth quality flag
        depth_val: Optional[float] = None
        resolved_depth_flag = "rejected_outlier"

        if depth_cm is not None:
            # Handle tuple/DepthResult: (depth_cm, quality_flag)
            if isinstance(depth_cm, tuple) and len(depth_cm) == 2:
                depth_val = depth_cm[0]
                resolved_depth_flag = str(depth_cm[1])
            elif isinstance(depth_cm, (int, float)):
                depth_val = float(depth_cm)
                resolved_depth_flag = "ok" if (0.0 <= depth_val <= 30.0) else "rejected_outlier"

        if depth_quality_flag is not None:
            resolved_depth_flag = depth_quality_flag
        elif depth_val is None:
            resolved_depth_flag = "rejected_outlier"

        # 3. Apply DEPTH_NULL_STRATEGY when depth_quality_flag is 'rejected_outlier'
        if resolved_depth_flag == "rejected_outlier":
            if self.depth_null_strategy == DepthNullStrategy.SENTINEL:
                depth_val = -1.0
            else:
                depth_val = None

        # 4. Assemble Bounding Box dictionary
        bounding_box = {
            "x1": int(detection.x1),
            "y1": int(detection.y1),
            "x2": int(detection.x2),
            "y2": int(detection.y2),
        }

        # 5. Assemble Metadata dictionary
        metadata: Dict[str, Any] = {
            "video_filename": self.video_filename,
            "frame_id": frame_data.frame_id,
            "class_label": detection.class_label,
            "class_id": detection.class_id,
            "session_id": self.session_id,
        }

        if gps_obj is not None:
            if getattr(gps_obj, "speed", None) is not None:
                metadata["gps_speed"] = gps_obj.speed
            if getattr(gps_obj, "altitude", None) is not None:
                metadata["gps_altitude"] = gps_obj.altitude
            if getattr(gps_obj, "accuracy", None) is not None:
                metadata["gps_accuracy"] = gps_obj.accuracy

        if hasattr(depth_cm, "road_baseline") and getattr(depth_cm, "road_baseline", None) is not None:
            metadata["road_baseline"] = depth_cm.road_baseline
        if hasattr(depth_cm, "interior_depth") and getattr(depth_cm, "interior_depth", None) is not None:
            metadata["interior_depth"] = depth_cm.interior_depth

        # Generate unique pothole record ID
        pothole_id = str(uuid.uuid4())

        # ISO timestamp string
        ts_str = frame_data.timestamp.isoformat() if isinstance(frame_data.timestamp, datetime) else str(frame_data.timestamp)

        return PotholeRecord(
            pothole_id=pothole_id,
            timestamp=ts_str,
            latitude=lat,
            longitude=lon,
            depth_cm=depth_val,
            confidence=round(float(detection.confidence), 4),
            gps_fix_quality=resolved_gps_quality,
            depth_quality_flag=resolved_depth_flag,
            bounding_box=bounding_box,
            metadata=metadata,
        )

    def record_to_dict(self, record: PotholeRecord) -> Dict[str, Any]:
        """
        Serializes PotholeRecord to a dictionary conforming to the JSON schema,
        applying DEPTH_NULL_STRATEGY (e.g. omitting 'depth_cm' if strategy is OMIT).
        """
        payload: Dict[str, Any] = {
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

        # Apply DEPTH_NULL_STRATEGY for serialization
        if record.depth_quality_flag == "rejected_outlier" or record.depth_cm is None:
            if self.depth_null_strategy == DepthNullStrategy.OMIT:
                payload.pop("depth_cm", None)
            elif self.depth_null_strategy == DepthNullStrategy.SENTINEL:
                payload["depth_cm"] = -1.0
            else:
                payload["depth_cm"] = None

        return payload

    def record_to_json(self, record: PotholeRecord) -> str:
        """Serializes PotholeRecord to a JSON string."""
        return json.dumps(self.record_to_dict(record))

    def quarantine_record(self, record: PotholeRecord) -> str:
        """
        Writes an unlocalized record to the session quarantine JSONL file.

        :param record: PotholeRecord with gps_fix_quality == 'no_fix'.
        :return: Absolute or relative filepath of the quarantine file.
        """
        os.makedirs(self.quarantine_dir, exist_ok=True)
        record_json = self.record_to_json(record)
        with open(self.quarantine_path, "a", encoding="utf-8") as f:
            f.write(record_json + "\n")

        logger.warning(
            f"Record {record.pothole_id} quarantined (gps_fix_quality={record.gps_fix_quality}) "
            f"to {self.quarantine_path}"
        )
        return self.quarantine_path

    def route_record(self, record: PotholeRecord) -> bool:
        """
        Routes the record based on GPS fix quality:
          - If gps_fix_quality == 'no_fix': quarantined to file, returns False (do not transmit).
          - Otherwise: returns True (proceed to transmission queue).

        :param record: PotholeRecord to route.
        :return: True if eligible for transmission, False if quarantined.
        """
        if record.gps_fix_quality == "no_fix":
            self.quarantine_record(record)
            return False
        return True

    def build_and_route(
        self,
        frame_data: FrameData,
        detection: Detection,
        gps_coord: Optional[Any] = None,
        depth_cm: Optional[Any] = None,
    ) -> Tuple[PotholeRecord, bool]:
        """
        Builds the PotholeRecord and immediately applies routing.

        :return: Tuple of (record, should_transmit: bool).
        """
        record = self.build_record(frame_data, detection, gps_coord, depth_cm)
        should_transmit = self.route_record(record)
        return record, should_transmit
