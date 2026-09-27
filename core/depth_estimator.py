import logging
from typing import List, Optional, Tuple, Any
import numpy as np

try:
    from detector.data_models import Detection
except ImportError:
    from data_models import Detection

logger = logging.getLogger(__name__)

# Default pretrained Depth Anything V2 metric model
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthQualityFlag:
    OK = "ok"
    REJECTED_OUTLIER = "rejected_outlier"


class DepthResult(tuple):
    """
    Result container for pothole metric depth calculation.
    
    Unpacks as a 2-tuple: (depth_cm, depth_quality_flag).
    Also provides named attribute access (depth_cm, depth_quality_flag,
    road_baseline, interior_depth, raw_depth_cm).
    """

    def __new__(
        cls,
        depth_cm: Optional[float],
        depth_quality_flag: str,
        raw_depth_cm: Optional[float] = None,
        road_baseline: Optional[float] = None,
        interior_depth: Optional[float] = None,
    ):
        return super().__new__(cls, (depth_cm, depth_quality_flag))

    def __init__(
        self,
        depth_cm: Optional[float],
        depth_quality_flag: str,
        raw_depth_cm: Optional[float] = None,
        road_baseline: Optional[float] = None,
        interior_depth: Optional[float] = None,
    ):
        self.depth_cm = depth_cm
        self.depth_quality_flag = depth_quality_flag
        self.raw_depth_cm = raw_depth_cm
        self.road_baseline = road_baseline
        self.interior_depth = interior_depth

    @property
    def quality_flag(self) -> str:
        return self.depth_quality_flag

    @property
    def baseline_m(self) -> Optional[float]:
        return self.road_baseline

    @property
    def interior_m(self) -> Optional[float]:
        return self.interior_depth

    def __repr__(self) -> str:
        return (
            f"DepthResult(depth_cm={self.depth_cm}, depth_quality_flag='{self.depth_quality_flag}', "
            f"raw_depth_cm={self.raw_depth_cm}, road_baseline={self.road_baseline}, interior_depth={self.interior_depth})"
        )


class MetricDepthEstimator:
    """
    Metric depth estimator wrapping Depth Anything V2 (metric variant).
    
    For each detected pothole bounding box:
      1. Computes the median depth of an annular rim just outside the box as road_baseline.
      2. Finds the deepest point inside the box using percentile filtering to reject edge artifacts.
      3. Calculates: depth_cm = (interior_depth - road_baseline) * 100.
      4. Rejects and flags as 'rejected_outlier' any negative depth or any depth over 30cm.
      5. Returns both depth_cm and depth_quality_flag ('ok' or 'rejected_outlier').
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        rim_margin_ratio: float = 0.20,
        interior_percentile: float = 95.0,
        min_depth_cm: float = 0.0,
        max_depth_cm: float = 30.0,
        lazy_load: bool = True,
    ):
        """
        Initializes the Depth Anything V2 metric estimator.

        :param model_name: Hugging Face model identifier (default: Depth-Anything-V2-Small-hf).
        :param device: Hardware device ('cuda', 'cpu', or auto-detect).
        :param rim_margin_ratio: Expansion margin around bounding box for road rim (default: 0.20).
        :param interior_percentile: Percentile (0-100) for interior deepest point to reject edge artifacts (default: 95.0).
        :param min_depth_cm: Minimum plausible depth threshold in cm (default: 0.0).
        :param max_depth_cm: Maximum plausible depth threshold in cm (default: 30.0).
        :param lazy_load: If True, defers model weight loading until first inference call.
        """
        self.model_name = model_name if model_name else DEFAULT_DEPTH_MODEL
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.rim_margin_ratio = rim_margin_ratio
        self.interior_percentile = interior_percentile
        self.min_depth_cm = min_depth_cm
        self.max_depth_cm = max_depth_cm

        self.processor = None
        self.model = None

        if not lazy_load:
            self._load_model()

    @staticmethod
    def _has_cuda() -> bool:
        """Checks if CUDA is available without failing if torch is not yet installed."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        """Loads Hugging Face image processor and Depth Anything V2 model onto target device."""
        if self.model is None:
            logger.info(f"Loading Depth Anything V2 model '{self.model_name}' on device '{self.device}'...")
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            except ImportError as e:
                raise ImportError(
                    "PyTorch and Hugging Face Transformers are required for Depth Anything V2. "
                    "Please ensure torch and transformers are installed."
                ) from e

            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def estimate_depth_map(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Runs metric depth prediction across the full BGR image using Depth Anything V2.

        :param bgr_image: OpenCV BGR image (H, W, 3).
        :return: Metric depth map in meters as 2D NumPy array (H, W).
        """
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            raise ValueError("Invalid or empty image provided for depth estimation.")

        try:
            import cv2
            import torch
        except ImportError as e:
            raise ImportError("OpenCV and PyTorch are required to run estimate_depth_map.") from e

        self._load_model()
        h, w = bgr_image.shape[:2]

        # Convert BGR (OpenCV) to RGB for transformers processor
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate metric depth map back to original image dimensions
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return prediction.squeeze().cpu().numpy()

    def compute_depth_from_map(
        self,
        depth_map: np.ndarray,
        detection: Detection,
        nullify_outliers: bool = True,
    ) -> DepthResult:
        """
        Computes pothole depth from a 2D metric depth map using annular rim road baseline
        and percentile-filtered interior deepest point.

        :param depth_map: 2D NumPy array of metric depth in meters.
        :param detection: Detection dataclass with bounding box (x1, y1, x2, y2).
        :param nullify_outliers: If True, sets depth_cm to None when flagged as outlier.
        :return: DepthResult tuple (depth_cm, depth_quality_flag).
        """
        if depth_map is None or getattr(depth_map, "size", 0) == 0:
            return DepthResult(depth_cm=None, depth_quality_flag=DepthQualityFlag.REJECTED_OUTLIER)

        img_h, img_w = depth_map.shape[:2]

        # Clamp bounding box coordinates to image boundaries
        x1 = max(0, min(detection.x1, img_w - 1))
        y1 = max(0, min(detection.y1, img_h - 1))
        x2 = max(x1 + 1, min(detection.x2, img_w))
        y2 = max(y1 + 1, min(detection.y2, img_h))

        box_w = x2 - x1
        box_h = y2 - y1

        # Calculate annular rim margin just outside the box
        margin_x = max(3, int(round(box_w * self.rim_margin_ratio)))
        margin_y = max(3, int(round(box_h * self.rim_margin_ratio)))

        ox1 = max(0, x1 - margin_x)
        oy1 = max(0, y1 - margin_y)
        ox2 = min(img_w, x2 + margin_x)
        oy2 = min(img_h, y2 + margin_y)

        # Build annular collar mask (outer expanded box excluding interior pothole box)
        collar_mask = np.zeros((img_h, img_w), dtype=bool)
        collar_mask[oy1:oy2, ox1:ox2] = True
        collar_mask[y1:y2, x1:x2] = False

        rim_pixels = depth_map[collar_mask]
        if rim_pixels.size == 0:
            logger.warning(f"No annular rim pixels available for detection at [{x1}, {y1}, {x2}, {y2}]")
            return DepthResult(depth_cm=None, depth_quality_flag=DepthQualityFlag.REJECTED_OUTLIER)

        # 1. Median depth of annular rim just outside the box as road baseline (in meters)
        road_baseline = float(np.median(rim_pixels))

        # 2. Deepest point inside the box using percentile filtering to reject edge artifacts
        interior_patch = depth_map[y1:y2, x1:x2]
        if interior_patch.size == 0:
            return DepthResult(depth_cm=None, depth_quality_flag=DepthQualityFlag.REJECTED_OUTLIER)

        interior_depth = float(np.percentile(interior_patch, self.interior_percentile))

        # 3. Calculate depth in centimeters: (interior_depth - road_baseline) * 100
        raw_depth_cm = (interior_depth - road_baseline) * 100.0

        # 4. Outlier verification: reject negative or > 30cm as 'rejected_outlier'
        if raw_depth_cm < self.min_depth_cm or raw_depth_cm > self.max_depth_cm:
            quality_flag = DepthQualityFlag.REJECTED_OUTLIER
            depth_cm = None if nullify_outliers else round(raw_depth_cm, 2)
            logger.debug(
                f"Pothole flagged as {quality_flag}: depth={raw_depth_cm:.2f}cm "
                f"(baseline={road_baseline:.3f}m, interior={interior_depth:.3f}m)"
            )
        else:
            quality_flag = DepthQualityFlag.OK
            depth_cm = round(raw_depth_cm, 2)

        return DepthResult(
            depth_cm=depth_cm,
            depth_quality_flag=quality_flag,
            raw_depth_cm=round(raw_depth_cm, 2),
            road_baseline=round(road_baseline, 4),
            interior_depth=round(interior_depth, 4),
        )

    def calculate_pothole_depth(
        self,
        bgr_image: np.ndarray,
        detection: Detection,
        depth_map: Optional[np.ndarray] = None,
        nullify_outliers: bool = True,
    ) -> DepthResult:
        """
        Calculates pothole depth for a detection within a frame.
        Reuses depth_map if provided to avoid redundant model inference across multiple detections.

        :param bgr_image: BGR frame from video capture.
        :param detection: Detection dataclass.
        :param depth_map: Optional precomputed 2D metric depth map.
        :param nullify_outliers: If True, sets depth_cm to None on outlier.
        :return: DepthResult tuple (depth_cm, depth_quality_flag).
        """
        if depth_map is None:
            depth_map = self.estimate_depth_map(bgr_image)

        return self.compute_depth_from_map(
            depth_map=depth_map,
            detection=detection,
            nullify_outliers=nullify_outliers,
        )
