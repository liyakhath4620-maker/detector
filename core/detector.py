import logging
from typing import List, Optional, Any

try:
    from detector.data_models import Detection
except ImportError:
    from data_models import Detection

logger = logging.getLogger(__name__)

# ==============================================================================
# PLACEHOLDER WEIGHTS:
# Defaulting to standard pretrained 'yolov8n.pt' for prototyping and verification.
#
# >>> SWAP IN REAL FINE-TUNED WEIGHTS HERE <<<
# When real fine-tuned weights (e.g. 'weights/pothole_yolov8n_best.pt') are available,
# replace DEFAULT_MODEL_PATH below or pass model_path directly to PotholeDetector().
# ==============================================================================
DEFAULT_MODEL_PATH = "yolov8n.pt"


class PotholeDetector:
    """
    YOLOv8 wrapper for road defect and pothole detection.
    
    Supports automatic hardware acceleration (CUDA if available, otherwise CPU),
    configurable confidence and NMS IoU thresholds, and maps model outputs
    directly to Detection dataclass instances with bounding box, confidence,
    and class label.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.50,
        device: Optional[str] = None,
    ):
        """
        Initializes the detector with a YOLOv8 model.

        :param model_path: Path to model weights (.pt). If None, defaults to DEFAULT_MODEL_PATH.
                           [SWAP WITH FINE-TUNED WEIGHTS PATH HERE]
        :param conf_thresh: Minimum detection confidence threshold (default: 0.40).
        :param iou_thresh: Non-Maximum Suppression (NMS) IoU threshold (default: 0.50).
        :param device: Inference device ('cuda', 'cpu', or None for auto-detection).
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Auto-detect device: CUDA if available, else CPU
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if self._has_cuda() else "cpu"

        # Resolve weights path
        self.model_path = model_path if model_path else DEFAULT_MODEL_PATH
        logger.info(
            f"Initializing PotholeDetector with weights: '{self.model_path}' on device: '{self.device}' "
            f"(conf: {self.conf_thresh}, iou: {self.iou_thresh})"
        )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics YOLO is required for PotholeDetector. Please install ultralytics."
            ) from e

        # Load YOLO model and send to target device
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    @staticmethod
    def _has_cuda() -> bool:
        """Checks if CUDA is available without failing if torch is not yet installed."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def detect(self, bgr_image) -> List[Detection]:
        """
        Runs inference on a single BGR image (e.g., OpenCV frame).

        :param bgr_image: NumPy array representing the BGR frame from OpenCV.
        :return: List of Detection dataclass instances containing bounding box,
                 confidence, class ID, and class label.
        """
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            logger.warning("Empty or invalid image passed to PotholeDetector.detect")
            return []

        # Run YOLO inference
        results = self.model.predict(
            source=bgr_image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        if not results:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        names = getattr(first_result, "names", None) or {}

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy]

            conf = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            class_label = names.get(class_id, str(class_id))

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=class_id,
                    class_label=class_label,
                )
            )

        return detections
