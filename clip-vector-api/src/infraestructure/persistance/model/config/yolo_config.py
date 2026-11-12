"""
YOLO Model Configuration
Configure YOLO settings for object detection
"""
from dataclasses import dataclass
from typing import Literal
import os

# Boolean conversion constants
TRUE_STR = "true"
FALSE_STR = "false"


@dataclass
class YoloModelConfig:
    """
    Configuration for YOLO model loading and inference.
    
    Attributes:
        model_version: YOLO version to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        confidence_threshold: Minimum confidence for detection (0.0 - 1.0)
        iou_threshold: IoU threshold for NMS (0.0 - 1.0)
        max_detections: Maximum number of detections per image
        min_box_size: Minimum bounding box size (width or height in pixels)
        device: Device to use ('cpu' only for this deployment)
        enable_tracking: Enable object tracking across frames
    """
    
    # Model selection
    model_version: str = "yolov8n"  # Options: yolov8n (nano), yolov8s (small), yolov8m, yolov8l, yolov8x
    
    # Detection parameters
    confidence_threshold: float = 0.25  # Minimum confidence (0.0 - 1.0)
    iou_threshold: float = 0.45  # IoU threshold for NMS
    max_detections: int = 100  # Maximum detections per image
    min_box_size: int = 50  # Minimum box size (pixels)
    
    # Device configuration (CPU only)
    device: Literal["cpu"] = "cpu"
    
    # Advanced options
    enable_tracking: bool = False  # Object tracking
    agnostic_nms: bool = False  # Class-agnostic NMS
    
    # Performance
    half_precision: bool = False  # FP16 inference (CPU doesn't support, but keep for future)
    
    @staticmethod
    def from_env() -> "YoloModelConfig":
        """Load configuration from environment variables."""
        return YoloModelConfig(
            model_version=os.getenv("YOLO_MODEL_VERSION", "yolov8n"),
            confidence_threshold=float(os.getenv("YOLO_CONFIDENCE", "0.25")),
            iou_threshold=float(os.getenv("YOLO_IOU_THRESHOLD", "0.45")),
            max_detections=int(os.getenv("YOLO_MAX_DETECTIONS", "100")),
            min_box_size=int(os.getenv("YOLO_MIN_BOX_SIZE", "50")),
            device="cpu",  # Always CPU
            enable_tracking=os.getenv("YOLO_ENABLE_TRACKING", FALSE_STR).lower() == TRUE_STR,
            agnostic_nms=os.getenv("YOLO_AGNOSTIC_NMS", FALSE_STR).lower() == TRUE_STR,
        )
    
    def get_model_name(self) -> str:
        """Get the full model name with extension."""
        return f"{self.model_version}.pt"


# Default configuration
DEFAULT_YOLO_CONFIG = YoloModelConfig()


# Fast configuration (nano model, lower confidence)
FAST_YOLO_CONFIG = YoloModelConfig(
    model_version="yolov8n",
    confidence_threshold=0.3,
    iou_threshold=0.45,
    max_detections=50,
    min_box_size=40,
)


# Balanced configuration (small model, good balance)
BALANCED_YOLO_CONFIG = YoloModelConfig(
    model_version="yolov8s",
    confidence_threshold=0.25,
    iou_threshold=0.45,
    max_detections=100,
    min_box_size=50,
)


# High precision configuration (medium model, stricter confidence)
HIGH_PRECISION_YOLO_CONFIG = YoloModelConfig(
    model_version="yolov8m",
    confidence_threshold=0.35,
    iou_threshold=0.40,
    max_detections=100,
    min_box_size=60,
)


# Maximum quality configuration (large model, best accuracy)
MAX_QUALITY_YOLO_CONFIG = YoloModelConfig(
    model_version="yolov8l",
    confidence_threshold=0.30,
    iou_threshold=0.40,
    max_detections=150,
    min_box_size=50,
)
