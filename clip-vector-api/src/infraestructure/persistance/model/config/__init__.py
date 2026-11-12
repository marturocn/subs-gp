"""
CLIP Model Configuration Module
"""
from .clip_config import (
    ClipModelConfig,
    DEFAULT_CONFIG,
    HIGH_PRECISION_CONFIG,
    FAST_CONFIG,
    MAX_PRECISION_CONFIG,
)

from .yolo_config import (
    YoloModelConfig,
    DEFAULT_YOLO_CONFIG,
    FAST_YOLO_CONFIG,
    BALANCED_YOLO_CONFIG,
    HIGH_PRECISION_YOLO_CONFIG,
    MAX_QUALITY_YOLO_CONFIG,
)

from .grounding_dino_config import (
    GroundingDinoConfig,
    DEFAULT_DINO_CONFIG,
    FAST_DINO_CONFIG,
    BALANCED_DINO_CONFIG,
    HIGH_PRECISION_DINO_CONFIG,
    ARTWORK_DINO_CONFIG,
)

from .dinov3_config import (
    DINOv3Config,
    DEFAULT_DINOV3_CONFIG,
    BASE_DINOV3_CONFIG,
    SMALL_DINOV3_CONFIG,
    GIANT_DINOV3_CONFIG,
)

__all__ = [
    # CLIP configs
    "ClipModelConfig",
    "DEFAULT_CONFIG",
    "HIGH_PRECISION_CONFIG",
    "FAST_CONFIG",
    "MAX_PRECISION_CONFIG",
    # YOLO configs
    "YoloModelConfig",
    "DEFAULT_YOLO_CONFIG",
    "FAST_YOLO_CONFIG",
    "BALANCED_YOLO_CONFIG",
    "HIGH_PRECISION_YOLO_CONFIG",
    "MAX_QUALITY_YOLO_CONFIG",
    # GroundingDINO configs
    "GroundingDinoConfig",
    "DEFAULT_DINO_CONFIG",
    "FAST_DINO_CONFIG",
    "BALANCED_DINO_CONFIG",
    "HIGH_PRECISION_DINO_CONFIG",
    "ARTWORK_DINO_CONFIG",
    # DINOv3 configs
    "DINOv3Config",
    "DEFAULT_DINOV3_CONFIG",
    "BASE_DINOV3_CONFIG",
    "SMALL_DINOV3_CONFIG",
    "GIANT_DINOV3_CONFIG",
]
