"""
Grounding DINO Model Configuration
Configure Grounding DINO settings for open-set object detection
"""
from dataclasses import dataclass
from typing import Literal
import os

# Environment variable constants
CUDA_AVAILABLE_ENV = "CUDA_AVAILABLE"

# Boolean conversion constants
TRUE_STR = "true"
FALSE_STR = "false"

# Default text prompt for artwork detection
DEFAULT_TEXT_PROMPT = "object . thing . item . artwork . piece"


@dataclass
class GroundingDinoConfig:
    """
    Configuration for Grounding DINO model loading and inference.
    
    Grounding DINO is an open-set object detector that uses text prompts
    to detect arbitrary objects without being limited to predefined classes.
    
    Attributes:
        config_path: Path to model config file (e.g., "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        model_path: Path to model weights (e.g., "weights/groundingdino_swint_ogc.pth")
        text_prompt: Default text prompt for detection (e.g., "person . car . dog")
        box_threshold: Minimum confidence for detection (0.0 - 1.0)
        text_threshold: Minimum text similarity threshold (0.0 - 1.0)
        device: Device to use ('cpu' or 'cuda')
        max_detections: Maximum number of detections per image
        min_box_size: Minimum bounding box size (width or height in pixels)
    """
    
    # Model paths
    config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_path: str = "weights/groundingdino_swint_ogc.pth"
    
    # Text prompt for detection
    text_prompt: str = DEFAULT_TEXT_PROMPT  # Default for artworks
    
    # Detection parameters
    box_threshold: float = 0.35  # Minimum box confidence (0.0 - 1.0)
    text_threshold: float = 0.25  # Minimum text similarity threshold
    max_detections: int = 100  # Maximum detections per image
    min_box_size: int = 50  # Minimum box size (pixels)
    
    # Device configuration
    device: Literal["cpu", "cuda"] = "cuda"  # Default to CUDA for maximum performance
    # If True, force CPU-only mode regardless of available CUDA devices.
    cpu_only: bool = False  # Default to CUDA-enabled for performance
    
    # Advanced options
    nms_threshold: float = 0.45  # Non-maximum suppression threshold
    # Auto-download behavior: if false, repository will NOT download weights/config automatically
    auto_download: bool = True
    # Maximum dimension (pixels) for the longest side used during detection to speed up inference
    # Lower = faster but less accurate. 1024 is balanced, 768 is fast, 640 is very fast, 512 is ultra-fast
    max_detection_dimension: int = 860  # Optimizado para velocidad (era 768)
    
    use_compile: bool = False  # torch.compile not compatible with Grounding DINO (mixed device issues)
    
    # ONNX export settings
    onnx_export_path: str = "models/grounding_dino_optimized.onnx"
    use_onnx: bool = False  # Use ONNX model for inference instead of PyTorch
    onnx_provider: str = "auto"  # 'auto', 'tensorrt', 'cuda', 'cpu', 'openvino'
    
    @staticmethod
    def from_env() -> "GroundingDinoConfig":
        """Load configuration from environment variables."""
        # Allow override for Grounding DINO device (default to CUDA for performance)
        dino_cuda_available = os.getenv(CUDA_AVAILABLE_ENV, TRUE_STR).lower() == TRUE_STR
        cpu_only = not dino_cuda_available

        device = "cuda" if dino_cuda_available else "cpu"

        return GroundingDinoConfig(
            config_path=os.getenv("DINO_CONFIG_PATH", "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            model_path=os.getenv("DINO_MODEL_PATH", "weights/groundingdino_swint_ogc.pth"),
            text_prompt=os.getenv("DINO_TEXT_PROMPT", DEFAULT_TEXT_PROMPT),
            box_threshold=float(os.getenv("DINO_BOX_THRESHOLD", "0.35")),
            text_threshold=float(os.getenv("DINO_TEXT_THRESHOLD", "0.25")),
            max_detections=int(os.getenv("DINO_MAX_DETECTIONS", "100")),
            min_box_size=int(os.getenv("DINO_MIN_BOX_SIZE", "50")),
            device=device,
            cpu_only=cpu_only,
            nms_threshold=float(os.getenv("DINO_NMS_THRESHOLD", "0.45")),
            auto_download=os.getenv("DINO_AUTO_DOWNLOAD", TRUE_STR).lower() == TRUE_STR,
            max_detection_dimension=int(os.getenv("DINO_MAX_DETECTION_DIMENSION", "640")),  # Optimizado para velocidad
            use_compile=os.getenv("DINO_USE_COMPILE", FALSE_STR).lower() == TRUE_STR,  # Default false
            onnx_export_path=os.getenv("DINO_ONNX_PATH", "models/grounding_dino_optimized.onnx"),
            use_onnx=os.getenv("DINO_USE_ONNX", FALSE_STR).lower() == TRUE_STR,
            onnx_provider=os.getenv("DINO_ONNX_PROVIDER", "auto"),
        )


# Default configuration
DEFAULT_DINO_CONFIG = GroundingDinoConfig()


# Fast configuration (lower thresholds, fewer detections)
FAST_DINO_CONFIG = GroundingDinoConfig(
    text_prompt="object . thing . item",
    box_threshold=0.40,
    text_threshold=0.30,
    max_detections=50,
    min_box_size=40,
)


# Balanced configuration (good balance)
BALANCED_DINO_CONFIG = GroundingDinoConfig(
    text_prompt=DEFAULT_TEXT_PROMPT,
    box_threshold=0.35,
    text_threshold=0.25,
    max_detections=100,
    min_box_size=50,
)


# High precision configuration (stricter thresholds)
HIGH_PRECISION_DINO_CONFIG = GroundingDinoConfig(
    text_prompt="artwork . painting . sculpture . piece . object",
    box_threshold=0.45,
    text_threshold=0.35,
    max_detections=100,
    min_box_size=60,
)

advanced_text_prompt = (
    "oil painting . watercolor painting . acrylic painting . portrait painting . "
    "landscape painting . abstract painting . modern painting . "
    "marble sculpture . bronze sculpture . classical sculpture . modern sculpture . statue . "
    "canvas . art piece . installation art . drawing . sketch"
)
ARTWORK_DINO_ADVANCED_CONFIG = GroundingDinoConfig(
    text_prompt=advanced_text_prompt,
    box_threshold=0.55,  # Very strict
    text_threshold=0.40,  # Very strict
    max_detections=30,  # Only top detections
    min_box_size=100,  # Only substantial objects
)

# Artwork-specific configuration
ARTWORK_DINO_CONFIG = GroundingDinoConfig(
    text_prompt=advanced_text_prompt,
    box_threshold=0.55,  # Very strict
    text_threshold=0.40,  # Very strict
    max_detections=30,  # Only top detections
    min_box_size=100,  # Only substantial objects
)
