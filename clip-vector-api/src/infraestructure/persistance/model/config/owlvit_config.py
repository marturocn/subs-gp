"""
OWL-ViT v2 Model Configuration
Configure OWL-ViT v2 settings for open-vocabulary object detection
"""
from dataclasses import dataclass, field
from typing import Literal, List, Optional
import os

# Boolean conversion constants
TRUE_STR = "true"
FALSE_STR = "false"


@dataclass
class OwlvitConfig:
    """
    Configuration for OWL-ViT v2 model loading and inference.
    
    OWL-ViT v2 is an open-vocabulary object detector that uses text prompts
    to detect arbitrary objects without being limited to predefined classes.
    
    Attributes:
        model_name: Hugging Face model name (e.g., "google/owlv2-large-patch14-ensemble")
        text_queries: Default text queries for detection (list of object names)
        score_threshold: Minimum confidence for detection (0.0 - 1.0)
        device: Device to use ('cpu' or 'cuda')
        max_detections: Maximum number of detections per image
        min_box_size: Minimum bounding box size (width or height in pixels)
    """
    
    # Model configuration
    model_name: str = "google/owlv2-large-patch14-ensemble"
    
    # Text queries for detection (OWL-ViT v2 supports max 16 queries per batch)
    # Keep default queries limited to avoid tensor size mismatch errors
    text_queries: List[str] = field(default_factory=lambda: [
        "object", "thing", "item", "artwork", "piece",
        "painting", "sculpture", "canvas"
    ])
    
    # Detection parameters
    score_threshold: float = 0.1  # Minimum detection confidence (0.0 - 1.0)
    max_detections: int = 100  # Maximum detections per image
    min_box_size: int = 50  # Minimum box size (pixels)
    
    # Device configuration
    device: Literal["cpu", "cuda"] = "cuda"  # Default to CUDA for performance
    cpu_only: bool = False  # If True, force CPU-only mode
    
    # Advanced options
    nms_threshold: float = 0.3  # Non-maximum suppression threshold
    use_fp16: bool = True  # Use FP16 for faster inference on GPU
    
    # Image preprocessing
    max_detection_dimension: int = 1024  # Maximum image dimension for detection
    
    # Warmup configuration
    warmup_iterations: int =5  # Number of warmup iterations
    warmup_image_size: int = 256  # Size of warmup image
    
    # ONNX export settings (for future optimization)
    onnx_export_path: str = "models/owlvit_optimized.onnx"
    use_onnx: bool = False  # Use ONNX model for inference
    onnx_provider: str = "auto"  # 'auto', 'tensorrt', 'cuda', 'cpu'
    
    @staticmethod
    def from_env() -> "OwlvitConfig":
        """Load configuration from environment variables."""
        cuda_available = os.getenv("CUDA_AVAILABLE", TRUE_STR).lower() == TRUE_STR
        cpu_only = not cuda_available
        device = "cuda" if cuda_available else "cpu"
        
        # Parse text queries from environment (comma-separated)
        text_queries_str = os.getenv(
            "OWLVIT_TEXT_QUERIES", 
            "object, thing, item, artwork, piece, plate, book, cup"
        )
        text_queries = [q.strip() for q in text_queries_str.split(",")]
        
        return OwlvitConfig(
            model_name=os.getenv("OWLVIT_MODEL_NAME", "google/owlv2-large-patch14-ensemble"),
            text_queries=text_queries,
            score_threshold=float(os.getenv("OWLVIT_SCORE_THRESHOLD", "0.1")),
            max_detections=int(os.getenv("OWLVIT_MAX_DETECTIONS", "100")),
            min_box_size=int(os.getenv("OWLVIT_MIN_BOX_SIZE", "50")),
            device=device,
            cpu_only=cpu_only,
            nms_threshold=float(os.getenv("OWLVIT_NMS_THRESHOLD", "0.3")),
            use_fp16=os.getenv("OWLVIT_USE_FP16", TRUE_STR).lower() == TRUE_STR,
            max_detection_dimension=int(os.getenv("OWLVIT_MAX_DETECTION_DIMENSION", "1024")),
            warmup_iterations=int(os.getenv("OWLVIT_WARMUP_ITERATIONS", "2")),
            warmup_image_size=int(os.getenv("OWLVIT_WARMUP_IMAGE_SIZE", "256")),
            onnx_export_path=os.getenv("OWLVIT_ONNX_PATH", "models/owlvit_optimized.onnx"),
            use_onnx=os.getenv("OWLVIT_USE_ONNX", FALSE_STR).lower() == TRUE_STR,
            onnx_provider=os.getenv("OWLVIT_ONNX_PROVIDER", "auto"),
        )


# Default configuration
DEFAULT_OWLVIT_CONFIG = OwlvitConfig()


# Fast configuration (lower thresholds, fewer detections)
FAST_OWLVIT_CONFIG = OwlvitConfig(
    text_queries=["object", "thing", "item"],
    score_threshold=0.15,
    max_detections=50,
    min_box_size=40,
)


# Balanced configuration (good balance)
BALANCED_OWLVIT_CONFIG = OwlvitConfig(
    text_queries=["object", "thing", "item", "artwork", "piece"],
    score_threshold=0.1,
    max_detections=100,
    min_box_size=50,
)


# High precision configuration (stricter thresholds)
HIGH_PRECISION_OWLVIT_CONFIG = OwlvitConfig(
    text_queries=["artwork", "painting", "sculpture", "piece", "object"],
    score_threshold=0.2,
    max_detections=100,
    min_box_size=60,
)


# Artwork-specific configuration
ARTWORK_OWLVIT_CONFIG = OwlvitConfig(
    text_queries=[
        "oil painting", "watercolor painting", "acrylic painting", 
        "portrait painting", "landscape painting", "abstract painting",
        "marble sculpture", "bronze sculpture", "classical sculpture",
        "canvas", "art piece", "installation art", "drawing", "sketch"
    ],
    score_threshold=0.15,
    max_detections=30,
    min_box_size=100,
)
