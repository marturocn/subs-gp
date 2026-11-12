"""
CLIP Model Configuration
Configure model settings for optimal precision and performance
"""
from dataclasses import dataclass
from typing import Literal
import os

# Boolean conversion constants
TRUE_STR = "true"
FALSE_STR = "false"


@dataclass
class ClipModelConfig:
    """
    Configuration for CLIP model loading and inference.
    
    Attributes:
        model_name: Name of the CLIP model architecture
        pretrained: Pretrained weights to use
        device: Device to use ('auto', 'cuda', 'cpu')
        use_fp16: Use float16 precision (only for CUDA)
        image_resolution: Input image resolution (higher = more precise)
        enable_amp: Enable automatic mixed precision (CUDA only)
    """
    
    # Model selection
    model_name: str = "ViT-L-14"  # Options: ViT-L-14, ViT-L-14-336, ViT-H-14, ViT-g-14
    pretrained: str = "laion2b_s32b_b82k"  # laion2b_s32b_b82k, laion400m_e32, etc.
    
    # Device configuration (CPU only for this deployment)
    device: Literal["cpu"] = "cpu"
    
    # Precision settings (CPU optimized)
    use_fp64: bool = False  # Use float64 for maximum precision (slower on CPU)
    optimize_for_inference: bool = True  # Enable CPU optimizations
    
    # Image preprocessing
    interpolation_mode: str = "bicubic"  # bicubic, bilinear, lanczos
    preserve_aspect_ratio: bool = False  # Experimental: preserve aspect ratio
    
    # Normalization
    normalize_embeddings: bool = True  # L2 normalization
    
    @staticmethod
    def from_env() -> "ClipModelConfig":
        """Load configuration from environment variables."""
        return ClipModelConfig(
            model_name=os.getenv("CLIP_MODEL_NAME", "ViT-L-14"),
            pretrained=os.getenv("CLIP_PRETRAINED", "laion2b_s32b_b82k"),
            device="cpu",  # Always use CPU
            use_fp64=os.getenv("CLIP_USE_FP64", FALSE_STR).lower() == TRUE_STR,
            optimize_for_inference=os.getenv("CLIP_OPTIMIZE_INFERENCE", TRUE_STR).lower() == TRUE_STR,
            interpolation_mode=os.getenv("CLIP_INTERPOLATION", "bicubic"),
        )
    
    def get_vector_dimension(self) -> int:
        """Get the vector dimension for this model."""
        # Mapping of model names to their output dimensions
        dimensions = {
            "ViT-B-32": 512,
            "ViT-B-16": 512,
            "ViT-L-14": 768,
            "ViT-L-14-336": 768,
            "ViT-H-14": 1024,
            "ViT-g-14": 1024,
        }
        return dimensions.get(self.model_name, 768)
    
    def get_image_size(self) -> int:
        """Get the image size for this model."""
        if "336" in self.model_name:
            return 336
        return 224


# Default configuration
DEFAULT_CONFIG = ClipModelConfig()


# High precision configuration (slower, more accurate) - CPU optimized
HIGH_PRECISION_CONFIG = ClipModelConfig(
    model_name="ViT-L-14-336",  # Higher resolution (336x336)
    pretrained="laion2b_s32b_b82k",
    device="cpu",
    interpolation_mode="bicubic",
    normalize_embeddings=True,
    optimize_for_inference=True,
)


# Fast configuration (faster, good accuracy) - CPU optimized
FAST_CONFIG = ClipModelConfig(
    model_name="ViT-L-14",
    pretrained="laion2b_s32b_b82k",
    device="cpu",
    normalize_embeddings=True,
    optimize_for_inference=True,
)


# Maximum precision configuration (slowest, best accuracy) - CPU optimized
MAX_PRECISION_CONFIG = ClipModelConfig(
    model_name="ViT-H-14",  # Largest model (1024 dims)
    pretrained="laion2b_s32b_b82k",
    device="cpu",
    interpolation_mode="bicubic",
    normalize_embeddings=True,
    use_fp64=False,  # float64 puede ser muy lento en CPU
    optimize_for_inference=True,
)
