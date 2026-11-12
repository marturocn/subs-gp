"""
DINOv3 Model Configuration
Configure DINOv3 settings for image embedding generation
"""
from dataclasses import dataclass
from typing import Literal
import os

# Boolean conversion constants
TRUE_STR = "true"
FALSE_STR = "false"


@dataclass
class DINOv3Config:
    """
    Configuration for DINOv3 model loading and inference.
    
    DINOv3 is a vision transformer model that generates high-quality
    image embeddings for visual similarity search.
    
    Attributes:
        model_name: HuggingFace model name
        device: Device to use ('cpu' or 'cuda')
        use_auth_token: HuggingFace auth token for model access
        embedding_dim: Expected embedding dimension
        normalize_embeddings: Whether to L2-normalize embeddings
        use_better_transformer: Enable BetterTransformer optimization
        use_compile: Enable torch.compile optimization
    """
    
    # Model configuration
    model_name: str = "facebook/dinov2-large"
    
    # Device configuration
    device: Literal["cpu", "cuda"] = "cuda"
    
    # Authentication
    use_auth_token: str = ""
    
    # Embedding settings
    normalize_embeddings: bool = True
    
    # Performance settings
    batch_size: int = 1
    num_workers: int = 0
    
    # Optimization settings
    use_better_transformer: bool = True
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"  # 'default', 'reduce-overhead', 'max-autotune'
    use_half_precision: bool = True
    
    # ONNX export settings
    onnx_export_path: str = "models/dinov3_optimized.onnx"
    use_onnx: bool = False  # Use ONNX model for inference instead of PyTorch
    onnx_provider: str = "auto"  # 'auto', 'tensorrt', 'cuda', 'cpu', 'openvino'
    use_fp32_export: bool = True  # Use FP32 for ONNX export (better quality, slower) 
    
    @staticmethod
    def from_env() -> "DINOv3Config":
        """Load configuration from environment variables."""
        # Check CUDA availability
        cuda_available = os.getenv("CUDA_AVAILABLE", TRUE_STR).lower() == TRUE_STR

        device = "cuda" if cuda_available else "cpu"
        
        # Get HuggingFace token from environment
        hf_token = (
            os.getenv("HF_AUTH_TOKEN") or 
            ""
        )
        
        model_name = os.getenv("DINOV3_MODEL_NAME", "facebook/dinov2-large")
        
        
        return DINOv3Config(
            model_name=model_name,
            device=device,
            use_auth_token=hf_token,
            normalize_embeddings=os.getenv("DINOV3_NORMALIZE", TRUE_STR).lower() == TRUE_STR,
            batch_size=int(os.getenv("DINOV3_BATCH_SIZE", "1")),
            num_workers=int(os.getenv("DINOV3_NUM_WORKERS", "0")),
            use_better_transformer=os.getenv("DINOV3_USE_BETTER_TRANSFORMER", TRUE_STR).lower() == TRUE_STR,
            use_compile=os.getenv("DINOV3_USE_COMPILE", TRUE_STR).lower() == TRUE_STR,
            compile_mode=os.getenv("DINOV3_COMPILE_MODE", "reduce-overhead"),
            use_half_precision=os.getenv("DINOV3_USE_HALF_PRECISION", TRUE_STR).lower() == TRUE_STR,
            onnx_export_path=os.getenv("DINOV3_ONNX_PATH", "models/dinov3_optimized.onnx"),
            use_onnx=os.getenv("DINOV3_USE_ONNX", FALSE_STR).lower() == TRUE_STR,  # Default to False for better performance
            onnx_provider=os.getenv("DINOV3_ONNX_PROVIDER", "auto"),
            use_fp32_export=os.getenv("DINOV3_USE_FP32_EXPORT", TRUE_STR).lower() == TRUE_STR,  # Default to FP32 for quality
        )


# Default configuration - DINOv2 Large
DEFAULT_DINOV3_CONFIG = DINOv3Config()


# Base model configuration (faster, smaller)
BASE_DINOV3_CONFIG = DINOv3Config(
    model_name="facebook/dinov2-base",
)


# Small model configuration (fastest, smallest)
SMALL_DINOV3_CONFIG = DINOv3Config(
    model_name="facebook/dinov2-small",
)


# Giant model configuration (best quality, slowest)
GIANT_DINOV3_CONFIG = DINOv3Config(
    model_name="facebook/dinov2-giant",
)
