"""
CLIP Model Repository Implementation - Infrastructure Layer
"""
import logging
from typing import List, Optional
from io import BytesIO

from src.domain.model.repositories.clip_model_repository import ClipModelRepository
from src.infraestructure.persistance.model.config.clip_config import (
    ClipModelConfig,
    DEFAULT_CONFIG,
)


logger = logging.getLogger(__name__)


class ClipModelRepositoryImpl(ClipModelRepository):
    """
    Real implementation of CLIP service using open_clip.
    Follows the repository pattern with Observable responses.
    """
    
    def __init__(self, config: Optional[ClipModelConfig] = None):
        self.config = config or ClipModelConfig.from_env()
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = None  # Will be set during model loading
        self._load_model()
    
    def _load_model(self) -> None:
        """Load CLIP model with optimized configuration. Falls back to mock if real model fails."""
        try:
            import os
            import torch

            # Prepare cache directories and env vars for open_clip/huggingface
            try:
                self._prepare_clip_cache()
            except Exception:
                logger.debug("_prepare_clip_cache failed or not applicable")

            import open_clip
            print("EVA02-CLIP-bigE-14-plus" in open_clip.list_models())

            logger.info(f"Loading CLIP model: {self.config.model_name} with {self.config.pretrained}...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.model_name, 
                pretrained=self.config.pretrained
            )
            
            # Convert to evaluation mode and disable gradients for inference
            model.eval()
            
            # CPU optimizations
            if self.config.optimize_for_inference:
                # Disable gradient computation globally for inference
                for param in model.parameters():
                    param.requires_grad = False
                
                # Enable CPU-specific optimizations
                torch.set_num_threads(torch.get_num_threads())  # Use all available CPU threads
                torch.set_flush_denormal(True)  # Improve numerical stability
            
            # Store model and preprocess transforms
            self.model = model
            self.preprocess = preprocess  # This is the actual transform function
            
            # Set device (always CPU for this configuration)
            self.device = torch.device("cpu")
            
            # Move model to CPU and set precision
            if hasattr(self.model, 'to'):
                dtype = torch.float64 if self.config.use_fp64 else torch.float32
                self.model = self.model.to(self.device, dtype=dtype)
            
            logger.info(f"✅ CLIP model loaded successfully on device: {self.device}")
            logger.info(f"📊 Model: {self.config.model_name}, Vector dimension: {self.config.get_vector_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to load real CLIP model: {e}")
            logger.info("Using mock implementation")
            self.model = None
            self.preprocess = None
            self.device = None

    def _prepare_clip_cache(self) -> None:
        """Prepare cache directories from CLIP_VENV or .env so model downloads go there."""
        import os

        clip_venv = os.getenv('CLIP_VENV')
        if not clip_venv:
            # try dotenv first
            try:
                from dotenv import load_dotenv, find_dotenv

                dotenv_path = find_dotenv()
                if dotenv_path:
                    load_dotenv(dotenv_path)
                    clip_venv = os.getenv('CLIP_VENV')
            except Exception:
                clip_venv = None

        # last-resort manual parse of project .env
        if not clip_venv:
            clip_venv = self._find_clip_venv_from_files()

        if not clip_venv:
            return

        # Ensure directory exists
        try:
            os.makedirs(clip_venv, exist_ok=True)
        except Exception:
            logger.warning(f"Could not create CLIP_VENV dir: {clip_venv}")

        os.environ.setdefault('HF_HOME', clip_venv)
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(clip_venv, 'hub'))
        os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(clip_venv, 'transformers'))
        logger.info(f"Using CLIP cache directory from CLIP_VENV: {clip_venv}")

    def _find_clip_venv_from_files(self) -> str | None:
        """Search project files for CLIP_VENV definition (.env)."""
        import os

        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            project_env = os.path.join(project_root, '.env')
            if os.path.exists(project_env):
                with open(project_env, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('CLIP_VENV='):
                            return line.split('=', 1)[1].strip().strip('"').strip("'")
        except Exception:
            return None

        return None

    async def encode_image(self, image_data: BytesIO) -> List[float]:
        """
        Convert an image to a vector using CLIP model.
        Async wrapper that calls the sync implementation.
        """
        return self.encode_image_sync(image_data)
    
    def encode_image_sync(self, image_data: BytesIO) -> List[float]:
        """
        Convert an image to a vector using CLIP model (synchronous).
        """
        try:
            # Reset stream position
            image_data.seek(0)
            if self.model is None:
                raise RuntimeError("CLIP model no disponible")
            return self._generate_real_vector(image_data)
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise RuntimeError(f"Failed to encode image: {e}")
    
    def _generate_real_vector(self, image_data: BytesIO) -> List[float]:
        """Generate vector using real CLIP model."""
        # Preprocess image to tensor
        image_tensor = self._preprocess_image(image_data)
        # Generate embeddings with model
        vector = self._encode_with_model(image_tensor)
        logger.info(f"Generated real CLIP vector of dimension: {len(vector)}")
        return vector
    
    def _preprocess_image(self, image_data: BytesIO):
        """
        Preprocess image data to tensor format with high precision.
        Ensures no distortion or unnecessary processing.
        Uses configuration for optimal quality.
        """
        import torch
        from PIL import Image
        
        # Load image with proper error handling
        try:
            image = Image.open(image_data)
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
        
        # Convert to RGB without quality loss
        image = self._ensure_rgb_format(image)
        
        # Apply preprocessing transform
        if self.preprocess is None:
            raise ValueError("Preprocessor not available")
        
        # The preprocess from open_clip can be a tuple or callable
        # Extract the actual transform and apply it
        try:
            # Handle tuple format (model, _, preprocess) from create_model_and_transforms
            transform = self.preprocess[0] if isinstance(self.preprocess, tuple) else self.preprocess
            image_tensor = transform(image)
        except Exception as e:
            raise ValueError(f"Failed to apply preprocessing transform: {e}")
        
        return self._process_tensor(image_tensor)
    
    def _ensure_rgb_format(self, image):
        """Ensure image is in RGB format, handling transparency properly."""
        from PIL import Image
        
        if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        
        if image.mode == 'RGB':
            return image
            
        if image.mode == 'RGBA':
            # Handle transparency properly by compositing on white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
            return background
            
        # Convert palette/L images to RGB
        return image.convert('RGB')

    def _process_tensor(self, image_tensor):
        """Validate and prepare tensor for model input."""
        import torch
        
        # Validate tensor output
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Transform did not return a PyTorch tensor")
        
        # Ensure proper shape (C, H, W)
        if image_tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W), got {image_tensor.dim()}D")
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to CPU and set precision based on config
        if self.device is not None:
            dtype = torch.float64 if self.config.use_fp64 else torch.float32
            image_tensor = image_tensor.to(self.device, dtype=dtype)
        
        return image_tensor
    
    def _encode_with_model(self, image_tensor):
        """
        Encode image tensor to vector using CLIP model.
        Uses proper normalization for precise embeddings.
        Optimized for CPU inference.
        """
        import torch
        
        if self.model is None or not hasattr(self.model, 'encode_image'):
            raise ValueError("Model not available")
        
        # Generate embeddings with no gradient computation (CPU only)
        with torch.no_grad():
            # Set deterministic mode for reproducible results
            if self.config.optimize_for_inference:
                torch.use_deterministic_algorithms(False)  # Allow faster operations
            
            # Encode image - produces feature vector
            image_features = self.model.encode_image(image_tensor) # pyright: ignore[reportCallIssue]
            
            # Ensure consistent precision
            target_dtype = torch.float64 if self.config.use_fp64 else torch.float32
            if image_features.dtype != target_dtype:
                image_features = image_features.to(target_dtype)
            
            # L2 normalization - essential for CLIP embeddings
            # This ensures all vectors have unit length for proper similarity computation
            if self.config.normalize_embeddings:
                feature_norm = torch.linalg.norm(image_features, ord=2, dim=-1, keepdim=True)
                
                # Avoid division by zero with epsilon
                epsilon = 1e-12 if self.config.use_fp64 else 1e-8
                feature_norm = torch.clamp(feature_norm, min=epsilon)
                image_features = image_features / feature_norm
            
            # Convert to numpy array
            vector = image_features.squeeze(0).cpu().numpy()
            
            # Ensure the vector has proper shape and convert to list
            if vector.ndim != 1:
                raise ValueError(f"Expected 1D vector, got shape {vector.shape}")
            
            vector_list = vector.tolist()
        
        return vector_list
    

