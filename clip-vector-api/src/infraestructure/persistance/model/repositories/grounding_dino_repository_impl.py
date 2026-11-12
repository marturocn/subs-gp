"""
Grounding DINO Model Repository Implementation - Infrastructure Layer
"""
import logging
import time
import torch
from torch.amp.autocast_mode import autocast
import numpy as np
import cv2
from typing import Optional
from io import BytesIO
from PIL import Image
import tempfile
import warnings
from pathlib import Path

from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.models.detection_models import DetectionResult, BoundingBox
from src.infraestructure.persistance.model.config.grounding_dino_config import GroundingDinoConfig


logger = logging.getLogger(__name__)

# Suprimir warnings de torch.cuda.amp deprecation
warnings.filterwarnings('ignore', category=FutureWarning, module='groundingdino')


class GroundingDinoRepositoryImpl(GroundingDinoRepository):
    """
    Real implementation of Grounding DINO service.
    Follows the repository pattern with synchronous and asynchronous methods.
    Supports open-set object detection using text prompts.
    Supports ONNX inference for optimized performance.
    """
    
    def __init__(self, config: Optional[GroundingDinoConfig] = None):
        """
        Initialize Grounding DINO model repository.
        
        Args:
            config: Grounding DINO model configuration. If None, loads from environment.
        """
        self.config = config or GroundingDinoConfig.from_env()
        self.model = None
        self.device = None
        self._warm_up_done = False
        
        # ONNX support
        self.onnx_session = None
        self.use_onnx = self.config.use_onnx
        
        # Optimization settings
        self.use_optimizer = getattr(self.config, 'use_optimizer', True)
        self.optimize_compile_mode = getattr(self.config, 'optimize_compile_mode', None)  # Disabled - causes hangs
        
        # NOTE: do not load model in constructor to avoid automatic downloads at import/instantiation.
        # Model will be loaded lazily on first detection call via _ensure_model_loaded().
    
    def _download_model_files(self) -> tuple[str, str]:
        """
        Download Grounding DINO model files automatically if they don't exist.
        
        Returns:
            Tuple of (config_path, model_path)
        """
        import os
        import urllib.request
        from pathlib import Path

        # Resolve file paths from config (these may be absolute or relative)
        config_path = Path(self.config.config_path)
        model_path = Path(self.config.model_path)

        # Determine directories from the provided paths and create them
        config_dir = config_path.parent
        weights_dir = model_path.parent
        config_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Download config file if it doesn't exist
        if not config_path.exists():
            if not getattr(self.config, "auto_download", True):
                raise RuntimeError(
                    f"Grounding DINO config file not found at {config_path}. "
                    "Auto-download is disabled (DINO_AUTO_DOWNLOAD=false). "
                    "Please download the config file manually from: "
                    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py "
                    f"and place it at {config_path}"
                )

            logger.info("📥 Downloading Grounding DINO config file...")
            config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            try:
                urllib.request.urlretrieve(config_url, str(config_path))
                logger.info(f"✅ Config file downloaded to {config_path}")
            except Exception as e:
                logger.error(f"Failed to download config file: {e}")
                raise RuntimeError(f"Failed to download config file: {e}")
        
        # Download model weights if they don't exist
        if not model_path.exists():
            if not getattr(self.config, "auto_download", True):
                raise RuntimeError(
                    f"Grounding DINO model weights not found at {model_path}. "
                    "Auto-download is disabled (DINO_AUTO_DOWNLOAD=false). "
                    "Please download the weights manually from: "
                    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth "
                    f"and place it at {model_path}"
                )

            logger.info("📥 Downloading Grounding DINO model weights (~700MB, this may take a while)...")
            model_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            try:
                # Show download progress
                def report_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100 / total_size)
                    if block_num % 100 == 0:  # Update every 100 blocks
                        logger.info(f"⏳ Downloading: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
                
                urllib.request.urlretrieve(model_url, str(model_path), reporthook=report_progress)
                logger.info(f"✅ Model weights downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model weights: {e}")
                raise RuntimeError(f"Failed to download model weights: {e}")
        else:
            logger.info("✅ Using existing model files")
        
        return str(config_path), str(model_path)
    
    def _check_checkpoint_format(self) -> str:
        """
        Check which format of model to use: PyTorch checkpoint or original.
        
        Returns:
            "pytorch_checkpoint", "onnx", or "pytorch_original"
        """
        import json
        
        onnx_path = Path(self.config.onnx_export_path)
        
        # Check for new PyTorch checkpoint format with metadata
        metadata_path = onnx_path.with_stem(onnx_path.stem).with_suffix(".json")
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                if metadata.get("format") == "pytorch_checkpoint":
                    logger.info(f"✅ Found PyTorch checkpoint metadata: {metadata_path}")
                    logger.info(f"   Model: {metadata.get('model_type')}")
                    logger.info(f"   Precision: {metadata.get('precision')}")
                    return "pytorch_checkpoint"
            except Exception as e:
                logger.warning(f"Could not load metadata from {metadata_path}: {e}")
        
        # Check for ONNX format (legacy)
        if onnx_path.exists() and onnx_path.suffix == ".onnx":
            return "onnx"
        
        # Default to original PyTorch model loading
        return "pytorch_original"
    
    def _load_model(self) -> None:
        """
        Load Grounding DINO model with format detection.
        Supports: PyTorch checkpoints, ONNX models, or original PyTorch loading.
        """
        # Determine which format to use
        model_format = self._check_checkpoint_format()
        
        if model_format == "pytorch_checkpoint":
            logger.info("� Loading from PyTorch checkpoint (GPU-optimized)...")
            try:
                self._load_pytorch_checkpoint()
                return
            except Exception as e:
                logger.error(f"Failed to load PyTorch checkpoint: {e}")
                logger.error("Falling back to original PyTorch model...")
        
        if model_format == "onnx" and self.use_onnx:
            logger.info("🚀 Loading from ONNX model...")
            try:
                self._load_onnx_model()
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to load ONNX model: {e}")
                logger.warning("    Falling back to PyTorch model for full functionality")
        
        # Default: Load original PyTorch model
        logger.info("📦 Loading original PyTorch model...")
        self._load_pytorch_original()
    
    def _configure_device(self) -> None:
        """Configure device settings (CPU/CUDA)."""
        if getattr(self.config, 'cpu_only', False):
            self.device = "cpu"
            logger.info("CPU-only mode enabled")
        elif getattr(self.config, 'device', 'cuda') == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("CUDA optimizations enabled")
        else:
            self.device = "cpu"
        
        logger.info(f"Device: {self.device}")
    
    def _load_checkpoint_metadata(self) -> tuple:
        """Load checkpoint and metadata files."""
        import json
        
        onnx_path = Path(self.config.onnx_export_path)
        checkpoint_path = onnx_path.with_stem(onnx_path.stem).with_suffix(".pt")
        metadata_path = onnx_path.with_stem(onnx_path.stem).with_suffix(".json")
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        
        logger.info(f"📂 Checkpoint: {checkpoint_path}")
        logger.info(f"📋 Metadata: {metadata_path}")
        
        return checkpoint_path, metadata
    
    def _extract_state_dict(self, checkpoint: dict) -> dict:
        """Extract state_dict from different checkpoint formats."""
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            return checkpoint
    
    def _apply_precision_settings(self, metadata: dict) -> None:
        """Apply precision settings from metadata."""
        self.use_fp16 = False
        if self.device == "cuda" and metadata.get("precision") == "fp16":
            logger.info("📊 Using FP16 precision from checkpoint")
            self.use_fp16 = True

    def _load_pytorch_checkpoint(self) -> None:
        """
        Load Grounding DINO from GPU-optimized PyTorch checkpoint.
        This uses the checkpoint generated by the dynamic optimization system.
        """
        try:
            from groundingdino.util.inference import load_model, load_image, predict, annotate  # type: ignore
            
            # Configure device
            self._configure_device()
            
            # Load checkpoint and metadata
            checkpoint_path, metadata = self._load_checkpoint_metadata()
            
            # Download original model files if needed (for architecture)
            config_path, model_path = self._download_model_files()
            
            # Load base model architecture
            logger.info("Loading base model architecture...")
            self.model = load_model(
                model_config_path=config_path,
                model_checkpoint_path=model_path,
                device=self.device # pyright: ignore[reportArgumentType]
            )
            
            # Load optimized checkpoint if it exists
            if checkpoint_path.exists():
                logger.info("Loading optimized weights from checkpoint...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                state_dict = self._extract_state_dict(checkpoint)
                self.model.load_state_dict(state_dict)
                logger.info("✅ Checkpoint weights loaded")
            
            # Prepare model for inference
            self.model.eval()
            self.model.to(self.device)
            
            # Apply precision settings
            self._apply_precision_settings(metadata)
            
            # Apply PyTorch-native optimizations (torch.compile, FlashAttention, text cache)
            self._apply_optimizations()
            
            # Store utility functions
            self.load_image = load_image
            self.predict = predict
            self.annotate = annotate
            
            logger.info("✅ Grounding DINO checkpoint loaded successfully")
            logger.info(f"📊 Box threshold: {self.config.box_threshold}")
            logger.info(f"📊 Text threshold: {self.config.text_threshold}")
            
            # Warmup
            self._warmup_model()
            
        except ImportError as e:
            logger.error(f"Failed to import groundingdino: {e}")
            raise RuntimeError(f"Failed to import groundingdino: {e}")
        except Exception as e:
            logger.error(f"Failed to load PyTorch checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def _configure_device_original(self) -> None:
        """Configure device settings for original PyTorch loading."""
        if getattr(self.config, 'cpu_only', False):
            logger.info("Grounding DINO configured for CPU-only mode via config.cpu_only")
            self.device = "cpu"
        elif getattr(self.config, 'device', 'cpu') == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            if getattr(self.config, 'device', 'cpu') == "cuda":
                logger.warning("Requested device 'cuda' is not available; falling back to 'cpu'")
            self.device = "cpu"
        
        print(f"Using device: {self.device} in GroundingDinoRepositoryImpl")
    
    def _setup_model_inference(self) -> None:
        """Setup model for inference (eval mode, device placement, FP16)."""
        if self.model is None:
            return
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Move model to the chosen device if not already
        try:
            self.model.to(self.device)
        except Exception:
            logger.debug("Model does not support .to(device) or it's already on device")
        
        # Decide whether to use FP16
        self.use_fp16 = False
        try:
            requested_fp16 = bool(getattr(self.config, 'use_fp16', True))
        except Exception:
            requested_fp16 = True
        
        if self.device == 'cuda' and requested_fp16:
            self.use_fp16 = True
            logger.info("Configured to use automatic mixed precision (autocast) for inference")
    
    def _apply_optimizations(self) -> None:
        """
        Apply PyTorch-native optimizations using GroundingDINOOptimizer.
        
        Optimizations include:
        - torch.compile for kernel fusion (20-50% speedup)
        - FlashAttention / SDPA for faster attention (2-4x on attention layers)
        - Text embedding cache for repetitive queries (30-50% speedup)
        """
        if not self.use_optimizer or self.model is None:
            logger.info("ℹ️  Skipping optimizations (use_optimizer=False or model not loaded)")
            return
        
        if self.device != "cuda":
            logger.info("ℹ️  Skipping optimizations (device is not CUDA)")
            return
        
        try:
            from src.infraestructure.persistance.model.utils import GroundingDINOOptimizer
            
            logger.info("🚀 Applying PyTorch-native optimizations...")
            
            # Apply optimizations (torch.compile, FlashAttention, text cache)
            self.model = GroundingDINOOptimizer.optimize(
                self.model,
                compile_mode=self.optimize_compile_mode,
                enable_flash_attention=True,
                enable_text_cache=True,
                text_cache_size=1000,
                fp16=getattr(self, 'use_fp16', False),
                device=self.device,
            )
            
            # Log optimization status
            status = GroundingDINOOptimizer.get_status(self.model)
            if status:
                logger.info(str(status))
            
        except ImportError as e:
            logger.warning(f"⚠️  Could not import GroundingDINOOptimizer: {e}")
            logger.warning("   Continuing without optimizations")
        except Exception as e:
            logger.warning(f"⚠️  Failed to apply optimizations: {e}")
            logger.warning("   Continuing without optimizations")
    
    def _load_pytorch_original(self) -> None:
        """
        Load Grounding DINO using original PyTorch loading from GitHub.
        This is the fallback when checkpoint is not available.
        """

        # Load PyTorch model (original implementation)
        try:
            from groundingdino.util.inference import load_model, load_image, predict, annotate # type: ignore
            
            logger.info("Loading Grounding DINO model...")
            
            # Configure device and download model files
            self._configure_device_original()
            config_path, model_path = self._download_model_files()
            
            # Load Grounding DINO model
            self.model = load_model(
                model_config_path=config_path,
                model_checkpoint_path=model_path,
                device=self.device # pyright: ignore[reportArgumentType]
            )
            
            # Setup model for inference
            self._setup_model_inference()
            
            # Apply PyTorch-native optimizations (torch.compile, FlashAttention, text cache)
            self._apply_optimizations()
            
            # Store the utility functions
            self.load_image = load_image
            self.predict = predict
            self.annotate = annotate
            
            logger.info(f"✅ Grounding DINO model loaded successfully on device: {self.device}")
            logger.info(f"📊 Box threshold: {self.config.box_threshold}, Text threshold: {self.config.text_threshold}")
            logger.info(f"🎯 Default text prompt: '{self.config.text_prompt}'")
            
            # Warmup del modelo
            self._warmup_model()
            
        except ImportError as e:
            logger.error(f"Failed to import groundingdino: {e}")
            logger.error("Make sure groundingdino is installed properly")
            raise RuntimeError(f"Failed to import groundingdino: {e}")
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO model: {e}")
            raise RuntimeError(f"Failed to load Grounding DINO model: {e}")
    
    def _load_onnx_model(self) -> None:
        """
        Load Grounding DINO model from ONNX format for optimized inference.
        Note: Grounding DINO ONNX export is complex due to text conditioning.
        This is a simplified version that may require custom preprocessing.
        """
        try:
            from src.infraestructure.persistance.model.utils.onnx_optimizer import (
                ONNXInferenceSession,
                ONNXOptimizer
            )
            
            logger.info(f"Loading ONNX model: {self.config.onnx_export_path}")
            logger.warning("⚠️  Note: Grounding DINO ONNX inference is experimental")
            logger.warning("    Consider using PyTorch model for full functionality")
            
            # Get providers based on config
            providers = ONNXOptimizer.get_onnx_providers(self.config.onnx_provider)
            
            # Load ONNX session
            self.onnx_session = ONNXInferenceSession(
                model_path=self.config.onnx_export_path,
                providers=providers,
            )
            
            # Set device info for logging
            self.device = "cpu"  # ONNX handles device internally
            
            logger.info("✅ Grounding DINO ONNX model loaded successfully")
            logger.info(f"📊 Box threshold: {self.config.box_threshold}, Text threshold: {self.config.text_threshold}")
            logger.info("⚡ Using optimized ONNX inference")
            
        except ImportError as e:
            logger.error(f"Failed to import ONNX dependencies: {e}")
            logger.error("Install with: pip install onnxruntime-gpu")
            raise RuntimeError(f"Failed to import ONNX dependencies: {e}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _warmup_model(self) -> None:
        """
        Warmup model with dummy inference to initialize CUDA kernels.
        This will perform a small number of safe dummy inferences to initialize CUDA kernels
        and internal buffers. Warmup is only attempted when the model is loaded and
        the device is CUDA. The routine is defensive: any failure will be logged and
        skipped so it never prevents real inference.
        """
        if self._warm_up_done or self.model is None:
            return
        
        # Skip warmup for ONNX models (they are already optimized)
        if self.use_onnx and self.onnx_session:
            logger.info("ℹ️  Skipping warmup for ONNX model (already optimized)")
            self._warm_up_done = True
            return

        # Only warmup on CUDA devices
        if self.device != 'cuda':
            logger.info("ℹ️  Skipping Grounding DINO warmup (device is not CUDA)")
            self._warm_up_done = True
            return

        # Perform a couple of small dummy inferences to initialize kernels and caches.
        try:
            import math

            warmup_iters = int(getattr(self.config, 'warmup_iterations', 5))
            warmup_size = int(getattr(self.config, 'warmup_image_size', 256))

            logger.info(f"🔥 Running Grounding DINO warmup: {warmup_iters} iters, size={warmup_size}")

            # Create a dummy image tensor in the expected [C,H,W] range normalized like the real pipeline
            # Use float16 when model was converted to half precision
            dtype = torch.float16 if getattr(self, 'use_fp16', False) else torch.float32
            dummy = torch.ones((3, warmup_size, warmup_size), dtype=dtype, device=self.device)
            # Normalize roughly to ImageNet range used in pipeline
            # Both float32 and float16 need normalization for the inference to work correctly
            dummy = (dummy - 0.485) / 0.229

            # Run warmup iterations with inference_mode and autocast when using fp16
            with torch.inference_mode():
                for i in range(warmup_iters):
                    try:
                        if getattr(self, 'use_fp16', False):
                            with autocast(device_type='cuda'):
                                _ = self.predict(
                                    model=self.model,
                                    image=dummy,
                                    caption=(getattr(self.config, 'text_prompt', '') or ''),
                                    box_threshold=self.config.box_threshold,
                                    text_threshold=self.config.text_threshold,
                                )
                        else:
                            _ = self.predict(
                                model=self.model,
                                image=dummy,
                                caption=(getattr(self.config, 'text_prompt', '') or ''),
                                box_threshold=self.config.box_threshold,
                                text_threshold=self.config.text_threshold,
                            )
                    except Exception as e:
                        # If any warmup iteration fails, log and continue (do not block real inference)
                        logger.debug(f"Warmup iteration {i} failed (ignored): {e}")

            logger.info("✅ Grounding DINO warmup complete")
        except Exception as e:
            logger.warning(f"Grounding DINO warmup failed or skipped: {e}")
        finally:
            self._warm_up_done = True
    
    async def detect_objects(
        self, 
        image_data: BytesIO, 
        text_prompt: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect objects in an image using Grounding DINO model (async).
        Async wrapper that calls the sync implementation.
        
        Args:
            image_data: The image data as BytesIO
            text_prompt: Text prompt for detection. If None, uses default from config.
            
        Returns:
            DetectionResult containing all detected objects
        """
        # Ensure model is loaded before running detection
        self._ensure_model_loaded()
        return self.detect_objects_sync(image_data, text_prompt)
    
    def detect_objects_sync(
        self, 
        image_data: BytesIO, 
        text_prompt: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect objects in an image using Grounding DINO model (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            text_prompt: Text prompt for detection (e.g., "person . car . dog").
                        If None, uses default prompt from config.
            
        Returns:
            DetectionResult containing all detected objects with bounding boxes
        """
        self._ensure_model_loaded()
        
        if self.use_onnx and self.onnx_session:
            logger.warning("⚠️  ONNX inference for Grounding DINO is experimental")
            logger.warning("    Falling back to PyTorch for best results")

        try:
            image_data.seek(0)
            if self.model is None:
                raise RuntimeError("Grounding DINO model not available")
            
            prompt = text_prompt or self.config.text_prompt
            image = Image.open(image_data)
            image = self._convert_image_to_rgb(image)
            image_width, image_height = image.size
            start_time = time.time()
            
            max_dim = getattr(self.config, 'max_detection_dimension', 1024)
            resized_image = self._resize_image_for_detection(image, max_dim)
            
            boxes, logits, phrases = self._detect_with_resized_image(resized_image, prompt)
            
            inference_time = (time.time() - start_time) * 1000
            bounding_boxes = self._convert_detections_to_bboxes(
                boxes, logits, phrases, image_width, image_height, 
                resized_image.size, max_dim
            )
            
            logger.info(f"⚡ Detected {len(bounding_boxes)} objects in {inference_time:.2f}ms")
            logger.debug(f"Used text prompt: '{prompt}'")
            
            return DetectionResult(
                bounding_boxes=bounding_boxes,
                image_width=image_width,
                image_height=image_height,
                inference_time=inference_time,
            )
            
        except Exception as e:
            logger.error(f"Error detecting objects with Grounding DINO: {e}")
            raise RuntimeError(f"Failed to detect objects: {e}")
    
    def _convert_image_to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB, handling RGBA and other formats."""
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
                image = background
            else:
                image = image.convert('RGB')
        return image
    
    def _resize_image_for_detection(self, image: Image.Image, max_dim: int) -> Image.Image:
        """Resize image if exceeds max_detection_dimension."""
        original_width, original_height = image.size
        if max(original_width, original_height) <= max_dim:
            return image
        
        scale = max_dim / float(max(original_width, original_height))
        new_w = int(original_width * scale)
        new_h = int(original_height * scale)
        
        resampling_attr = getattr(Image, 'Resampling', None)
        resample_filter = (resampling_attr.LANCZOS if (resampling_attr and hasattr(resampling_attr, 'LANCZOS'))
                          else getattr(Image, 'LANCZOS', None) or getattr(Image, 'NEAREST', 0))
        
        resized = image.resize((new_w, new_h), resample=resample_filter)
        logger.info(f"Resized image from {original_width}x{original_height} -> {new_w}x{new_h}")
        return resized
    
    def _detect_with_resized_image(self, resized_image: Image.Image, prompt: str) -> tuple:
        """Load image and run detection."""
        import tempfile
        import os
        from torchvision.transforms import functional as TF
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as tmp:
            resized_image.save(tmp, format='JPEG', quality=95, optimize=False)
            tmp_path = tmp.name
        
        try:
            image_data, _ = self.load_image(tmp_path)  # pyright: ignore[reportOptionalCall]
            image_tensor = self._prepare_tensor(image_data)
            
            with torch.inference_mode():
                if getattr(self, 'use_fp16', False) and self.device == 'cuda':
                    with autocast(device_type='cuda'):
                        boxes, logits, phrases = self.predict(  # pyright: ignore[reportOptionalCall]
                            model=self.model, image=image_tensor, caption=prompt,
                            box_threshold=self.config.box_threshold,
                            text_threshold=self.config.text_threshold,
                            device=self.device)
                else:
                    boxes, logits, phrases = self.predict(  # pyright: ignore[reportOptionalCall]
                        model=self.model, image=image_tensor, caption=prompt,
                        box_threshold=self.config.box_threshold,
                        text_threshold=self.config.text_threshold,
                        device=self.device)
            return boxes, logits, phrases
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _prepare_tensor(self, image_data) -> torch.Tensor:
        """Convert image data to normalized tensor."""
        from torchvision.transforms import functional as TF
        
        if isinstance(image_data, np.ndarray):
            image_data_copy = np.copy(image_data)
            image_tensor = TF.to_tensor(image_data_copy)
            image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif torch.is_tensor(image_data):
            image_tensor = image_data.squeeze(0) if image_data.dim() == 4 else image_data
        else:
            raise TypeError(f"Unexpected image type: {type(image_data)}")
        
        if self.device == "cuda" and not image_tensor.is_cuda:
            image_tensor = image_tensor.cuda(non_blocking=True)
        
        # Match model precision to avoid dtype mismatch
        if getattr(self, 'use_fp16', False):
            image_tensor = image_tensor.half()
        
        return image_tensor
    
    def _convert_detections_to_bboxes(self, boxes, logits, phrases, orig_w, orig_h, det_size, max_dim) -> list:
        """Convert detection outputs to BoundingBox objects."""
        bounding_boxes = []
        if boxes is None or len(boxes) == 0:
            return bounding_boxes
        
        boxes_np = (boxes.detach().cpu().numpy() if boxes.requires_grad 
                   else boxes.cpu().numpy())
        logits_np = (logits.detach().cpu().numpy() if logits.requires_grad 
                    else logits.cpu().numpy())
        
        detection_width, detection_height = det_size
        needs_rescale = max(orig_w, orig_h) > max_dim
        inv_scale = (orig_w / detection_width if needs_rescale else 1.0,
                    orig_h / detection_height if needs_rescale else 1.0)
        
        for i in range(len(boxes_np)):
            cx, cy, w, h = boxes_np[i]
            x1 = int((cx - w / 2) * detection_width)
            y1 = int((cy - h / 2) * detection_height)
            x2 = int((cx + w / 2) * detection_width)
            y2 = int((cy + h / 2) * detection_height)
            
            if needs_rescale:
                x1, y1, x2, y2 = int(x1 * inv_scale[0]), int(y1 * inv_scale[1]), int(x2 * inv_scale[0]), int(y2 * inv_scale[1])
            
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            if (x2 - x1) < self.config.min_box_size or (y2 - y1) < self.config.min_box_size:
                continue
            
            bounding_boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(logits_np[i]),
                class_id=i,
                class_name=phrases[i],
            ))
        
        return bounding_boxes

    def _ensure_model_loaded(self) -> None:
        """
        Load the model if it hasn't been loaded yet. This method is safe to call
        multiple times (idempotent).
        """
        if self.model is None:
            logger.info("Grounding DINO model not loaded yet — loading now (lazy)")
            self._load_model()
    
    def crop_object(
        self, 
        image: Image.Image, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int
    ) -> Image.Image:
        """
        Crop an object from an image using bounding box coordinates.
        
        Args:
            image: The source image
            x1: Top-left x coordinate
            y1: Top-left y coordinate
            x2: Bottom-right x coordinate
            y2: Bottom-right y coordinate
            
        Returns:
            Cropped image of the detected object
        """
        try:
            # Validate coordinates
            if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
                logger.warning(f"Bounding box coordinates out of bounds: ({x1},{y1},{x2},{y2}), image size: {image.size}")
                # Clip coordinates to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid bounding box: ({x1},{y1},{x2},{y2})")
            
            # Crop image
            cropped = image.crop((x1, y1, x2, y2))
            
            logger.debug(f"Cropped object: bbox=({x1},{y1},{x2},{y2}), size={cropped.size}")
            
            return cropped
            
        except Exception as e:
            logger.error(f"Error cropping object: {e}")
            raise RuntimeError(f"Failed to crop object: {e}")
