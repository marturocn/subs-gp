"""
OWL-ViT v2 Model Repository Implementation - Infrastructure Layer
"""
import logging
import time
import torch
from torch.amp.autocast_mode import autocast
import numpy as np
from typing import Optional, List, Union
from io import BytesIO
from PIL import Image
import warnings

from src.domain.model.repositories.owlvit_repository import OwlvitRepository
from src.domain.model.models.detection_models import DetectionResult, BoundingBox
from src.infraestructure.persistance.model.config.owlvit_config import OwlvitConfig


logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class OwlvitRepositoryImpl(OwlvitRepository):
    """
    Real implementation of OWL-ViT v2 service.
    Follows the repository pattern with synchronous and asynchronous methods.
    Supports open-vocabulary object detection using text prompts.
    """
    
    def __init__(self, config: Optional[OwlvitConfig] = None):
        """
        Initialize OWL-ViT v2 model repository.
        
        Args:
            config: OWL-ViT model configuration. If None, loads from environment.
        """
        self.config = config or OwlvitConfig.from_env()
        self.processor: Optional[object] = None
        self.model: Optional[object] = None
        self.device: str = "cpu"
        self._warm_up_done = False
        
        # ONNX/TensorRT support
        self.onnx_session = None
        self.use_onnx = self.config.use_onnx
        self.use_fp16 = False
        
        # NOTE: do not load model in constructor to avoid automatic downloads at import/instantiation.
        # Model will be loaded lazily on first detection call via _ensure_model_loaded().
    
    def _load_model(self) -> None:
        """
        Load OWL-ViT v2 model from optimized formats or PyTorch native.
        Priority:
        1. PyTorch checkpoint (.pt) if use_onnx=True
        2. PyTorch native model (fallback)
        """
        # First, try to load optimized model if enabled
        if self.use_onnx:
            try:
                self._load_pytorch_checkpoint()
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to load optimized checkpoint: {e}")
                logger.info("   Falling back to PyTorch model...")
        
        # Fall back to standard PyTorch model loading
        self._load_pytorch_native()
    
    def _load_pytorch_checkpoint(self) -> None:
        """
        Load OWL-ViT v2 from optimized PyTorch checkpoint (.pt file).
        This is saved without text encoder complexity.
        """
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        from pathlib import Path
        
        onnx_path = Path(self.config.onnx_export_path)
        checkpoint_path = onnx_path.with_suffix('.pt')
        
        logger.info("🚀 Loading optimized PyTorch checkpoint...")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        logger.info("   📦 Loading processor...")
        self.processor = Owlv2Processor.from_pretrained(self.config.model_name)
        
        logger.info(f"   📦 Loading model checkpoint: {checkpoint_path.name}")
        self.model = Owlv2ForObjectDetection.from_pretrained(self.config.model_name)
        
        # Load state dict from checkpoint
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Set device
        if getattr(self.config, 'cpu_only', False):
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)  # type: ignore
        self.model.eval()  # type: ignore
        
        # Configure FP16
        self.use_fp16 = (self.device == 'cuda' and self.config.use_fp16)
        if self.use_fp16:
            logger.info("   ⚡ FP16 precision enabled")
        
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(f"   ✅ PyTorch checkpoint loaded ({checkpoint_size_mb:.1f}MB)")
        logger.info(f"   📊 Score threshold: {self.config.score_threshold}")
        logger.info(f"   🎯 Default text queries: {self.config.text_queries}")
        
        self._warmup_model()
    
    def _load_pytorch_native(self) -> None:
        """
        Load OWL-ViT v2 model and processor from Hugging Face (native PyTorch).
        """
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        
        logger.info(f"Loading OWL-ViT v2 model: {self.config.model_name}")
        
        # Set device
        if getattr(self.config, 'cpu_only', False):
            logger.info("OWL-ViT configured for CPU-only mode via config.cpu_only")
            self.device = "cpu"
        else:
            if getattr(self.config, 'device', 'cpu') == "cuda" and torch.cuda.is_available():
                self.device = "cuda"
                # CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                if getattr(self.config, 'device', 'cpu') == "cuda":
                    logger.warning("Requested device 'cuda' is not available; falling back to 'cpu'")
                self.device = "cpu"

        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = Owlv2Processor.from_pretrained(self.config.model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.config.model_name)
        
        # Move model to device
        self.model.to(self.device)  # type: ignore
        
        # Set model to evaluation mode
        self.model.eval()  # type: ignore
        
        # Configure FP16 if enabled and on CUDA
        self.use_fp16 = False
        if self.device == 'cuda' and self.config.use_fp16:
            self.use_fp16 = True
            logger.info("Configured to use automatic mixed precision (autocast) for inference")
        
        logger.info(f"✅ OWL-ViT v2 model loaded successfully on device: {self.device}")
        logger.info(f"📊 Score threshold: {self.config.score_threshold}")
        logger.info(f"🎯 Default text queries: {self.config.text_queries}")
        
        # Warmup del modelo
        self._warmup_model()
    
    def _warmup_model(self) -> None:
        """
        Warmup model with dummy inference to initialize CUDA kernels.
        """
        if self._warm_up_done or self.model is None:
            return
        
        # Skip warmup for ONNX models
        if self.use_onnx and self.onnx_session:
            logger.info("ℹ️  Skipping warmup for ONNX model (already optimized)")
            self._warm_up_done = True
            return

        # Only warmup on CUDA devices
        if self.device != 'cuda':
            logger.info("ℹ️  Skipping OWL-ViT warmup (device is not CUDA)")
            self._warm_up_done = True
            return

        try:
            warmup_iters = self.config.warmup_iterations
            warmup_size = self.config.warmup_image_size

            logger.info(f"🔥 Running OWL-ViT warmup: {warmup_iters} iters, size={warmup_size}")

            # Create a dummy image
            dummy_image = Image.new('RGB', (warmup_size, warmup_size), color='white')
            dummy_texts = [["object", "thing"]]

            # Run warmup iterations
            with torch.inference_mode():
                for i in range(warmup_iters):
                    try:
                        inputs = self.processor(  # type: ignore
                            text=dummy_texts, 
                            images=dummy_image, 
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        if self.use_fp16:
                            with autocast(device_type='cuda'):
                                _ = self.model(**inputs)  # type: ignore
                        else:
                            _ = self.model(**inputs)  # type: ignore
                    except Exception as e:
                        logger.debug(f"Warmup iteration {i} failed (ignored): {e}")

            logger.info("✅ OWL-ViT warmup complete")
        except Exception as e:
            logger.warning(f"OWL-ViT warmup failed or skipped: {e}")
        finally:
            self._warm_up_done = True
    
    async def detect_objects(
        self, 
        image_data: BytesIO, 
        text_prompt: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect objects in an image using OWL-ViT v2 model (async).
        Async wrapper that calls the sync implementation.
        
        Args:
            image_data: The image data as BytesIO
            text_prompt: Text prompt for detection (comma-separated string or None)
            
        Returns:
            DetectionResult containing all detected objects
        """
        self._ensure_model_loaded()
        return self.detect_objects_sync(image_data, text_prompt)
    
    def _parse_text_queries(self, text_prompt: Optional[str]) -> list:
        """Parse and validate text queries from prompt."""
        if text_prompt:
            if ',' in text_prompt:
                text_queries = [q.strip() for q in text_prompt.split(',')]
            elif '.' in text_prompt:
                text_queries = [q.strip() for q in text_prompt.split('.') if q.strip()]
            else:
                text_queries = [text_prompt.strip()]
        else:
            text_queries = self.config.text_queries
        
        # OWL-ViT v2 has a limit of 16 text queries per batch
        MAX_QUERIES = 16
        if len(text_queries) > MAX_QUERIES:
            logger.warning(f"OWL-ViT v2 supports max {MAX_QUERIES} queries, truncating from {len(text_queries)}")
            text_queries = text_queries[:MAX_QUERIES]
        
        return text_queries
    
    def _prepare_image(self, image: Image.Image) -> tuple:
        """Convert image to RGB and return original dimensions."""
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
                image = background
            else:
                image = image.convert('RGB')
        
        return image, image.size
    
    def _resize_image_if_needed(self, image: Image.Image, original_width: int, original_height: int) -> tuple:
        """Resize image if it exceeds max dimension."""
        max_dim = self.config.max_detection_dimension
        
        if max(original_width, original_height) <= max_dim:
            return image, False
        
        if original_width >= original_height:
            scale = max_dim / float(original_width)
        else:
            scale = max_dim / float(original_height)
        
        new_w = int(original_width * scale)
        new_h = int(original_height * scale)
        
        resampling_attr = getattr(Image, 'Resampling', None)
        if resampling_attr is not None and hasattr(resampling_attr, 'LANCZOS'):
            resample_filter = resampling_attr.LANCZOS
        else:
            resample_filter = getattr(Image, 'LANCZOS', None) or getattr(Image, 'NEAREST', 0)
        
        resized_image = image.resize((new_w, new_h), resample=resample_filter)
        logger.info(f"Resized image from {original_width}x{original_height} -> {new_w}x{new_h}")
        
        return resized_image, True
    
    def _run_inference(self, texts: list, resized_image: Image.Image) -> dict:
        """Run model inference."""
        inputs = self.processor(text=texts, images=resized_image, return_tensors="pt")  # type: ignore
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            if self.use_fp16 and self.device == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = self.model(**inputs)  # type: ignore
            else:
                outputs = self.model(**inputs)  # type: ignore
        
        return outputs
    
    def _process_detections(self, results: list, text_queries: list, original_width: int, 
                           original_height: int, detection_width: int, detection_height: int, 
                           needs_rescale: bool) -> list:
        """Process detection results and create BoundingBox objects."""
        bounding_boxes = []
        
        result = results[0]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        
        if needs_rescale:
            inv_scale_x = original_width / float(detection_width) if detection_width != 0 else 1.0
            inv_scale_y = original_height / float(detection_height) if detection_height != 0 else 1.0
        else:
            inv_scale_x = inv_scale_y = 1.0
        
        for i in range(len(boxes)):
            box = boxes[i]
            score = float(scores[i])
            label_idx = int(labels[i])
            
            class_name = text_queries[label_idx] if label_idx < len(text_queries) else f"object_{label_idx}"
            
            x1, y1, x2, y2 = [int(c) for c in box]
            
            if needs_rescale:
                x1, y1 = int(x1 * inv_scale_x), int(y1 * inv_scale_y)
                x2, y2 = int(x2 * inv_scale_x), int(y2 * inv_scale_y)
            
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))
            
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < self.config.min_box_size or box_height < self.config.min_box_size:
                continue
            
            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=score,
                class_id=label_idx,
                class_name=class_name,
            )
            bounding_boxes.append(bbox)
        
        return bounding_boxes

    def detect_objects_sync(
        self, 
        image_data: BytesIO, 
        text_prompt: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect objects in an image using OWL-ViT v2 model (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            text_prompt: Text prompt for detection (comma-separated string).
                        If None, uses default queries from config.
            
        Returns:
            DetectionResult containing all detected objects with bounding boxes
        """
        self._ensure_model_loaded()
        
        try:
            image_data.seek(0)
            
            if self.model is None or self.processor is None:
                raise RuntimeError("OWL-ViT model not available")
            
            # Parse text queries
            text_queries = self._parse_text_queries(text_prompt)
            
            # Load and prepare image
            image = Image.open(image_data)
            image, original_size = self._prepare_image(image)
            original_width, original_height = original_size
            
            # Resize if necessary
            resized_image, needs_rescale = self._resize_image_if_needed(image, original_width, original_height)
            
            # Run inference
            start_time = time.time()
            texts = [text_queries]
            outputs = self._run_inference(texts, resized_image)
            inference_time = (time.time() - start_time) * 1000
            
            # Post-process results
            target_sizes = torch.tensor([resized_image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(  # type: ignore
                outputs=outputs,
                threshold=self.config.score_threshold,
                target_sizes=target_sizes
            )
            
            # Process detections
            detection_width, detection_height = resized_image.size
            bounding_boxes = self._process_detections(
                results, text_queries, original_width, original_height,
                detection_width, detection_height, needs_rescale
            )
            
            logger.info(f"⚡ Detected {len(bounding_boxes)} objects in {inference_time:.2f}ms")
            logger.debug(f"Used text queries: {text_queries}")
            
            return DetectionResult(
                bounding_boxes=bounding_boxes,
                image_width=original_width,
                image_height=original_height,
                inference_time=inference_time,
            )
            
        except Exception as e:
            logger.error(f"Error detecting objects with OWL-ViT: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to detect objects: {e}")

    def _ensure_model_loaded(self) -> None:
        """
        Load the model if it hasn't been loaded yet. This method is safe to call
        multiple times (idempotent).
        """
        if self.model is None or self.processor is None:
            logger.info("OWL-ViT model not loaded yet — loading now (lazy)")
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
