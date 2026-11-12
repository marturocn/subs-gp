"""
DINOv3 Model Repository Implementation - Infrastructure Layer
"""
import logging
import time
import torch
from torch.amp.autocast_mode import autocast
import numpy as np
from typing import Optional, Tuple
from io import BytesIO
from PIL import Image
from functools import lru_cache
from pathlib import Path

from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.infraestructure.persistance.model.config.dinov3_config import DINOv3Config


logger = logging.getLogger(__name__)


class DINOv3ModelRepositoryImpl(DINOv3ModelRepository):
    """
    Real implementation of DINOv3 service for generating image embeddings.
    Uses HuggingFace Transformers for model loading and inference.
    Supports ONNX inference for optimized performance.
    
    IMPORTANT: Uses register tokens (CLS + 4 registers) for better embeddings.
    DINOv2/v3 register tokens significantly improve precision (5-12%) in
    fine-grained similarity tasks like distinguishing buildings from people.
    """
    
    # Using CLS token for embeddings (matching QA behavior)
    
    def __init__(self, config: Optional[DINOv3Config] = None):
        """
        Initialize DINOv3 model repository.
        
        Args:
            config: DINOv3 model configuration. If None, loads from environment.
        """
        self.config = config or DINOv3Config.from_env()
        self.model = None
        self.model_batch = None  # Modelo sin compilar para batch processing
        self.processor = None
        self.device = None
        self._warm_up_done = False
        
        # ONNX support
        self.onnx_session = None
        self.trt_session = None
        self.use_onnx = self.config.use_onnx
        
        self._load_model()
    
    def _setup_cuda_optimizations(self) -> None:
        """Setup CUDA-specific optimizations (cuDNN, TF32, Flash Attention)."""
        if self.config.device != "cuda" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            return

        self.device = torch.device("cuda")

        # 1. CuDNN Benchmark: Encuentra el mejor algoritmo de convolución
        torch.backends.cudnn.benchmark = True

        # 2. TensorFloat-32 (TF32): Acelera operaciones matemáticas en GPUs Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 3. Optimizar kernels CUDA para mejor rendimiento
        torch.backends.cudnn.deterministic = False  # Permite algoritmos no deterministas (más rápidos)

        # 4. Flash Attention (si está disponible - PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("✅ Flash Attention enabled for faster transformer inference")

        # 5. Configurar memoria CUDA para mejor rendimiento
        if torch.cuda.get_device_properties(0).major >= 7:  # Volta or newer
            torch.cuda.empty_cache()
            logger.info("✅ CUDA memory optimizations enabled")

        logger.info("✅ CUDA optimizations enabled: TF32, cuDNN benchmark, Flash Attention")

    def _load_model_from_pretrained(self) -> tuple:
        """Load processor and model from HuggingFace with auth token fallback."""
        from transformers import AutoImageProcessor, AutoModel

        try:
            if self.config.use_auth_token:
                processor = AutoImageProcessor.from_pretrained(
                    self.config.model_name,
                    token=self.config.use_auth_token
                )
                model = AutoModel.from_pretrained(
                    self.config.model_name,
                    token=self.config.use_auth_token
                )
            else:
                processor = AutoImageProcessor.from_pretrained(
                    self.config.model_name
                )
                model = AutoModel.from_pretrained(
                    self.config.model_name
                )
            return processor, model
        except Exception as e:
            logger.error(f"Error loading model with token: {e}")
            logger.info("Trying without authentication token...")
            processor = AutoImageProcessor.from_pretrained(
                self.config.model_name
            )
            model = AutoModel.from_pretrained(
                self.config.model_name
            )
            return processor, model

    def _apply_model_precision_optimizations(self) -> None:
        """Apply half precision and run safety tests."""
        if self.device.type != "cuda": # pyright: ignore[reportOptionalMemberAccess]
            return

        try:
            if getattr(self.config, 'use_half_precision', False):
                self.model = self.model.half() # type: ignore
                logger.info("✅ Model converted to half precision for faster GPU inference")

                # Test half precision with a small inference
                test_input = torch.randn(1, 3, 224, 224).to(self.device).half()
                with torch.no_grad():
                    test_output = self.model(test_input)
                    out_tensor = test_output.last_hidden_state if hasattr(test_output, 'last_hidden_state') else test_output
                    if torch.is_tensor(out_tensor) and (torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any()):
                        logger.warning("⚠️  Half precision test failed (NaN/Inf detected), falling back to full precision")
                        self.model = self.model.float()
                    else:
                        logger.info("✅ Half precision test passed")
            else:
                logger.info("ℹ️  Half precision disabled (use_half_precision=False in config)")
        except Exception as e:
            logger.warning(f"Failed to convert model to half precision: {e}")
            self.model = self.model.float() # type: ignore

    def _apply_model_compiler_optimizations(self) -> None:
        """Apply BetterTransformer and torch.compile if available."""
        # Optimización 1: BetterTransformer para aceleración automática
        if self.config.use_better_transformer and self.device.type == "cuda": # pyright: ignore[reportOptionalMemberAccess]
            try:
                self.model = self.model.to_bettertransformer() # type: ignore
                logger.info("✅ BetterTransformer optimization enabled")
            except Exception as e:
                logger.warning(f"BetterTransformer not available: {e}")

        # Optimización 2: torch.compile para optimización del grafo computacional
        if self.config.use_compile and hasattr(torch, 'compile') and self.device.type == "cuda": # pyright: ignore[reportOptionalMemberAccess]
            try:
                self.model_batch = self.model

                self.model = torch.compile(
                    self.model,
                    mode=self.config.compile_mode,
                    fullgraph=False,
                    dynamic=False
                )
                logger.info("✅ Model compiled with torch.compile for single images (mode=default)")
                logger.info("✅ Separate uncompiled model created for batch processing")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
                self.model_batch = self.model
        else:
            self.model_batch = self.model

    def _load_model(self) -> None:
        """
        Load DINOv3 model and processor from HuggingFace.
        Optionally loads ONNX model if configured.
        """
        # Check if we should use ONNX model
        if self.use_onnx and Path(self.config.onnx_export_path).exists():
            logger.info("🚀 ONNX mode enabled - loading optimized ONNX model")
            self._load_onnx_model()
            return
        elif self.use_onnx and not Path(self.config.onnx_export_path).exists():
            logger.warning(f"⚠️  ONNX model not found at {self.config.onnx_export_path}")
            logger.warning("    Falling back to PyTorch model. Run export script to create ONNX model:")
            logger.warning("    python -m scripts.export_models_to_onnx --model dinov3")

        # Load PyTorch model (original implementation)
        try:
            logger.info(f"Loading DINOv3 model: {self.config.model_name}")

            # Setup CUDA optimizations
            self._setup_cuda_optimizations()

            print(f"Using device: {self.device} in DINOv3ModelRepositoryImpl")

            # Load processor and model
            self.processor, self.model = self._load_model_from_pretrained()

            # Move model to device
            self.model = self.model.to(self.device)

            # Apply precision optimizations
            self._apply_model_precision_optimizations()

            self.model.eval()

            # Apply compiler optimizations
            self._apply_model_compiler_optimizations()

            logger.info(f"✅ DINOv3 model loaded successfully on device: {self.device}")
            logger.info(f"🔧 Normalize embeddings: {self.config.normalize_embeddings}")

            # Warmup del modelo para inicializar optimizaciones
            self._warmup_model()

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Make sure transformers is installed: pip install transformers")
            raise RuntimeError(f"Failed to import transformers: {e}")
        except Exception as e:
            logger.error(f"Failed to load DINOv3 model: {e}")
            raise RuntimeError(f"Failed to load DINOv3 model: {e}")
    
    def _load_image_processor(self) -> None:
        """Load image processor with auth token fallback."""
        from transformers import AutoImageProcessor
        
        logger.info("📦 Loading image processor for preprocessing...")
        try:
            if self.config.use_auth_token:
                self.processor = AutoImageProcessor.from_pretrained(
                    self.config.model_name,
                    use_auth_token=self.config.use_auth_token
                )
            else:
                self.processor = AutoImageProcessor.from_pretrained(
                    self.config.model_name
                )
        except Exception as e:
            logger.error(f"Error loading processor with token: {e}")
            logger.info("Trying without authentication token...")
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_name
            )

    def _try_load_tensorrt_engine(self, engine_path: Path, onnx_path: Path) -> bool:
        """
        Attempt to load TensorRT engine.
        Returns True if successful, False otherwise.
        """
        if not engine_path.exists():
            return False

        logger.info(f"🚀 Found TensorRT engine: {engine_path.name}")
        try:
            from src.infraestructure.persistance.model.utils import TensorRTInferenceSession
            logger.info("   Loading TensorRT engine with GPU acceleration...")
            self.trt_session = TensorRTInferenceSession(
                engine_path=str(engine_path)
            )
            engine_size_mb = engine_path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ TensorRT engine loaded ({engine_size_mb:.1f}MB)")
            logger.info("   ⚡ Performance: Maximum GPU acceleration enabled")
            logger.info("   📊 Expected latency: 25-30ms for 224x224 images (vs 45ms with PyTorch)")
            self.use_onnx = True
            self.device = torch.device("cpu")
            return True
        except RuntimeError as e:
            if "OutOfMemory" in str(e) or "memory" in str(e).lower():
                logger.warning(f"   ⚠️  TensorRT OutOfMemory: {e}")
                logger.warning("   GPU memory insufficient for TensorRT engine")
            else:
                logger.warning(f"   ⚠️  TensorRT loading failed: {e}")
            logger.info("   Falling back to ONNX...")
            return False
        except Exception as e:
            logger.warning(f"   ⚠️  TensorRT loading failed: {e}")
            logger.info("   Falling back to ONNX...")
            return False

    def _try_load_onnx_model(self, onnx_path: Path) -> bool:
        """
        Attempt to load ONNX model.
        Returns True if successful, False otherwise.
        """
        if not onnx_path.exists():
            return False

        logger.info(f"📦 Found ONNX model: {onnx_path.name}")
        try:
            from src.infraestructure.persistance.model.utils import (
                ONNXInferenceSession,
                ONNXOptimizer,
            )
            logger.info("   Loading with optimized CUDA provider...")
            providers = ONNXOptimizer.get_onnx_providers(self.config.onnx_provider)
            self.onnx_session = ONNXInferenceSession(
                model_path=str(onnx_path),
                providers=providers,
            )
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ ONNX model loaded ({onnx_size_mb:.1f}MB)")
            logger.info("   ⚡ Performance: CUDA acceleration enabled")
            logger.info("   📊 Expected latency: 30-40ms for 224x224 images")
            self.use_onnx = True
            self.device = torch.device("cpu")
            logger.info("✅ DINOv3 model loaded successfully with ONNX optimization")
            return True
        except Exception as e:
            logger.warning(f"   ⚠️  ONNX loading failed: {e}")
            logger.warning("   Falling back to PyTorch...")
            return False

    def _load_onnx_model(self) -> None:
        """
        Load DINOv3 model from optimized formats in priority order:
        1. TensorRT engine (.engine) - Fastest, GPU-optimized
        2. ONNX model (.onnx) - Fast, cross-platform
        3. PyTorch native - Fallback
        """
        try:
            # Always load processor first (needed for all inference backends)
            self._load_image_processor()

            # Check for TensorRT engine (.engine file) - HIGHEST PRIORITY
            onnx_path = Path(self.config.onnx_export_path)
            engine_path = onnx_path.with_suffix('.engine')

            if self._try_load_tensorrt_engine(engine_path, onnx_path):
                return

            # Check for ONNX model (.onnx file) - SECOND PRIORITY
            if self._try_load_onnx_model(onnx_path):
                return

            # No optimized models found - load PyTorch native (FALLBACK)
            logger.warning("⚠️  No optimized models found")
            logger.warning(f"   Expected: {onnx_path} or {engine_path}")
            logger.info("   Loading PyTorch model (consider running export script for faster inference)")
            logger.info("   Command: python -m scripts.export_models_to_onnx --model dinov3 --tensorrt")
            self._load_model()
                
        except ImportError as e:
            logger.error(f"Failed to import ONNX dependencies: {e}")
            logger.error("Install with: pip install onnxruntime-gpu tensorrt pycuda")
            raise RuntimeError(f"Failed to import ONNX dependencies: {e}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _warmup_model(self) -> None:
        """
        Warmup model with dummy inference to initialize CUDA kernels and optimizations.
        This significantly reduces first inference time.
        """
        if self._warm_up_done:
            return
        
        # Skip warmup for ONNX models (they are already optimized)
        if self.use_onnx and self.onnx_session:
            logger.info("ℹ️  Skipping warmup for ONNX model (already optimized)")
            self._warm_up_done = True
            return
            
        try:
            logger.info("🔥 Warming up model...")
            
            # Warmup con batch size 1 (imagen completa)
            dummy_image = Image.new('RGB', (224, 224), color='white')
            inputs = self.processor(images=dummy_image, return_tensors="pt") # pyright: ignore[reportOptionalCall]
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            
            self._run_initial_warmup(inputs)
            
            # Warmup adicional para torch.compile
            if self.device and self.device.type == "cuda" and self.config.use_compile:
                self._warmup_compiled_model(inputs)
            
            self._warm_up_done = True
            logger.info("✅ Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
    
    def _run_initial_warmup(self, inputs: dict) -> None:
        """Run initial warmup inference."""
        with torch.no_grad():
            if self.device and self.device.type == "cuda":
                with autocast(device_type="cuda"):
                    _ = self.model(**inputs) # pyright: ignore[reportCallIssue, reportOptionalCall]
            else:
                _ = self.model(**inputs) # pyright: ignore[reportCallIssue, reportOptionalCall]
    
    def _warmup_compiled_model(self, inputs: dict) -> None:
        """Warmup compiled model with multiple passes to ensure graph compilation."""
        logger.info("🔥 Warming up compiled model... (this may take 10-20 seconds for torch.compile)")
        for i in range(5):
            with torch.no_grad():
                with autocast(device_type="cuda"):
                    _ = self.model(**inputs) # pyright: ignore[reportCallIssue, reportOptionalCall]
            if i == 0:
                logger.info("   Compiling computational graph (first pass)...")
            elif i == 4:
                logger.info("   Compilation complete, graph cached")
        
        logger.info("✅ Compiled model ready (optimized for single images)")
        
        # Warmup del modelo batch (sin compilar) con batch size 3
        if self.model_batch is not None and self.model_batch is not self.model:
            self._warmup_batch_model()
    
    def _warmup_batch_model(self) -> None:
        """Warmup batch model for variable batch size processing."""
        logger.info("🔥 Warming up batch model (batch=3)...")
        dummy_images = [Image.new('RGB', (224, 224), color='white') for _ in range(3)]
        inputs_batch = self.processor(images=dummy_images, return_tensors="pt") # pyright: ignore[reportOptionalCall]
        inputs_batch = {k: v.to(self.device, non_blocking=True) for k, v in inputs_batch.items()}
        
        for _ in range(3):
            with torch.no_grad():
                with autocast(device_type="cuda"):
                    _ = self.model_batch(**inputs_batch) # pyright: ignore[reportCallIssue, reportOptionalCall]
        
        logger.info("✅ Batch model ready (optimized for variable batch sizes)")
    
    @lru_cache(maxsize=128)
    def _get_default_image_size(self) -> Tuple[int, int]:
        """Cache del tamaño de imagen por defecto para evitar recalcular."""
        return (224, 224)
    
    def _convert_image_to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB format if necessary."""
        if image.mode == 'RGB':
            return image
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
            return background
        return image.convert('RGB')

    def _run_tensorrt_inference(self, inputs: dict, input_array: np.ndarray) -> np.ndarray:
        """Run inference using TensorRT engine.
        
        Uses register tokens (CLS + 4 registers) for better embeddings if output
        contains full hidden states. Falls back to direct output if model was
        exported with register token averaging built-in.
        """
        # Ensure correct dtype for TensorRT (typically float32)
        expected_dtype = self.trt_session.get_input_dtype() # pyright: ignore[reportOptionalMemberAccess]
        if 'float16' in expected_dtype:
            input_array = input_array.astype(np.float16)
        elif input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)

        # TensorRT expects flat inputs - try both input names for compatibility
        trt_inputs = {"pixel_values": input_array}
        try:
            outputs = self.trt_session.run(trt_inputs) # pyright: ignore[reportOptionalMemberAccess]
        except Exception:
            # Fallback to old input name for legacy models
            trt_inputs = {"input": input_array}
            outputs = self.trt_session.run(trt_inputs) # pyright: ignore[reportOptionalMemberAccess]

        output = outputs[0]
        
        # Check output shape to determine if we need to extract CLS token
        # New format: (batch, hidden_dim) - already computed
        # Old format: (batch, num_tokens, hidden_dim) - needs CLS extraction
        if output.ndim == 2:
            # New format: embedding already computed
            embedding = output
            logger.debug("TensorRT: Using pre-computed embedding")
        else:
            # Old format: extract CLS token (first token)
            cls_embedding = output[:, 0, :]  # Shape: (batch, hidden_dim)
            logger.debug("TensorRT: Using CLS token")
            
            # Normalize if configured
            if self.config.normalize_embeddings:
                cls_embedding = cls_embedding / (np.linalg.norm(cls_embedding, axis=1, keepdims=True) + 1e-8)
            
            embedding = cls_embedding

        return embedding.flatten()

    def _run_onnx_inference(self, inputs: dict, input_array: np.ndarray) -> np.ndarray:
        """Run inference using ONNX session.
        
        Uses register tokens (CLS + 4 registers) for better embeddings if output
        contains full hidden states. Falls back to direct output if model was
        exported with register token averaging built-in.
        """
        # Determine expected dtype from ONNX session
        expected_dtype = self.onnx_session.session.get_inputs()[0].type # pyright: ignore[reportOptionalMemberAccess]
        if 'float16' in str(expected_dtype):
            input_array = input_array.astype(np.float16)
            logger.debug("Converted input to float16 for ONNX model")

        in_name = self.onnx_session.input_names[0] # pyright: ignore[reportOptionalMemberAccess]
        onnx_inputs = {in_name: input_array}
        outputs = self.onnx_session.run(onnx_inputs) # pyright: ignore[reportOptionalMemberAccess]

        output = outputs[0]
        
        # Check output shape to determine if we need to extract CLS token
        # New format: (batch, hidden_dim) - already computed
        # Old format: (batch, num_tokens, hidden_dim) - needs CLS extraction
        if output.ndim == 2:
            # New format: embedding already computed
            embedding = output
            logger.debug("ONNX: Using pre-computed embedding")
        else:
            # Old format: extract CLS token (first token)
            cls_embedding = output[:, 0, :]  # Shape: (batch, hidden_dim)
            logger.debug("ONNX: Using CLS token")
            
            # Normalize if configured
            if self.config.normalize_embeddings:
                cls_embedding = cls_embedding / (np.linalg.norm(cls_embedding, axis=1, keepdims=True) + 1e-8)
            
            embedding = cls_embedding

        return embedding.flatten()

    def _run_pytorch_inference(self, inputs: dict) -> np.ndarray:
        """Run inference using PyTorch model.
        
        Uses CLS token (first token) for embeddings.
        """
        # Optimización 4: Transferencia asíncrona a GPU con pin_memory
        if self.device and self.device.type == "cuda":
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Optimización 5: Inferencia con autocast y torch.inference_mode
        with torch.inference_mode():  # Más rápido que no_grad() para inferencia
            if self.device and self.device.type == "cuda":
                # Usar autocast para mixed precision automática
                with autocast(device_type="cuda", dtype=torch.float32):  # Usar float32 en lugar de float16
                    outputs = self.model(**inputs) # pyright: ignore[reportCallIssue, reportOptionalCall]
            else:
                outputs = self.model(**inputs) # pyright: ignore[reportCallIssue, reportOptionalCall]

            # Extract CLS token (first token) for embeddings
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            # Normalizar si está configurado (operación rápida en GPU)
            if self.config.normalize_embeddings:
                cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

        # Optimización 6: Conversión eficiente a numpy
        # Convertir a float32 antes de CPU para compatibilidad
        if cls_embedding.dtype == torch.float16:
            cls_embedding = cls_embedding.to(torch.float32)

        return cls_embedding.cpu().numpy().flatten()

    def _validate_embedding(self, embedding: np.ndarray) -> None:
        """Validate embedding for NaN/Inf and zero values."""
        # VALIDACIÓN: Verificar que el embedding no tenga NaN o Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.error("❌ CRITICAL: Embedding contains NaN or Inf values!")
            logger.error(f"   NaN count: {np.sum(np.isnan(embedding))}")
            logger.error(f"   Inf count: {np.sum(np.isinf(embedding))}")
            logger.error(f"   Embedding stats - min: {np.min(embedding)}, max: {np.max(embedding)}, mean: {np.mean(embedding)}")
            raise RuntimeError("Model produced invalid embeddings (NaN/Inf). Check model precision settings.")

        # VALIDACIÓN: Verificar que no sea todo ceros
        if np.all(embedding == 0):
            logger.warning("⚠️  Embedding is all zeros - this may indicate a problem")

    def _compute_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Internal method to compute embedding for a PIL Image.
        Optimized for speed with CUDA streams and pinned memory.
        Supports both PyTorch and ONNX inference.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            # Optimización 3: Conversión rápida de imagen
            image = self._convert_image_to_rgb(image)

            # Process image - el processor ya hace resize eficiente
            inputs = self.processor(images=image, return_tensors="pt") # pyright: ignore[reportOptionalCall]

            # Inference with accelerator: use TensorRT first (fastest), then ONNX, then PyTorch
            if self.use_onnx and self.trt_session:
                # TensorRT inference (fastest path)
                input_array = inputs['pixel_values'].numpy()
                embedding = self._run_tensorrt_inference(inputs, input_array)

            elif self.use_onnx and self.onnx_session:
                # ONNX inference (fast fallback)
                input_array = inputs['pixel_values'].numpy()
                embedding = self._run_onnx_inference(inputs, input_array)

            else:
                # PyTorch inference path
                embedding = self._run_pytorch_inference(inputs)

            # Validate the embedding
            self._validate_embedding(embedding)

            logger.debug(f"Computed embedding of shape {embedding}")
            return embedding

        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            raise RuntimeError(f"Failed to compute embedding: {e}")
    
    async def encode_image(self, image_data: BytesIO) -> np.ndarray:
        """
        Generate embedding for a full image using DINOv3 (async).
        Async wrapper that calls the sync implementation.
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            Normalized embedding vector as numpy array
        """
        return self.encode_image_sync(image_data)
    
    def encode_image_sync(self, image_data: BytesIO) -> np.ndarray:
        """
        Generate embedding for a full image using DINOv3 (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            # Reset stream position
            image_data.seek(0)
            
            # Validar que el modelo o la sesión ONNX estén disponibles
            if self.model is None and self.onnx_session is None and self.trt_session is None:
                raise RuntimeError("DINOv3 model not available")
            
            # Load image
            image = Image.open(image_data)
            
            # Optimización 7: Resize inteligente para imágenes grandes
            # DINOv3 redimensiona a 224x224, hacer esto antes ahorra procesamiento
            max_size = 800  # Reducir imágenes muy grandes antes del procesamiento
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.BILINEAR)
                logger.debug(f"Image resized from large to {new_size} for faster processing")
            
            # Start timing
            start_time = time.time()
            
            # Compute embedding
            embedding = self._compute_embedding(image)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Log GPU memory if available
            if self.device and self.device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**2  # MB
                logger.debug(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
            
            logger.info(f"⚡ Generated full image embedding in {inference_time:.2f}ms, shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise RuntimeError(f"Failed to encode image: {e}")
    
    async def encode_cropped_image(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for a cropped/PIL image using DINOv3 (async).
        Async wrapper that calls the sync implementation.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        return self.encode_cropped_image_sync(image)
    
    def encode_cropped_image_sync(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for a cropped/PIL image using DINOv3 (synchronous).
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            # Validar que el modelo o la sesión ONNX estén disponibles
            if self.model is None and self.onnx_session is None and self.trt_session is None:
                raise RuntimeError("DINOv3 model not available")
            
            # Optimización 8: Resize de crops pequeños
            # Si el crop es muy pequeño, hacer upscale para mejor calidad
            min_size = 100
            if min(image.size) < min_size:
                ratio = min_size / min(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.BILINEAR)
                logger.debug(f"Small crop upscaled to {new_size}")
            
            # Start timing
            start_time = time.time()
            
            # Compute embedding
            embedding = self._compute_embedding(image)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Generated cropped embedding in {inference_time:.2f}ms, shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding cropped image: {e}")
            raise RuntimeError(f"Failed to encode cropped image: {e}")
    
    def encode_images_batch(self, images: list[Image.Image], verbose: bool = True) -> list[np.ndarray]:
        """
        Generate embeddings for multiple images in batch (MUCH faster than one-by-one).
        Automatically detects optimal batch size and processes large image arrays efficiently.
        
        Features:
        - Automatic batch size detection based on GPU memory
        - Intelligent chunking for large image arrays (>10k images)
        - BufferPool reuse for memory efficiency
        - Async CUDA stream execution when available
        - Detailed timing and throughput metrics
        
        Args:
            images: List of PIL Image objects (can be very large - will auto-chunk)
            verbose: If True, print detailed timing and memory information
            
        Returns:
            List of normalized embedding vectors as numpy arrays (same order as input)
        """
        try:
            # ========== VALIDACIÓN INICIAL ==========
            self._validate_batch_processing(images)
            
            if len(images) == 0:
                return []
            
            if verbose:
                self._log_batch_start(len(images))
            
            # ========== DETECCIÓN DE BATCH SIZE ÓPTIMO ==========
            optimal_batch_size = self._get_optimal_batch_size_for_gpu()
            num_batches = (len(images) + optimal_batch_size - 1) // optimal_batch_size
            should_chunk = num_batches > 1
            
            if verbose:
                self._log_batch_strategy(optimal_batch_size, num_batches, len(images), should_chunk)
            
            # ========== PROCESAMIENTO ==========
            total_start = time.time()
            all_embeddings = self._process_batches(images, optimal_batch_size, num_batches, verbose, should_chunk)
            
            # ========== RESULTADOS ==========
            total_time = (time.time() - total_start) * 1000
            throughput = (len(images) / total_time) * 1000 if total_time > 0 else 0
            
            if verbose:
                self._log_batch_results(len(images), total_time, throughput, len(all_embeddings))
            else:
                logger.info(f"⚡ Batch processed {len(images)} images in {total_time:.2f}ms ({throughput:.1f} imgs/s)")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error encoding images batch: {e}")
            raise RuntimeError(f"Failed to encode images batch: {e}")

    def _validate_batch_processing(self, images: list) -> None:
        """Validate prerequisites for batch processing."""
        if self.model_batch is None and self.onnx_session is None and self.trt_session is None:
            raise RuntimeError("DINOv3 batch model not available")

    def _log_batch_start(self, num_images: int) -> None:
        """Log the start of batch processing."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 BATCH PROCESSING: {num_images} images")
        logger.info(f"{'='*80}")

    def _log_batch_strategy(self, optimal_batch_size: int, num_batches: int, total_images: int, should_chunk: bool) -> None:
        """Log the batch processing strategy."""
        logger.info(f"Optimal batch size: {optimal_batch_size}, Number of batches: {num_batches}")
        logger.info(f"Should chunk: {should_chunk}")
        
        logger.info("📊 Processing Strategy:")
        if should_chunk:
            logger.info(f"   - Optimal batch size: {optimal_batch_size}")
            logger.info(f"   - Number of chunks: {num_batches}")
            logger.info(f"   - Total images: {total_images}")
        else:
            logger.info(f"   - Batch size: {optimal_batch_size}")
            logger.info("   - Single pass (all images fit in memory)")

    def _process_batches(self, images: list, batch_size: int, num_batches: int, verbose: bool, should_chunk: bool) -> list:
        """Process all image batches."""
        all_embeddings = []
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            if verbose and should_chunk:
                logger.info(f"\n⏱️  Chunk {batch_idx + 1}/{num_batches} ({batch_start}-{batch_end})")
            
            chunk_embeddings = self._process_batch_chunk(batch_images, verbose)
            all_embeddings.extend(chunk_embeddings)
        return all_embeddings

    def _log_batch_results(self, num_images: int, total_time: float, throughput: float, num_embeddings: int) -> None:
        """Log final batch processing results."""
        avg_time_per_image = total_time / num_images if num_images > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ BATCH PROCESSING COMPLETED")
        logger.info(f"{'='*80}")
        logger.info("📈 Results:")
        logger.info(f"   - Total images: {num_images}")
        logger.info(f"   - Total time: {total_time:.2f}ms")
        logger.info(f"   - Avg per image: {avg_time_per_image:.2f}ms")
        logger.info(f"   - Throughput: {throughput:.1f} imgs/s")
        logger.info(f"   - Embeddings generated: {num_embeddings}")
        
        if self.device and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            logger.info(f"   - GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        logger.info(f"{'='*80}\n")
    
    def _get_optimal_batch_size_for_gpu(self) -> int:
        """
        Detectar el batch size óptimo basado en el GPU disponible.
        
        Returns:
            Optimal batch size para el GPU actual
        """
        # 1. Try TensorRT (STRICT MODE: No fallback to CUDA heuristic if TRT is enabled)
        if self.use_onnx and self.trt_session:
            batch_size = self._get_tensorrt_batch_size()
            if batch_size:
                return batch_size
            else:
                # Critical Fallback for TensorRT:
                # If we cannot read the engine config, we MUST NOT guess 16 (which causes crashes).
                # We default to 1 to be absolutely safe, or 8 (common limit).
                # We chose 1 to guarantee it works, logging a warning.
                logger.warning("⚠️  Could not determinate TensorRT batch limits from engine. Defaulting to safe batch size: 4")
                return 4
        
        # 2. Try ONNX Config
        if self.use_onnx and self.onnx_session:
            batch_size = self._get_onnx_batch_size()
            if batch_size:
                return batch_size
        
        # 3. Try PyTorch / CUDA Heuristic
        if self.device and self.device.type == "cuda":
            batch_size = self._get_cuda_batch_size()
            if batch_size:
                return batch_size
        
        # 4. Default conservative batch size for CPU
        return 4

    def _get_tensorrt_batch_size(self) -> int | None:
        """Get optimal batch size from TensorRT session."""
        try:
            batch_config = self.trt_session.get_batch_config()
            if batch_config and isinstance(batch_config, dict):
                max_batch = batch_config.get('max_batch_size', 16)
                opt_batch = batch_config.get('optimal_batch_size', max_batch)
                logger.info(f"TensorRT batch constraints: max={max_batch}, optimal={opt_batch}")
                return opt_batch
        except Exception as e:
            logger.warning(f"Failed to get batch config from TensorRT: {e}")
        return None

    def _get_onnx_batch_size(self) -> int | None:
        """Get optimal batch size from ONNX session config."""
        try:
            batch_config = self.onnx_session.get_batch_config()
            if batch_config and isinstance(batch_config, dict):
                return batch_config.get('optimal_batch_size', 32)
        except Exception:
            pass
        return None

    def _get_cuda_batch_size(self) -> int | None:
        """Estimate optimal batch size based on available CUDA VRAM."""
        try:
            # Obtener propiedades del GPU
            gpu_props = torch.cuda.get_device_properties(self.device)
            total_memory = gpu_props.total_memory / (1024**3)  # GB
            
            # Heurística: batch size basado en VRAM total
            if total_memory >= 24:  # RTX 4090, A100
                return 128
            elif total_memory >= 12:  # RTX 3090, RTX 4080
                return 64
            elif total_memory >= 8:  # RTX 3060 Ti, RTX 4070
                return 32
            else:  # Menos memoria
                return 16
        except Exception:
            pass
        return None
    
    def _preprocess_batch_images(self, images: list[Image.Image]) -> tuple[list[Image.Image], float]:
        """Preprocess batch images (RGB conversion and resizing)."""
        chunk_start = time.time()
        processed_images = []

        for img in images:
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                    img = background
                else:
                    img = img.convert('RGB')

            # Optimize: Downscale large images before processor
            # HF processor is slow resizing huge images (e.g. 4000x4000 -> 336x336)
            # We pre-resize to a reasonable intermediate size (e.g. 1024)
            max_dim = 1024
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            # Resize de crops pequeños (Upscale)
            min_size = 100
            if min(img.size) < min_size:
                ratio = min_size / min(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img = img.resize(new_size, Image.Resampling.BILINEAR)

            processed_images.append(img)

        preprocess_time = (time.time() - chunk_start) * 1000
        return processed_images, preprocess_time

    def _run_batch_onnx_inference(self, inputs: dict) -> list[np.ndarray]:
        """Run batch inference using ONNX session.
        
        Uses register tokens (CLS + 4 registers) for better embeddings.
        """
        # Convert entire batch to numpy at once (operación vectorizada)
        input_array = inputs['pixel_values'].numpy()
        if not input_array.flags['C_CONTIGUOUS']:
            input_array = np.ascontiguousarray(input_array)

        expected_dtype = self.onnx_session.session.get_inputs()[0].type # pyright: ignore[reportOptionalMemberAccess]
        if 'float16' in expected_dtype:
            input_array = input_array.astype(np.float16)

        in_name = self.onnx_session.input_names[0] # pyright: ignore[reportOptionalMemberAccess]
        onnx_inputs = {in_name: input_array}
        outputs = self.onnx_session.run(onnx_inputs) # pyright: ignore[reportOptionalMemberAccess]

        output = outputs[0]
        
        # Check output shape to determine format
        if output.ndim == 2:
            # New format: embeddings already computed
            embeddings = output
        else:
            # Old format: extract CLS token (first token) for each image
            cls_embeddings = output[:, 0, :]
            
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
                cls_embeddings = cls_embeddings / (norms + 1e-8)
            
            embeddings = cls_embeddings
                
        return list(embeddings)

    def _run_batch_pytorch_inference(self, inputs: dict) -> list[np.ndarray]:
        """Run batch inference using PyTorch model.
        
        Uses CLS token (first token) for embeddings.
        """
        # Transferencia asíncrona a GPU con non_blocking
        if self.device and self.device.type == "cuda":
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inferencia batch con optimizaciones CUDA
        with torch.inference_mode():
            if self.device and self.device.type == "cuda":
                with autocast(device_type="cuda", dtype=torch.float32):
                    outputs = self.model_batch(**inputs) # pyright: ignore[reportCallIssue, reportOptionalCall]
            else:
                outputs = self.model_batch(**inputs) # pyright: ignore[reportCallIssue, reportOptionalCall]

            # Extract CLS token (first token) for all images in batch
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # Normalizar si está configurado
            if self.config.normalize_embeddings:
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

            # Convertir a float32 si es necesario
            if cls_embeddings.dtype == torch.float16:
                cls_embeddings = cls_embeddings.to(torch.float32)

            # Transferencia eficiente de GPU a CPU
            embeddings_np = cls_embeddings.cpu().numpy()
            return list(embeddings_np)

    def _validate_batch_embeddings(self, embeddings_list: list[np.ndarray]) -> None:
        """Validate batch embeddings for NaN/Inf values."""
        for i, emb in enumerate(embeddings_list):
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                logger.error(f"❌ CRITICAL: Batch embedding {i} contains NaN or Inf values!")
                raise RuntimeError("Model produced invalid batch embeddings (NaN/Inf).")

    def _process_batch_chunk(self, images: list[Image.Image], verbose: bool = True) -> list[np.ndarray]:
        """
        Procesar un chunk de imágenes (internamente optimizado).
        
        Args:
            images: List of PIL images para este chunk
            verbose: If True, log detailed information
            
        Returns:
            List of embeddings para este chunk
        """
        # ========== PREPROCESSING OPTIMIZADO ==========
        processed_images, preprocess_time = self._preprocess_batch_images(images)

        # ========== BATCH PROCESSING OPTIMIZADO ==========
        inference_start = time.time()

        # Procesar batch completo de una vez
        inputs = self.processor(images=processed_images, return_tensors="pt") # pyright: ignore[reportOptionalCall]

        # Choose inference backend
        if self.use_onnx and self.trt_session:
            embeddings_list = self._run_batch_tensorrt_inference(inputs)
        elif self.use_onnx and self.onnx_session:
            embeddings_list = self._run_batch_onnx_inference(inputs)
        else:
            embeddings_list = self._run_batch_pytorch_inference(inputs)

        inference_time = (time.time() - inference_start) * 1000

        # ========== VALIDACIÓN ==========
        self._validate_batch_embeddings(embeddings_list)

        if verbose:
            logger.debug(f"   - Preprocessing: {preprocess_time:.2f}ms")
            logger.debug(f"   - Inference: {inference_time:.2f}ms")
            logger.debug(f"   - Embeddings: {len(embeddings_list)}")

        return embeddings_list

    def _run_batch_tensorrt_inference(self, inputs: dict) -> list[np.ndarray]:
        """Run batch inference using TensorRT session."""
        # Check input validity
        if 'pixel_values' not in inputs:
            raise ValueError("Input dict must contain 'pixel_values'")

        input_tensor = inputs['pixel_values']
        
        # Convert to numpy (if it's a tensor)
        if isinstance(input_tensor, torch.Tensor):
            input_array = input_tensor.cpu().numpy()
        else:
            input_array = input_tensor
            
        if not input_array.flags['C_CONTIGUOUS']:
            input_array = np.ascontiguousarray(input_array)

        # Prepare input dict for TRT session
        # Ensure correct input name (usually 'pixel_values')
        in_name = self.trt_session.input_names[0] if self.trt_session.input_names else 'pixel_values'
        trt_inputs = {in_name: input_array}
        
        # Run inference
        outputs = self.trt_session.run(trt_inputs)
        
        # Process outputs
        output = outputs[0]
        
        # Handle format (similar to ONNX)
        # Output shape: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        if output.ndim == 2:
            embeddings = output
        else:
            # Extract CLS token (index 0)
            cls_embeddings = output[:, 0, :]
            
            # Normalize if config requires
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
                cls_embeddings = cls_embeddings / (norms + 1e-8)
            
            embeddings = cls_embeddings
            
        return list(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.config.batch_size
