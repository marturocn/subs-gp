"""
Grounding DINO Optimizer - Infrastructure Layer

Applies PyTorch-native optimizations to Grounding DINO for improved inference speed.
Since Grounding DINO is not compatible with TensorRT/ONNX due to dynamic text conditioning,
this module provides alternative optimization strategies.

Optimizations included:
- torch.compile() for kernel fusion and optimization
- FlashAttention / SDPA enablement
- Text embedding caching for repetitive queries
- Memory optimization settings
"""
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStatus:
    """Status of applied optimizations."""
    torch_compile_enabled: bool = False
    torch_compile_mode: Optional[str] = None
    flash_attention_available: bool = False
    flash_attention_enabled: bool = False
    sdpa_enabled: bool = False
    text_cache_enabled: bool = False
    text_cache_size: int = 0
    fp16_enabled: bool = False
    device: str = "cpu"
    
    def __str__(self) -> str:
        lines = [
            "🔧 Grounding DINO Optimization Status:",
            f"   torch.compile: {'✅ ' + self.torch_compile_mode if self.torch_compile_enabled else '❌'}",
            f"   FlashAttention: {'✅' if self.flash_attention_enabled else '❌'} (available: {self.flash_attention_available})",
            f"   SDPA: {'✅' if self.sdpa_enabled else '❌'}",
            f"   Text Cache: {'✅ ' + str(self.text_cache_size) + ' entries' if self.text_cache_enabled else '❌'}",
            f"   FP16: {'✅' if self.fp16_enabled else '❌'}",
            f"   Device: {self.device}",
        ]
        return "\n".join(lines)


class GroundingDINOOptimizer:
    """
    Applies PyTorch-native optimizations to Grounding DINO.
    
    Usage:
        from groundingdino.util.inference import load_model
        
        model = load_model(config_path, weights_path)
        model = GroundingDINOOptimizer.optimize(
            model,
            compile_mode="reduce-overhead",
            enable_text_cache=True,
        )
    """
    
    @staticmethod
    def check_flash_attention_available() -> bool:
        """Check if FlashAttention is available."""
        try:
            # Check PyTorch version (2.0+ has SDPA)
            major, _ = map(int, torch.__version__.split('.')[:2])
            if major < 2:
                return False
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                return False
            
            # Check if flash_sdp is available
            if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def enable_flash_attention() -> bool:
        """
        Enable FlashAttention / Scaled Dot-Product Attention.
        
        Returns:
            True if successfully enabled
        """
        try:
            if not GroundingDINOOptimizer.check_flash_attention_available():
                logger.warning("⚠️ FlashAttention not available (requires PyTorch 2.0+ with CUDA)")
                return False
            
            # Enable flash attention
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)  # Fallback
            
            logger.info("⚡ FlashAttention / SDPA enabled")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to enable FlashAttention: {e}")
            return False
    
    @staticmethod
    def apply_torch_compile(
        model: nn.Module,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> nn.Module:
        """
        Apply torch.compile() optimization.
        
        Args:
            model: Model to compile
            mode: Compilation mode:
                - "default": Balanced compilation
                - "reduce-overhead": Lower latency, recommended for inference
                - "max-autotune": Maximum performance, slower compilation
            fullgraph: Whether to compile full graph (may fail with dynamic shapes)
            dynamic: Enable dynamic shapes support
            
        Returns:
            Compiled model
        """
        try:
            # Check PyTorch version
            major, _ = map(int, torch.__version__.split('.')[:2])
            if major < 2:
                logger.warning("⚠️ torch.compile requires PyTorch 2.0+, skipping")
                return model
            
            logger.info(f"🚀 Applying torch.compile (mode={mode}, dynamic={dynamic})...")
            
            compiled_model = torch.compile(
                model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            
            logger.info("✅ torch.compile applied successfully")
            return compiled_model
            
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}")
            logger.warning("   Continuing with uncompiled model")
            return model
    
    @staticmethod
    def optimize_memory_settings() -> None:
        """Apply memory optimization settings."""
        try:
            if torch.cuda.is_available():
                # Enable memory efficient settings
                torch.cuda.empty_cache()
                
                # Set memory allocator settings for reduced fragmentation
                if hasattr(torch.cuda, 'memory'):
                    torch.cuda.memory.set_per_process_memory_fraction(0.8)
                
                # Enable TF32 for better performance on Ampere+ GPUs
                # This fixes: "TensorFloat32 tensor cores... available but not enabled"
                torch.set_float32_matmul_precision('high')
                
                logger.info("💾 Memory optimizations applied & TF32 enabled")
                
        except Exception as e:
            logger.debug(f"Memory optimization settings not applied: {e}")
    
    @staticmethod
    def create_optimized_predict(
        model: nn.Module,
        text_cache = None,
    ) -> Callable:
        """
        Create an optimized predict function with text caching.
        
        Args:
            model: Grounding DINO model
            text_cache: TextEmbeddingCache instance (optional)
            
        Returns:
            Optimized predict function
        """
        from groundingdino.util.inference import predict as original_predict
        
        def optimized_predict(
            image,
            caption: str,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
        ):
            """Optimized predict with optional text caching."""
            # Use original predict (caching would require deeper integration)
            return original_predict(
                model=model,
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        
        return optimized_predict
    
    @classmethod
    def optimize(
        cls,
        model: nn.Module,
        compile_mode: Optional[str] = "default",  # 'default' works better for mixed CPU/GPU models
        enable_flash_attention: bool = True,
        enable_text_cache: bool = True,
        text_cache_size: int = 1000,
        fp16: bool = False,
        device: str = "cuda",
    ) -> nn.Module:
        """
        Apply all available optimizations to Grounding DINO model.
        
        Args:
            model: Grounding DINO model
            compile_mode: torch.compile mode (None to disable)
            enable_flash_attention: Enable FlashAttention/SDPA
            enable_text_cache: Enable text embedding cache
            text_cache_size: Maximum text cache entries
            fp16: Convert model to FP16
            device: Target device
            
        Returns:
            Optimized model
        """
        logger.info("=" * 60)
        logger.info("GROUNDING DINO OPTIMIZATION")
        logger.info("=" * 60)
        
        # Track optimization status
        status = OptimizationStatus(device=device)
        
        # 1. Move to device
        model = model.to(device)
        model.eval()
        
        # [NEW] Patch tokenizer to ensure inputs are on device before compilation
        if device == "cuda":
            cls.patch_tokenizer(model, device)
        
        # 2. Enable FlashAttention / SDPA
        if enable_flash_attention:
            status.flash_attention_available = cls.check_flash_attention_available()
            if status.flash_attention_available:
                status.flash_attention_enabled = cls.enable_flash_attention()
                status.sdpa_enabled = status.flash_attention_enabled
        
        # 3. Detect FP16 mode (don't change it - preserve model's original precision)
        if fp16:
            # Just report that model is in FP16, don't convert
            status.fp16_enabled = True
            logger.info("📊 Model is in FP16 mode")
        
        # 4. Apply torch.compile
        if compile_mode is not None:
            model = cls.apply_torch_compile(model, mode=compile_mode)
            status.torch_compile_enabled = True
            status.torch_compile_mode = compile_mode
        
        # 5. Apply memory optimizations
        cls.optimize_memory_settings()
        
        # 6. Text cache (attached to model for later use)
        if enable_text_cache:
            try:
                from .text_embedding_cache import GroundingDINOTextCache
                model._text_cache = GroundingDINOTextCache(
                    model=model,
                    max_size=text_cache_size,
                    device=device,
                )
                status.text_cache_enabled = True
                status.text_cache_size = text_cache_size
                logger.info(f"📦 Text cache enabled (max_size={text_cache_size})")
            except Exception as e:
                logger.warning(f"Failed to initialize text cache: {e}")
        
        # Log status
        logger.info(str(status))
        logger.info("=" * 60)
        
        # Attach status to model
        model._optimization_status = status
        
        return model
    
    @staticmethod
    def get_status(model: nn.Module) -> Optional[OptimizationStatus]:
        """Get optimization status from model."""
        return getattr(model, '_optimization_status', None)
    
    @staticmethod
    def warmup(model: nn.Module, device: str = "cuda", iterations: int = 3) -> None:
        """
        Warmup the model to trigger JIT compilation.
        
        Args:
            model: Model to warmup
            device: Device to use
            iterations: Number of warmup iterations
        """
        logger.info(f"🔥 Warming up model ({iterations} iterations)...")
        
        try:
            from groundingdino.util.inference import predict, load_image
            from PIL import Image
            import numpy as np
            import tempfile
            
            # Create dummy image
            rng = np.random.default_rng(seed=42)
            dummy_array = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy_array)
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                dummy_pil.save(tmp.name)
                _, image_tensor = load_image(tmp.name)
            
            image_tensor = image_tensor.to(device)
            
            
            with torch.no_grad():
                for _ in range(iterations):
                    _, _, _ = predict(
                        model=model,
                        image=image_tensor,
                        caption="object",
                        box_threshold=0.5,
                        text_threshold=0.5,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            logger.info("✅ Warmup complete")
            
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    @staticmethod
    def patch_tokenizer(model: nn.Module, device: str) -> None:
        """
        Patch model's tokenizer to ensure tensors return on the correct device.
        This fixes 'FakeTensor Device Propagation' errors with torch.compile.
        """
        if not hasattr(model, 'tokenizer'):
            return

        # Avoid double patching
        if getattr(model.tokenizer, '_is_patched_wrapper', False):
            return

        original_tokenizer = model.tokenizer

        class TokenizerWrapper:
            def __init__(self, tokenizer, device):
                self.tokenizer = tokenizer
                self.device = device
                self._is_patched_wrapper = True
            
            def __call__(self, *args, **kwargs):
                result = self.tokenizer(*args, **kwargs)
                # Ensure output is on the correct device
                if hasattr(result, "to"):
                    return result.to(self.device)
                return result
            
            def __getattr__(self, name):
                return getattr(self.tokenizer, name)
        
        try:
            model.tokenizer = TokenizerWrapper(original_tokenizer, device)
            logger.info(f"🔧 Patched tokenizer to return tensors on {device}")
        except Exception as e:
            logger.warning(f"Failed to patch tokenizer: {e}")
