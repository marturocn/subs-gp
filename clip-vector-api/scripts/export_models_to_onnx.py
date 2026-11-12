"""
Export Models to ONNX Format - Dynamic GPU-Aware Optimization

Script to export DINOv3, OWL-ViT v2, and Grounding DINO models to optimized ONNX and TensorRT formats.
Supports dynamic GPU-aware optimization, FP16 conversion, CUDA/Tensor Core detection,
and TensorRT/OpenVINO acceleration.

Usage:
    python -m scripts.export_models_to_onnx_new --model dinov3 --tensorrt
    python -m scripts.export_models_to_onnx_new --model owlvit --tensorrt
    python -m scripts.export_models_to_onnx_new --model all --tensorrt
"""
import argparse
import logging
import os
import shutil
import sys
import subprocess
import urllib.request
import traceback
import json
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infraestructure.persistance.model.config.dinov3_config import DINOv3Config
from src.infraestructure.persistance.model.config.owlvit_config import OwlvitConfig
from src.infraestructure.persistance.model.utils.onnx_optimizer import (
    ONNXOptimizer,
    GPUProfiler,
    OptimizationStrategy,
    OPTIMIZATION_STRATEGY_MSG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Precision mode constants
PRECISION_FP16 = "FP16"
PRECISION_FP32 = "FP32"


def _replace_adaptive_pooling(model):
    """Replace AdaptiveAvgPool2d with SimpleMeanPool for ONNX compatibility."""
    class SimpleMeanPool(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # average over H and W -> keep dims to produce (N,C,1,1)
            return x.mean(dim=(2, 3), keepdim=True)

    # Build a map of named modules for quick parent lookup
    named = dict(model.named_modules())
    replaced = 0
    for full_name, mod in model.named_modules():
        # Skip the root empty name
        if full_name == "":
            continue
        # Identify AdaptiveAvgPool2d instances by type name to avoid import issues
        if type(mod).__name__ == "AdaptiveAvgPool2d":
            parent_name, attr = full_name.rsplit('.', 1) if '.' in full_name else ('', full_name)
            parent = named.get(parent_name, model)
            try:
                setattr(parent, attr, SimpleMeanPool())
                replaced += 1
            except Exception:
                logger.debug(f"Couldn't replace pooling module at {full_name}")

    if replaced:
        logger.info(f"🔁 Replaced {replaced} AdaptiveAvgPool2d modules with SimpleMeanPool for ONNX compatibility")


class DINOv3WithCLSToken(nn.Module):
    """
    Wrapper for DINOv3 that extracts the CLS token for embeddings.
    
    This wrapper:
    1. Extracts the CLS token (first token, position 0)
    2. Optionally L2-normalizes the result
    """
    
    def __init__(self, model: nn.Module, normalize: bool = True):
        super().__init__()
        self.model = model
        self.normalize = normalize
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that extracts CLS token.
        
        Args:
            pixel_values: Input tensor of shape (batch, 3, H, W)
            
        Returns:
            Embedding tensor of shape (batch, hidden_dim)
        """
        # Get full model output
        outputs = self.model(pixel_values)
        
        # Extract CLS token (first token)
        # Shape: (batch, num_tokens, hidden_dim) -> (batch, hidden_dim)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # L2 normalize for cosine similarity
        if self.normalize:
            cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
        
        return cls_embedding


def export_dinov3_to_onnx(use_fp32=True, optimize_tensorrt=False, optimize_openvino=False):
    """Export DINOv3 model to ONNX with GPU-aware dynamic optimizations.
    
    Args:
        use_fp32: Use FP32 precision for better embedding quality (default: True).
                  Set to False for FP16 (faster but lower precision).
        optimize_tensorrt: Also build TensorRT engine.
        optimize_openvino: Also optimize with OpenVINO.
    """
    try:
        _log_export_start("DINOv3")

        # Profile GPU and get optimization strategy
        gpu_profile = GPUProfiler.profile_gpu()
        strategy = OptimizationStrategy.get_export_strategy(gpu_profile, "dinov3", use_fp32=use_fp32)

        # Load config
        config = DINOv3Config.from_env()
        logger.info(f"Export path: {config.onnx_export_path}")
        device = torch.device(config.device if (torch.cuda.is_available() and config.device == "cuda") else "cpu")

        # Load and wrap model
        model = _load_and_wrap_dinov3_model(config, device)

        # Apply optimization strategy
        batch_size = strategy["batch_size"]
        use_fp16_strategy = strategy.get("use_fp16", False)
        opset_version = strategy.get("onnx_opset", 14)
        
        _log_optimization_strategy(batch_size, use_fp16_strategy, opset_version)

        input_shape = (batch_size, 3, 224, 224)
        dummy_input = torch.randn(input_shape, device=device)

        if use_fp16_strategy and device.type == "cuda":
            dummy_input = dummy_input.half()
            model = model.half()

        # Warmup
        logger.info("🔥 Warming up model...")
        with torch.no_grad():
            _ = model(dummy_input)

        # Export to ONNX with dynamic optimization
        logger.info("📤 Exporting to ONNX (dynamic GPU-aware optimization)...")
        logger.info("   Output: normalized embedding vector (batch, hidden_dim)")
        success = ONNXOptimizer.export_to_onnx(
            model=model,
            export_path=config.onnx_export_path,
            input_shape=input_shape,
            input_names=["pixel_values"],
            output_names=["embedding"],
            dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=opset_version,
            use_fp16=use_fp16_strategy,
            optimize=True,
            model_type="dinov3",
        )

        if not success:
            logger.error("❌ Failed to export DINOv3 to ONNX")
            return False

        logger.info("✅ DINOv3 ONNX export successful")

        # Create metadata JSON
        _create_export_metadata(config, use_fp16_strategy, optimize_tensorrt, optimize_openvino, input_shape, opset_version, model(dummy_input).shape[1])

        # TensorRT Optimization
        _optimize_tensorrt_if_needed(config, optimize_tensorrt, device, strategy)

        # Optional OpenVINO optimization
        if optimize_openvino:
            logger.warning("⚠️  OpenVINO optimization not supported for DINOv3 at this time")

        logger.info("✅ DINOv3 export completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to export DINOv3: {e}", exc_info=True)
        return False

def _log_export_start(model_name: str) -> None:
    """Log start of export process."""
    logger.info("=" * 80)
    logger.info(f"EXPORTING {model_name} MODEL TO ONNX")
    logger.info("=" * 80)

def _log_optimization_strategy(batch_size: int, use_fp16: bool, opset_version: int) -> None:
    """Log optimization strategy details."""
    logger.info(f"📊 Precision Mode: {'FP32 (high quality)' if not use_fp16 else 'FP16 (fast)'}")
    logger.info(OPTIMIZATION_STRATEGY_MSG)
    logger.info(f"   Batch Size: {batch_size}")
    logger.info(f"   Precision: {PRECISION_FP16 if use_fp16 else PRECISION_FP32}")
    logger.info(f"   ONNX OpSet: {opset_version}")

def _load_and_wrap_dinov3_model(config, device) -> nn.Module:
    """Load DINOv3 model and wrap it with CLS token extraction."""
    from transformers import AutoImageProcessor, AutoModel
    
    logger.info("📥 Loading DINOv3 model...")
    _ = AutoImageProcessor.from_pretrained(config.model_name, token=os.getenv("HF_AUTH_TOKEN", ""))
    model = AutoModel.from_pretrained(config.model_name, token=os.getenv("HF_AUTH_TOKEN", ""))
    model = model.to(device).eval()
    logger.info("✅ Model loaded successfully")

    # Replace AdaptiveAvgPool2d to improve ONNX compatibility
    _replace_adaptive_pooling(model)
    
    # Wrap model to extract CLS token for embeddings (QA consistent)
    logger.info("🔧 Wrapping model with CLS token extraction...")
    logger.info("   Using CLS token only (matching QA behavior)")
    wrapped_model = DINOv3WithCLSToken(
        model=model,
        normalize=config.normalize_embeddings
    )
    return wrapped_model.to(device).eval()

def _create_export_metadata(config, use_fp16, tensorrt, openvino, input_shape, opset, embedding_dim) -> None:
    """Create metadata JSON file for exported model."""
    import time
    metadata = {
        "model_name": config.model_name,
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "precision": "FP16" if use_fp16 else "FP32",
        "optimization": {
            "tensorrt": tensorrt,
            "openvino": openvino
        },
        "input_shape": list(input_shape),
        "opset_version": opset,
        "embedding_dim": embedding_dim
    }
    
    metadata_path = str(Path(config.onnx_export_path).with_suffix('.json'))
    try:
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"📄 Generated metadata: {metadata_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to write metadata: {e}")

def _optimize_tensorrt_if_needed(config, optimize_tensorrt, device, strategy) -> None:
    """Run TensorRT optimization if requested and supported."""
    logger.info(f"🔍 DEBUG: TRT Check - optimize_tensorrt={optimize_tensorrt}, device={device.type}, strategy_trt={strategy.get('use_tensorrt')}")
        
    if optimize_tensorrt and device.type == "cuda" and strategy.get("use_tensorrt", False):
        logger.info("\n" + "=" * 80)
        logger.info("TENSORRT OPTIMIZATION FOR DINOv3")
        logger.info("=" * 80)
        
        onnx_path = Path(config.onnx_export_path)
        engine_path = onnx_path.with_suffix(".engine")
        
        success = ONNXOptimizer.optimize_with_tensorrt_dynamic(
            onnx_path=str(onnx_path),
            output_path=str(engine_path),
            model_type="dinov3",
        )
        
        if success:
            logger.info("✅ DINOv3 TensorRT optimization successful")
        else:
            logger.warning("⚠️  DINOv3 TensorRT optimization failed, continuing without it")


def export_owlvit_to_onnx(use_fp16=True, optimize_tensorrt=False, optimize_openvino=False):
    """Export OWL-ViT v2 model to PyTorch checkpoint with dynamic optimizations."""
    # Note: optimize_openvino and optimize_tensorrt parameters kept for API compatibility
    try:
        logger.info("=" * 80)
        logger.info("EXPORTING OWL-ViT v2 MODEL")
        logger.info("=" * 80)

        # Profile GPU and get optimization strategy
        gpu_profile = GPUProfiler.profile_gpu()
        strategy = OptimizationStrategy.get_export_strategy(gpu_profile, "owlvit")

        # Load config
        config = OwlvitConfig.from_env()
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Export path: {config.onnx_export_path}")
        device = torch.device(config.device if (torch.cuda.is_available() and config.device == "cuda") else "cpu")

        # Load OWL-ViT v2 model and processor
        logger.info("📥 Loading OWL-ViT v2 model...")
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        _ = Owlv2Processor.from_pretrained(config.model_name)
        model = Owlv2ForObjectDetection.from_pretrained(config.model_name)
        
        # Move to device and eval mode
        model = model.to(device) # pyright: ignore[reportArgumentType]
        model = model.eval()

        logger.info("✅ OWL-ViT v2 model loaded successfully")

        # Apply optimization strategy
        batch_size = strategy["batch_size"]
        use_fp16_strategy = strategy.get("use_fp16", False)

        logger.info(OPTIMIZATION_STRATEGY_MSG)
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Precision: {PRECISION_FP16 if use_fp16_strategy else PRECISION_FP32}")
        logger.info(f"   Use PyTorch Optimization: {strategy.get('pytorch_optimization', False)}")

        # OWL-ViT v2 uses variable input sizes, but we'll export with a standard size
        input_shape = (batch_size, 3, 768, 768)
        dummy_input = torch.randn(input_shape, device=device)
        dummy_input_ids = torch.zeros((batch_size, 5), dtype=torch.long, device=device)

        if use_fp16_strategy and device.type == "cuda":
            dummy_input = dummy_input.half()
            model = model.half()

        # Warmup
        logger.info("🔥 Warming up OWL-ViT model...")
        with torch.no_grad():
            try:
                _ = model(input_ids=dummy_input_ids, pixel_values=dummy_input)
                logger.info("✅ Warmup successful")
            except Exception as e:
                logger.warning(f"⚠️  Warmup warning (non-critical): {e}")

        # Export to PyTorch checkpoint (OWL-ViT not ONNX-compatible due to dynamic position embeddings)
        logger.info("📤 Exporting OWL-ViT v2 to optimized format...")
        logger.info("⚠️  OWL-ViT has dynamic position embeddings - using PyTorch native export")
        
        try:
            onnx_path = Path(config.onnx_export_path)
            pt_path = onnx_path.with_stem(onnx_path.stem).with_suffix(".pt")
            metadata_path = onnx_path.with_stem(onnx_path.stem).with_suffix(".json")
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set model to eval mode
            model.eval()
            
            logger.info("📦 Exporting OWL-ViT v2 model...")
            
            with torch.no_grad():
                # Save as PyTorch checkpoint
                logger.info("  → Saving model checkpoint (.pt)...")
                torch.save(model.state_dict(), str(pt_path))
                logger.info(f"     ✅ PyTorch Checkpoint: {pt_path.name}")
                
                # Create metadata file  
                logger.info("  → Creating metadata (.json)...")
                metadata = {
                    "model_type": "owlv2-large-patch14-ensemble",
                    "format": "pytorch_checkpoint",
                    "checkpoint_file": pt_path.name,
                    "precision": "fp16" if use_fp16_strategy else "fp32",
                    "input_type": "image",
                    "note": "Use PyTorch checkpoint for inference. TensorRT not supported.",
                    "strategy": strategy,
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"     ✅ Metadata: {metadata_path.name}")
                
            logger.info("✅ OWL-ViT v2 exported successfully")
            logger.info("="*80)
            logger.info("✅ OWL-ViT will use PyTorch + CUDA inference")
            logger.info(f"   Checkpoint: {pt_path.name}")
            logger.info(f"   Metadata: {metadata_path.name}")
            logger.info(f"   Precision: {PRECISION_FP16 if use_fp16_strategy else PRECISION_FP32}")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            traceback.print_exc()
            return False

    except Exception as e:
        logger.error(f"Failed to export OWL-ViT v2: {e}", exc_info=True)
        return False


def export_grounding_dino_to_onnx(use_fp16=True, optimize_tensorrt=False, optimize_openvino=False):
    """Export Grounding DINO model to PyTorch checkpoint with dynamic optimizations."""
    # Note: optimize_openvino and optimize_tensorrt parameters kept for API compatibility
    try:
        logger.info("=" * 80)
        logger.info("EXPORTING GROUNDING DINO MODEL")
        logger.info("=" * 80)

        # Profile GPU and get optimization strategy
        gpu_profile = GPUProfiler.profile_gpu()
        strategy = OptimizationStrategy.get_export_strategy(gpu_profile, "grounding_dino")

        # Load config
        from src.infraestructure.persistance.model.config.grounding_dino_config import GroundingDinoConfig
        config = GroundingDinoConfig.from_env()
        logger.info("Model: Grounding DINO (text-conditional detector)")
        logger.info(f"Export path: {config.onnx_export_path}")
        device = torch.device(config.device if (torch.cuda.is_available() and config.device == "cuda") else "cpu")

        # Load Grounding DINO model
        logger.info("📥 Loading Grounding DINO model...")
        from groundingdino.util.inference import load_model
        
        config_path, model_path = _download_grounding_dino_files(config)
        
        model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=model_path,
            device=str(device)
        )
        
        # Move to device and eval mode
        model = model.to(device)
        model = model.eval()

        logger.info("✅ Grounding DINO model loaded successfully")

        # Apply optimization strategy
        batch_size = strategy["batch_size"]
        use_fp16_strategy = strategy.get("use_fp16", False)

        logger.info(OPTIMIZATION_STRATEGY_MSG)
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Precision: {'FP16' if use_fp16_strategy else 'FP32'}")
        
        # IMPORTANT: Save reference to original model state dict BEFORE torch.compile
        # torch.compile adds '_orig_mod.' prefix to keys which breaks checkpoint loading
        original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        logger.info("📋 Saved original state dict for checkpoint export")
        
        # Apply PyTorch-native optimizations (torch.compile, FlashAttention, text cache)
        if device.type == "cuda":
            try:
                from src.infraestructure.persistance.model.utils import GroundingDINOOptimizer
                
                logger.info("🚀 Applying PyTorch-native optimizations...")
                model = GroundingDINOOptimizer.optimize(
                    model,
                    compile_mode=None,  # Disabled - causes hangs with Grounding DINO
                    enable_flash_attention=True,
                    enable_text_cache=True,
                    text_cache_size=1000,
                    fp16=False,  # Don't convert to FP16 yet (warmup needs FP32)
                    device="cuda",
                )
                
                # Log optimization status
                status = GroundingDINOOptimizer.get_status(model)
                if status:
                    logger.info("✅ Optimizations applied successfully:")
                    logger.info(str(status))
                else:
                    logger.info("✅ GroundingDINOOptimizer applied (no status available)")
                    
            except ImportError as e:
                logger.warning(f"⚠️  Could not import GroundingDINOOptimizer: {e}")
                logger.warning("   Continuing without optimizations")
            except Exception as e:
                logger.warning(f"⚠️  Failed to apply optimizations: {e}")
                logger.warning("   Continuing without optimizations")
        else:
            logger.info("ℹ️  Skipping optimizations (device is not CUDA)")

        # Grounding DINO uses variable input sizes
        input_shape = (batch_size, 3, 800, 800)
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup BEFORE FP16 conversion - Grounding DINO predict requires FP32
        logger.info("🔥 Warming up Grounding DINO model...")
        with torch.no_grad():
            try:
                # Grounding DINO expects preprocessed image tensor + caption
                from groundingdino.util.inference import predict, load_image
                import numpy as np
                from PIL import Image
                import tempfile
                
                # Create a temporary dummy image file
                rng = np.random.default_rng(seed=42)
                dummy_array = rng.integers(0, 255, (800, 800, 3), dtype=np.uint8)
                dummy_pil = Image.fromarray(dummy_array)
                
                # Save to temp file and load with groundingdino's loader
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    dummy_pil.save(tmp.name)
                    _, image_tensor = load_image(tmp.name)
                
                # Keep FP32 for warmup (groundingdino predict works in FP32)
                image_tensor = image_tensor.to(device)
                
                # Run predict with proper inputs
                _, _, _ = predict(
                    model=model,
                    image=image_tensor,
                    caption="object",
                    box_threshold=0.5,
                    text_threshold=0.5,
                )
                logger.info("✅ Warmup successful")
            except Exception as e:
                logger.warning(f"⚠️  Warmup warning (non-critical): {e}")
        
        # Convert to FP16 AFTER warmup for export
        if use_fp16_strategy and device.type == "cuda":
            dummy_input = dummy_input.half()
            model = model.half()
            
            # Also convert original_state_dict to FP16
            logger.info("   Converting checkpoint state dict to FP16...")
            for k, v in original_state_dict.items():
                 if torch.is_floating_point(v):
                    original_state_dict[k] = v.half()

        # Export to PyTorch checkpoint
        logger.info("📤 Exporting Grounding DINO to optimized format...")
        logger.info("⚠️  Grounding DINO uses text conditioning - exporting as PyTorch checkpoint")
        
        onnx_path = Path(config.onnx_export_path)
        pt_path = onnx_path.with_stem(onnx_path.stem).with_suffix(".pt")
        metadata_path = onnx_path.with_stem(onnx_path.stem).with_suffix(".json")
        
        try:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set model to eval mode
            model.eval()
            
            logger.info("📦 Exporting Grounding DINO model...")
            
            with torch.no_grad():
                logger.info("  → Saving model checkpoint (.pt)...")
                # Use original state dict to avoid '_orig_mod.' prefix from torch.compile
                torch.save(original_state_dict, str(pt_path))
                logger.info(f"     ✅ PyTorch Checkpoint: {pt_path.name}")
                
                logger.info("  → Creating metadata (.json)...")
                metadata = {
                    "model_type": "grounding-dino",
                    "format": "pytorch_checkpoint",
                    "checkpoint_file": pt_path.name,
                    "precision": "fp16" if use_fp16_strategy else "fp32",
                    "input_type": "text_image",
                    "note": "Use PyTorch checkpoint for inference. Text conditioning requires dynamic shapes.",
                    "strategy": strategy,
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"     ✅ Metadata: {metadata_path.name}")
                
            logger.info("✅ Grounding DINO exported successfully")
            logger.info("="*80)
            logger.info("✅ Grounding DINO will use PyTorch + CUDA inference")
            logger.info(f"   Checkpoint: {pt_path.name}")
            logger.info(f"   Metadata: {metadata_path.name}")
            logger.info(f"   Precision: {PRECISION_FP16 if use_fp16_strategy else PRECISION_FP32}")
            logger.info("   Optimizations: Text conditioning, Dynamic shapes")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            traceback.print_exc()
            return False

    except Exception as e:
        logger.error(f"Failed to export Grounding DINO: {e}", exc_info=True)
        return False


def _download_grounding_dino_files(config) -> tuple:
    """Helper function to download Grounding DINO model files if needed."""
    from pathlib import Path

    config_path = Path(config.config_path)
    model_path = Path(config.model_path)

    config_dir = config_path.parent
    weights_dir = model_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download config if needed
    if not config_path.exists():
        if not getattr(config, "auto_download", True):
            raise RuntimeError(f"Grounding DINO config not found at {config_path} and auto-download is disabled")
        
        logger.info("📥 Downloading Grounding DINO config...")
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        urllib.request.urlretrieve(config_url, str(config_path))
        logger.info(f"✅ Config downloaded to {config_path}")
    
    # Download model if needed
    if not model_path.exists():
        if not getattr(config, "auto_download", True):
            raise RuntimeError(f"Grounding DINO model not found at {model_path} and auto-download is disabled")
        
        logger.info("📥 Downloading Grounding DINO weights (~700MB)...")
        model_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        
        def report_progress(block_num, block_size, total_size):
            if block_num % 50 == 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) / total_size)
                logger.info(f"  Progress: {percent:.1f}%")
        
        urllib.request.urlretrieve(model_url, str(model_path), reporthook=report_progress)
        logger.info(f"✅ Model downloaded to {model_path}")
    
    return str(config_path), str(model_path)


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX/TensorRT formats with dynamic GPU optimization")
    parser.add_argument("--model", type=str, choices=["dinov3", "owlvit", "grounding_dino", "all"], default="dinov3",
                        help="Model to export (default: dinov3)")
    parser.add_argument("--fp16", action="store_true", default=False, 
                        help="Use FP16 precision (faster but lower quality for DINOv3 embeddings)")
    parser.add_argument("--fp32", action="store_true", default=True,
                        help="Force FP32 precision (better quality, default for DINOv3)")
    parser.add_argument("--tensorrt", action="store_true", help="Also export TensorRT engine")
    parser.add_argument("--openvino", action="store_true", help="Also optimize with OpenVINO")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("DYNAMIC GPU-AWARE MODEL EXPORT TO ONNX/TENSORRT")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    # Determine precision: --fp16 forces FP16, --fp32 forces FP32, default is FP32 for DINOv3
    use_fp32 = not args.fp16 if args.fp16 else (args.fp32 or True)
    logger.info(f"Precision: {'FP16 (fast)' if args.fp16 else 'FP32 (quality)'}")
    logger.info(f"TensorRT: {args.tensorrt}")
    logger.info(f"OpenVINO: {args.openvino}")
    logger.info("=" * 80)

    success = True

    if args.model in ["dinov3", "all"]:
        success &= export_dinov3_to_onnx(
            use_fp32=use_fp32,
            optimize_tensorrt=args.tensorrt,
            optimize_openvino=args.openvino,
        )

    if args.model == "owlvit":
        success &= export_owlvit_to_onnx(
            use_fp16=args.fp16,
            optimize_tensorrt=args.tensorrt,
            optimize_openvino=args.openvino,
        )

    if args.model in ["grounding_dino", "all"]:
        success &= export_grounding_dino_to_onnx(
            use_fp16=args.fp16,
            optimize_tensorrt=args.tensorrt,
            optimize_openvino=args.openvino,
        )

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL EXPORTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("Dynamic GPU-aware optimization applied to all models")
        logger.info("")
        logger.info("Set environment variables to use optimized models in the service:")
        logger.info("  DINOV3_USE_ONNX=true")
        logger.info("  DINOV3_ONNX_PROVIDER=tensorrt  # or cuda, cpu, auto")
        logger.info("  OWLVIT_USE_ONNX=false")
        logger.info("  OWLVIT_PYTORCH_FP16=true")
        logger.info("  DINO_USE_ONNX=false")
        logger.info("  DINO_PYTORCH_FP16=true")
    else:
        logger.error("\n" + "=" * 80)
        logger.error("❌ SOME EXPORTS FAILED")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
