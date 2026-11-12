"""
Optimization Strategy - Infrastructure Layer

Selects optimization strategy based on GPU profile for different model types.
"""
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Precision constants
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"


def _get_cpu_strategy(_: str) -> Dict[str, Any]:
    """Get optimization strategy for CPU inference."""
    return {
        "device": "cpu",
        "precision": PRECISION_FP32,
        "batch_size": 1,
        "use_fp16": False,
        "use_tensorrt": False,
        "use_openvino": True,
        "onnx_opset": 13,
        "optimize_graph": True,
        "num_threads": os.cpu_count() or 4,
    }


def _create_cuda_fp16_strategy(batch_size: int, workspace_mb: int) -> Dict[str, Any]:
    """Create a standard CUDA FP16 strategy with varying batch sizes and workspace."""
    return {
        "device": "cuda",
        "precision": PRECISION_FP16,
        "batch_size": batch_size,
        "use_fp16": True,
        "use_tensorrt": True,
        "tensorrt_precision": PRECISION_FP16,
        "tensorrt_batch_min": 1,
        "tensorrt_batch_opt": min(16, batch_size),
        "tensorrt_batch_max": batch_size,
        "tensorrt_workspace_mb": workspace_mb,
        "use_openvino": False,
        "onnx_opset": 17,
        "optimize_graph": True,
        "enable_cuda_graphs": True,
        "cuda_stream_optimization": True,
    }


def _create_cuda_fp32_strategy(batch_size: int, workspace_mb: int) -> Dict[str, Any]:
    """Create a CUDA FP32 strategy for better precision in embedding quality.
    
    FP32 provides ~7-8 decimal digits of precision vs ~3-4 for FP16, which is
    critical for cosine similarity calculations in embedding searches.
    """
    return {
        "device": "cuda",
        "precision": PRECISION_FP32,
        "batch_size": batch_size,
        "use_fp16": False,  # Keep FP32 for quality
        "use_tensorrt": True,
        "tensorrt_precision": PRECISION_FP32,
        "tensorrt_batch_min": 1,
        "tensorrt_batch_opt": min(8, batch_size),
        "tensorrt_batch_max": batch_size,
        "tensorrt_workspace_mb": workspace_mb,
        "use_openvino": False,
        "onnx_opset": 17,
        "optimize_graph": True,
        "enable_cuda_graphs": True,
        "cuda_stream_optimization": True,
    }


def _get_dinov3_strategy(vram_gb: float, _: int, __: Dict[str, Any], use_fp32: bool = True) -> Dict[str, Any]:
    """Get DINOv3-specific optimization strategy.
    
    Args:
        vram_gb: Available VRAM in GB
        use_fp32: If True, use FP32 for better embedding quality (default).
                  If False, use FP16 for faster inference at cost of precision.
    """
    # Choose strategy factory based on precision preference
    create_strategy = _create_cuda_fp32_strategy if use_fp32 else _create_cuda_fp16_strategy
    
    # DINOv3 is relatively lightweight but Tesla T4 needs conservative settings
    if vram_gb >= 24:
        return create_strategy(batch_size=32, workspace_mb=8192)
    elif vram_gb >= 16:
        # Tesla T4: reduce batch size significantly to avoid crashes
        return create_strategy(batch_size=12, workspace_mb=2048)
    elif vram_gb >= 12:
        return create_strategy(batch_size=8, workspace_mb=1024)
    elif vram_gb >= 8:
        return create_strategy(batch_size=4, workspace_mb=512)
    else:
        return create_strategy(batch_size=2, workspace_mb=256)


def _get_owlvit_strategy(vram_gb: float, _: int, __: Dict[str, Any]) -> Dict[str, Any]:
    """Get OWL-ViT-specific optimization strategy."""
    
    # OWL-ViT is heavier, requires more memory
    if vram_gb >= 24:
        return {
            "device": "cuda",
            "precision": PRECISION_FP16,
            "batch_size": 16,
            "use_fp16": True,
            "use_tensorrt": False,  # OWL-ViT not ONNX-compatible, use PyTorch
            "pytorch_optimization": True,
            "pytorch_compile": True,  # Experimental: PyTorch 2.0+ compilation
            "use_openvino": False,
            "onnx_opset": 17,
            "optimize_graph": True,
            "enable_cuda_graphs": True,
            "cuda_stream_optimization": True,
        }
    elif vram_gb >= 12:  # 12-24GB: Tesla T4 and similar - conservative batch size
        return {
            "device": "cuda",
            "precision": PRECISION_FP16,
            "batch_size": 4,
            "use_fp16": True,
            "use_tensorrt": False,
            "pytorch_optimization": True,
            "pytorch_compile": False,
            "use_openvino": False,
            "onnx_opset": 14,
            "optimize_graph": True,
            "enable_cuda_graphs": False,
            "cuda_stream_optimization": True,
        }
    elif vram_gb >= 8:
        return {
            "device": "cuda",
            "precision": PRECISION_FP16,
            "batch_size": 2,
            "use_fp16": True,
            "use_tensorrt": False,
            "pytorch_optimization": True,
            "pytorch_compile": False,
            "use_openvino": False,
            "onnx_opset": 14,
            "optimize_graph": True,
            "enable_cuda_graphs": False,
            "cuda_stream_optimization": False,
        }
    else:  # < 8GB
        return {
            "device": "cuda",
            "precision": PRECISION_FP32,
            "batch_size": 1,
            "use_fp16": False,
            "use_tensorrt": False,
            "pytorch_optimization": False,
            "pytorch_compile": False,
            "use_openvino": False,
            "onnx_opset": 13,
            "optimize_graph": False,
            "enable_cuda_graphs": False,
            "cuda_stream_optimization": False,
        }


def _get_grounding_dino_strategy(vram_gb: float, _: int, __: Dict[str, Any]) -> Dict[str, Any]:
    """Get Grounding DINO-specific optimization strategy."""
    
    # Grounding DINO is heavy due to text conditioning
    # Tesla T4 needs very conservative settings
    if vram_gb >= 16:
        # Tesla T4: reduce batch size to avoid crashes
        enable_cuda_graphs = False
        batch_size = 6
    elif vram_gb >= 8:
        enable_cuda_graphs = False
        batch_size = 4
    else:
        enable_cuda_graphs = False
        batch_size = 2
    
    return {
        "device": "cuda",
        "precision": PRECISION_FP16,
        "batch_size": batch_size,
        "use_fp16": True,
        "use_tensorrt": False,  # Text conditioning not ONNX-compatible
        "pytorch_optimization": True,
        "pytorch_compile": False,  # Complex text encoding, compile might not help
        "use_openvino": False,
        "onnx_opset": 17,
        "optimize_graph": True,
        "enable_cuda_graphs": enable_cuda_graphs,  # Scale optimization based on VRAM
        "cuda_stream_optimization": False,  # Conservative for stability
    }


def _get_default_strategy(vram_gb: float, _: int, __: Dict[str, Any]) -> Dict[str, Any]:
    """Get default optimization strategy."""
    if vram_gb >= 24:
        return {
            "device": "cuda",
            "precision": PRECISION_FP16,
            "batch_size": 16,
            "use_fp16": True,
            "use_tensorrt": True,
            "tensorrt_precision": PRECISION_FP16,
            "tensorrt_workspace_mb": 4096,
            "use_openvino": False,
            "onnx_opset": 17,
            "optimize_graph": True,
            "enable_cuda_graphs": True,
            "cuda_stream_optimization": True,
        }
    elif vram_gb >= 16:
        # Tesla T4 and similar: conservative settings
        return {
            "device": "cuda",
            "precision": PRECISION_FP16,
            "batch_size": 12,
            "use_fp16": True,
            "use_tensorrt": False,  # Disable TensorRT for stability on T4
            "tensorrt_precision": PRECISION_FP16,
            "tensorrt_workspace_mb": 2048,
            "use_openvino": False,
            "onnx_opset": 14,
            "optimize_graph": True,
            "enable_cuda_graphs": False,
            "cuda_stream_optimization": False,
        }
    else:
        return {
            "device": "cuda",
            "precision": PRECISION_FP16,
            "batch_size": 8,
            "use_fp16": True,
            "use_tensorrt": False,
            "tensorrt_precision": PRECISION_FP16,
            "tensorrt_workspace_mb": 1024,
            "use_openvino": False,
            "onnx_opset": 14,
            "optimize_graph": True,
            "enable_cuda_graphs": False,
            "cuda_stream_optimization": False,
        }


class OptimizationStrategy:
    """Selects optimization strategy based on GPU profile."""
    
    @staticmethod
    def get_export_strategy(
        gpu_profile: Dict[str, Any], 
        model_type: str = "dinov3",
        use_fp32: bool = True,
    ) -> Dict[str, Any]:
        """
        Get dynamic export optimization strategy based on GPU profile.
        
        Args:
            gpu_profile: GPU profile from GPUProfiler
            model_type: Type of model ("dinov3", "owlvit", "grounding_dino")
            use_fp32: If True, use FP32 for DINOv3 export (better quality, default).
                      If False, use FP16 for faster inference.
            
        Returns:
            Dictionary with optimization parameters
        """
        if not gpu_profile["available"]:
            return _get_cpu_strategy(model_type)
        
        vram_gb = gpu_profile["vram_gb"]
        cc_major = gpu_profile["compute_capability"][0]
        
        # Model-specific strategies
        if model_type == "dinov3":
            return _get_dinov3_strategy(vram_gb, cc_major, gpu_profile, use_fp32=use_fp32)
        elif model_type == "owlvit":
            return _get_owlvit_strategy(vram_gb, cc_major, gpu_profile)
        elif model_type == "grounding_dino":
            return _get_grounding_dino_strategy(vram_gb, cc_major, gpu_profile)
        else:
            return _get_default_strategy(vram_gb, cc_major, gpu_profile)
