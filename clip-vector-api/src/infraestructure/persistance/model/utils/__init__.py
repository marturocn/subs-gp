"""
Model utilities for optimization and export.

This module provides utilities for:
- GPU profiling and optimization strategy selection
- ONNX model export and optimization
- TensorRT engine building and inference
- CUDA stream and buffer management
- Performance optimizations (timing cache, CUDA graphs)
"""

__all__ = [
    # Core classes
    "ONNXOptimizer",
    "ONNXInferenceSession",
    "TensorRTInferenceSession",
    
    # GPU utilities
    "GPUProfiler",
    "OptimizationStrategy",
    
    # Performance utilities
    "StreamManager",
    "BatchBufferPool",
    
    # Speed optimization classes
    "TimingCacheManager",
    "CUDAGraphManager",
    "CUDAGraphPool",
    "EngineMetadata",
    "EngineMetadataBuilder",
    
    # Grounding DINO optimizations
    "GroundingDINOOptimizer",
    "TextEmbeddingCache",
    "GroundingDINOTextCache",
    
    # Utility functions
    "validate_engine_before_load",
    
    # Constants
    "PRECISION_FP16",
    "PRECISION_FP32",
]


def __getattr__(name: str):
    """Lazy-load attributes from submodules.

    This implements PEP 562 so code like `from ...utils import ONNXOptimizer`
    will import the symbol on demand without executing the entire package
    at import time (avoids circular imports).
    """
    if name == "ONNXOptimizer":
        from .onnx_optimizer import ONNXOptimizer
        return ONNXOptimizer
    
    if name == "ONNXInferenceSession":
        from .onnx_inference_session import ONNXInferenceSession
        return ONNXInferenceSession
    
    if name == "TensorRTInferenceSession":
        from .tensorrt_inference_session import TensorRTInferenceSession
        return TensorRTInferenceSession
    
    if name == "GPUProfiler":
        from .gpu_profiler import GPUProfiler
        return GPUProfiler
    
    if name == "OptimizationStrategy":
        from .optimization_strategy import OptimizationStrategy
        return OptimizationStrategy
    
    if name == "StreamManager":
        from .stream_manager import StreamManager
        return StreamManager
    
    if name == "BatchBufferPool":
        from .batch_buffer_pool import BatchBufferPool
        return BatchBufferPool
    
    if name == "TimingCacheManager":
        from .timing_cache_manager import TimingCacheManager
        return TimingCacheManager
    
    if name == "CUDAGraphManager":
        from .cuda_graph_manager import CUDAGraphManager
        return CUDAGraphManager
    
    if name == "CUDAGraphPool":
        from .cuda_graph_manager import CUDAGraphPool
        return CUDAGraphPool
    
    if name == "EngineMetadata":
        from .engine_metadata import EngineMetadata
        return EngineMetadata
    
    if name == "EngineMetadataBuilder":
        from .engine_metadata import EngineMetadataBuilder
        return EngineMetadataBuilder
    
    if name == "validate_engine_before_load":
        from .engine_metadata import validate_engine_before_load
        return validate_engine_before_load
    
    if name == "GroundingDINOOptimizer":
        from .grounding_dino_optimizer import GroundingDINOOptimizer
        return GroundingDINOOptimizer
    
    if name == "TextEmbeddingCache":
        from .text_embedding_cache import TextEmbeddingCache
        return TextEmbeddingCache
    
    if name == "GroundingDINOTextCache":
        from .text_embedding_cache import GroundingDINOTextCache
        return GroundingDINOTextCache
    
    if name in ("PRECISION_FP16", "PRECISION_FP32"):
        from .optimization_strategy import PRECISION_FP16, PRECISION_FP32
        return PRECISION_FP16 if name == "PRECISION_FP16" else PRECISION_FP32

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Make dir() show the exported names."""
    return sorted(list(globals().keys()) + __all__)
