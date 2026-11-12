"""
ONNX Inference Session - Infrastructure Layer

Wrapper for ONNX Runtime inference session with optimizations.
"""
import logging
from typing import Dict, Any, Optional

import numpy as np

from .gpu_profiler import GPUProfiler
from .stream_manager import StreamManager
from .batch_buffer_pool import BatchBufferPool

logger = logging.getLogger(__name__)


def _get_dynamic_cuda_options(gpu_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate dynamic CUDA options based on GPU profile.
    
    Args:
        gpu_profile: Dictionary with GPU information
        
    Returns:
        Dictionary with optimized CUDA options
    """
    if not gpu_profile.get("available"):
        return {}
    
    vram_gb = gpu_profile.get("vram_gb", 4)
    
    # Calculate gpu_mem_limit: use 65% of available VRAM for GPU <= 16GB
    # Leave margin for CUDA overhead, TensorRT workspace and fragmentation
    # Tesla T4 with 16GB needs to be conservative
    if vram_gb <= 16:
        gpu_mem_limit_percent = 0.65  # More conservative for T4
    else:
        gpu_mem_limit_percent = 0.70
    
    gpu_mem_limit_bytes = int(vram_gb * gpu_mem_limit_percent * (1024 ** 3))
    
    # Limit to max 10GB for small GPUs (T4, RTX 3060, etc)
    max_limit = 10 * 1024 * 1024 * 1024 if vram_gb <= 16 else 16 * 1024 * 1024 * 1024
    gpu_mem_limit_bytes = min(gpu_mem_limit_bytes, max_limit)
    
    logger.info(f"🧠 Dynamic CUDA memory limit: {gpu_mem_limit_bytes / (1024**3):.1f} GB (from {vram_gb} GB total)")
    
    cuda_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': gpu_mem_limit_bytes,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
        'cudnn_conv_use_max_workspace': '1',
        'cudnn_conv1d_pad_to_nc1d': '1',
    }
    
    return cuda_options


def _get_dynamic_tensorrt_options(gpu_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate dynamic TensorRT options based on GPU profile.
    
    Args:
        gpu_profile: Dictionary with GPU information
        
    Returns:
        Dictionary with optimized TensorRT options
    """
    if not gpu_profile.get("available"):
        return {}
    
    vram_gb = gpu_profile.get("vram_gb", 4)
    has_tensor_cores = gpu_profile.get("tensor_cores", False)
    
    # Calculate workspace dynamically: be MORE CONSERVATIVE for small GPUs
    # Tesla T4 (16GB) needs small workspace to avoid crashes on batch > 13
    if vram_gb <= 16:
        workspace_percent = 0.15  # Only 15% for T4 (2.4GB of 16GB)
    elif vram_gb >= 24:
        workspace_percent = 0.30  # 30% for large GPUs
    else:
        workspace_percent = 0.20  # 20% for medium GPUs (12-24GB)
    
    workspace_size_bytes = int(vram_gb * workspace_percent * (1024 ** 3))
    
    # Conservative max limits per GPU
    if vram_gb <= 16:
        max_workspace = 3 * 1024 * 1024 * 1024  # 3GB for T4
    elif vram_gb <= 24:
        max_workspace = 6 * 1024 * 1024 * 1024  # 6GB for medium GPUs
    else:
        max_workspace = 8 * 1024 * 1024 * 1024  # 8GB for large GPUs
    
    workspace_size_bytes = min(workspace_size_bytes, max_workspace)
    
    # FP16 only if it has tensor cores (Volta+)
    fp16_enabled = has_tensor_cores
    
    logger.info(f"🚀 Dynamic TensorRT workspace: {workspace_size_bytes / (1024**3):.1f} GB (from {vram_gb} GB total)")
    logger.info(f"   FP16 enabled: {fp16_enabled} (Tensor Cores: {has_tensor_cores})")
    
    trt_options = {
        'device_id': 0,
        'trt_max_workspace_size': workspace_size_bytes,
        'trt_fp16_enable': fp16_enabled,
        'trt_int8_enable': False,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache',
        'trt_max_partition_iterations': 1000,
        'trt_min_subgraph_size': 10,
        'trt_dla_enable': False,
        'trt_timing_cache_enable': True,
        'trt_force_sequential_engine_build': False,
    }
    
    return trt_options


def _get_dynamic_provider_options(
    providers: list[str], 
    gpu_profile: Optional[Dict[str, Any]] = None
) -> list[Optional[Dict[str, Any]]]:
    """
    Generate provider options dynamically based on GPU profile.
    
    Args:
        providers: List of available providers
        gpu_profile: GPU profile (None for auto-detection)
        
    Returns:
        List of options per provider
    """
    if gpu_profile is None:
        gpu_profile = GPUProfiler.profile_gpu()
    
    provider_options = []
    
    for provider_name in providers:
        if provider_name == "CUDAExecutionProvider":
            options = _get_dynamic_cuda_options(gpu_profile)
            provider_options.append(options)
            if options:
                logger.info("✅ CUDA provider with DYNAMIC optimizations enabled")
            else:
                logger.info("⚠️  CUDA provider: no GPU available, using defaults")
                
        elif provider_name == "TensorrtExecutionProvider":
            options = _get_dynamic_tensorrt_options(gpu_profile)
            provider_options.append(options)
            if options:
                logger.info("✅ TensorRT provider with DYNAMIC optimizations enabled (expect 3-5x speedup)")
            else:
                logger.info("⚠️  TensorRT provider: no GPU available, using defaults")
                
        else:
            # For other providers
            provider_options.append({})
    
    return provider_options


def _get_batch_optimization_config(gpu_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate optimization configuration for batch processing.
    
    Args:
        gpu_profile: GPU profile (None for auto-detect)
        
    Returns:
        Dictionary with optimized batch configuration
    """
    if gpu_profile is None:
        gpu_profile = GPUProfiler.profile_gpu()
    
    vram_gb = gpu_profile.get("vram_gb", 4)
    
    # Determine optimal batch size based on VRAM
    # IMPORTANT: Tesla T4 with 16GB needs small batches to avoid crashes
    if vram_gb >= 24:
        batch_config = {
            "optimal_batch_size": 64,
            "max_batch_size": 128,
            "prefetch_buffer_batches": 4,
            "buffer_pool_multiplier": 3,
            "num_streams": 2,
            "use_async_execution": True,
            "memory_fraction_per_batch": 0.08,  # 8% of VRAM per batch
        }

    elif vram_gb >= 12:
        batch_config = {
            "optimal_batch_size": 12,      # Reduced from 32 to 12
            "max_batch_size": 16,          # Reduced from 64 to 16 (doesn't exceed 13 which fails)
            "prefetch_buffer_batches": 2,
            "buffer_pool_multiplier": 1,
            "num_streams": 1,
            "use_async_execution": False,  # Disable async on T4
            "memory_fraction_per_batch": 0.13,  # 13% of VRAM per batch
        }
    elif vram_gb >= 8:
        batch_config = {
            "optimal_batch_size": 4,
            "max_batch_size": 8,
            "prefetch_buffer_batches": 1,
            "buffer_pool_multiplier": 1,
            "num_streams": 1,
            "use_async_execution": False,
            "memory_fraction_per_batch": 0.18,  # 18% of VRAM per batch
        }
    else:
        batch_config = {
            "optimal_batch_size": 2,
            "max_batch_size": 4,
            "prefetch_buffer_batches": 1,
            "buffer_pool_multiplier": 1,
            "num_streams": 1,
            "use_async_execution": False,
            "memory_fraction_per_batch": 0.20,  # 20% of VRAM per batch
        }
    
    logger.info(f"⚡ Batch Optimization Config (VRAM: {vram_gb:.1f}GB):")
    logger.info(f"   Optimal Batch Size: {batch_config['optimal_batch_size']}")
    logger.info(f"   Max Batch Size: {batch_config['max_batch_size']}")
    logger.info(f"   Prefetch Buffers: {batch_config['prefetch_buffer_batches']}")
    logger.info(f"   CUDA Streams: {batch_config['num_streams']}")
    logger.info(f"   Async Execution: {batch_config['use_async_execution']}")
    
    return batch_config


def _get_speed_optimization_config(gpu_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate optimization configuration for maximum speed.
    
    Args:
        gpu_profile: GPU profile (None for auto-detect)
        
    Returns:
        Dictionary with speed-optimized configuration
    """
    if gpu_profile is None:
        gpu_profile = GPUProfiler.profile_gpu()
    
    compute_cap = gpu_profile.get("compute_capability", (8, 0))
    cc_major = compute_cap[0]
    has_tensor_cores = gpu_profile.get("tensor_cores", False)
    has_rt_cores = gpu_profile.get("rt_cores", False)
    
    speed_config = {
        # Kernel optimization
        "enable_kernel_fusion": has_tensor_cores,  # Fuse kernels on Volta+
        "enable_graph_capture": has_tensor_cores,  # CUDA Graph on Volta+
        "enable_persistent_kernels": has_tensor_cores,  # Persistent kernels
        
        # Memory optimization
        "enable_memory_pooling": True,  # CUDA memory pool
        "enable_pinned_memory": True,   # Host-pinned memory for transfers
        "enable_memory_caching": has_tensor_cores,  # Allocation caching
        
        # Execution optimization
        "enable_graph_optimization": True,  # Optimize ONNX graph
        "enable_operator_fusion": True,     # Fuse operators
        "enable_tensor_cores_utilization": has_tensor_cores,
        "enable_rt_cores_utilization": has_rt_cores,
        
        # Concurrency
        "enable_concurrent_streams": cc_major >= 8,  # Concurrent streams
        "max_concurrent_streams": 4 if cc_major >= 8 else 1,
        
        # Precision
        "force_fp16": has_tensor_cores,  # Force FP16 if possible
        "use_tensor_core_primitives": has_tensor_cores,
    }
    
    logger.info(f"🚀 Speed Optimization Config (Compute {compute_cap[0]}.{compute_cap[1]}):")
    logger.info(f"   Kernel Fusion: {speed_config['enable_kernel_fusion']}")
    logger.info(f"   CUDA Graphs: {speed_config['enable_graph_capture']}")
    logger.info(f"   Memory Pooling: {speed_config['enable_memory_pooling']}")
    logger.info(f"   Pinned Memory: {speed_config['enable_pinned_memory']}")
    logger.info(f"   Tensor Cores: {speed_config['enable_tensor_cores_utilization']}")
    
    return speed_config


class ONNXInferenceSession:
    """
    Wrapper for ONNX Runtime inference session with optimizations.
    """
    
    @staticmethod
    def _configure_session_options(sess_options) -> None:
        """Configure ONNX Runtime session options."""
        import onnxruntime as ort
        
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        logger.info("✅ ONNX Runtime optimizations: Graph=ALL, ExecutionMode=PARALLEL, Threads=4x4")
    
    @staticmethod
    def _initialize_batch_optimization(enable_batch_optimization: bool, gpu_profile: Dict[str, Any]):
        """Initialize batch optimization configuration."""
        batch_config = None
        buffer_pool = None
        
        if enable_batch_optimization:
            batch_config = _get_batch_optimization_config(gpu_profile)
            
            if gpu_profile.get("available"):
                try:
                    input_shape = (3, 224, 224)
                    buffer_pool = BatchBufferPool(
                        batch_size=batch_config["optimal_batch_size"],
                        input_shape=input_shape,
                        num_buffers=batch_config["prefetch_buffer_batches"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize buffer pool: {e}")
                    buffer_pool = None
        
        return batch_config, buffer_pool
    
    @staticmethod
    def _initialize_speed_optimization(enable_speed_optimization: bool, gpu_profile: Dict[str, Any]):
        """Initialize speed optimization configuration."""
        import torch
        
        speed_config = None
        stream_manager = None
        
        if enable_speed_optimization:
            speed_config = _get_speed_optimization_config(gpu_profile)
            
            if gpu_profile.get("available") and torch.cuda.is_available():
                num_streams = speed_config.get("max_concurrent_streams", 2)
                stream_manager = StreamManager(num_streams=num_streams)
        
        return speed_config, stream_manager
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[list[str]] = None,
        provider_options: Optional[list[Optional[Dict[str, Any]]]] = None,
        gpu_profile: Optional[Dict[str, Any]] = None,
        enable_batch_optimization: bool = True,
        enable_speed_optimization: bool = True,
    ):
        """
        Initialize ONNX inference session with dynamic GPU-aware optimizations.
        
        Args:
            model_path: Path to ONNX model
            providers: List of execution providers
            provider_options: Options for each provider (if None, auto-generate from GPU profile)
            gpu_profile: GPU profile dict (if None, auto-detect with GPUProfiler)
            enable_batch_optimization: Enable batch processing optimizations
            enable_speed_optimization: Enable speed optimizations
        """
        try:
            import onnxruntime as ort
            
            # Import here to avoid circular dependency
            from .onnx_optimizer import ONNXOptimizer
            
            if gpu_profile is None:
                gpu_profile = GPUProfiler.profile_gpu()
            
            self.gpu_profile = gpu_profile
            
            if providers is None:
                providers = ONNXOptimizer.get_onnx_providers()
            
            logger.info(f"Loading ONNX model: {model_path}")
            logger.info(f"Providers: {providers}")
            
            sess_options = ort.SessionOptions()
            ONNXInferenceSession._configure_session_options(sess_options)
            
            if provider_options is None:
                provider_options = _get_dynamic_provider_options(providers, gpu_profile)
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=providers,
                provider_options=[p if p else {} for p in provider_options] if provider_options else None,
            )
            
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info("✅ ONNX session created")
            logger.info(f"   Inputs: {self.input_names}")
            logger.info(f"   Outputs: {self.output_names}")
            logger.info(f"   Active providers: {self.session.get_providers()}")
            
            self.batch_config, self.buffer_pool = ONNXInferenceSession._initialize_batch_optimization(
                enable_batch_optimization, gpu_profile
            )
            self.speed_config, self.stream_manager = ONNXInferenceSession._initialize_speed_optimization(
                enable_speed_optimization, gpu_profile
            )
            
        except ImportError as e:
            logger.error(f"Failed to import onnxruntime: {e}")
            raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime-gpu")
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise RuntimeError(f"Failed to create ONNX session: {e}")
    
    def _optimize_input_dtypes(self, name: str, array: np.ndarray) -> np.ndarray:
        """Optimize input array data types."""
        input_info = None
        for inp in self.session.get_inputs():
            if inp.name == name:
                input_info = inp
                break
        
        if input_info:
            expected_dtype = str(input_info.type)
            if 'float16' in expected_dtype and array.dtype != np.float16:
                array = array.astype(np.float16)
                logger.debug(f"Converted {name} to float16 for ONNX model")
            elif 'float32' in expected_dtype and array.dtype != np.float32:
                array = array.astype(np.float32)
                logger.debug(f"Converted {name} to float32 for ONNX model")
        
        return array
    
    def run(self, inputs: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Run inference on the ONNX model with optimized execution.
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            List of output arrays
        """
        try:
            optimized_inputs = {}
            for name, array in inputs.items():
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)
                
                array = self._optimize_input_dtypes(name, array)
                optimized_inputs[name] = array
            
            outputs = self.session.run(self.output_names, optimized_inputs)
            return outputs
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise RuntimeError(f"ONNX inference failed: {e}")
    
    def _optimize_batch_inputs(self, batch_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Optimize batch inputs for inference."""
        optimized_inputs = {}
        for name, array in batch_inputs.items():
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)
            optimized_inputs[name] = array
        return optimized_inputs
    
    def run_batch(self, batch_inputs: Dict[str, np.ndarray], 
                  verbose: bool = False) -> list[np.ndarray]:
        """
        Run batch inference with optimizations.
        
        Supports multiple images in a batch with intelligent synchronization.
        
        Args:
            batch_inputs: Dictionary with batch of inputs
            verbose: Show timing stats
            
        Returns:
            List of outputs
        """
        import time
        import torch
        
        start_time = time.time()
        
        try:
            if verbose:
                logger.info("⚡ Running optimized batch inference...")
            
            optimized_inputs = self._optimize_batch_inputs(batch_inputs)
            outputs = self.session.run(self.output_names, optimized_inputs)
            
            if verbose:
                self._log_batch_stats(start_time, optimized_inputs)
            
            return outputs
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise RuntimeError(f"Batch inference failed: {e}")
    
    def _log_batch_stats(self, start_time: float, optimized_inputs: Dict[str, np.ndarray]) -> None:
        """Log batch inference statistics."""
        import time
        elapsed = time.time() - start_time
        batch_size = optimized_inputs[self.input_names[0]].shape[0]
        logger.info("✅ Batch inference completed:")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Total Time: {elapsed:.3f}s")
        logger.info(f"   Time per item: {elapsed/batch_size*1000:.1f}ms")
    
    def run_batch_async(self, batch_inputs: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Run batch inference asynchronously using streams.
        
        Args:
            batch_inputs: Dictionary with batch of inputs
            
        Returns:
            List of outputs
        """
        import torch
        
        try:
            if not self.stream_manager:
                logger.warning("StreamManager not initialized, using sync execution")
                return self.run_batch(batch_inputs)
            
            logger.info("🌊 Async batch execution with CUDA streams...")
            
            # Get stream and execute
            stream = self.stream_manager.get_stream()
            
            # Optimize inputs
            optimized_inputs = {}
            for name, array in batch_inputs.items():
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)
                optimized_inputs[name] = array
            
            # Execute in stream
            with torch.cuda.stream(stream):
                outputs = self.session.run(self.output_names, optimized_inputs)
            
            # Synchronize
            stream.synchronize()
            
            return outputs
            
        except Exception as e:
            logger.error(f"Async batch inference failed: {e}")
            raise RuntimeError(f"Async batch inference failed: {e}")
    
    def get_batch_config(self) -> Optional[Dict[str, Any]]:
        """Get batch optimization configuration."""
        return self.batch_config
    
    def get_speed_config(self) -> Optional[Dict[str, Any]]:
        """Get speed optimization configuration."""
        return self.speed_config
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on GPU."""
        if self.batch_config:
            return self.batch_config.get("optimal_batch_size", 1)
        return 1
    
    def get_max_batch_size(self) -> int:
        """Get max batch size based on GPU."""
        if self.batch_config:
            return self.batch_config.get("max_batch_size", 1)
        return 1
