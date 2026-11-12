"""
ONNX Model Optimizer - Infrastructure Layer

Core utilities for exporting PyTorch models to ONNX format with optimizations:
- FP16 precision for faster inference
- TensorRT optimization for NVIDIA GPUs
- OpenVINO optimization for Intel CPUs
- Dynamic GPU-aware optimization for optimal response times

This module has been refactored. Related classes are now in separate files:
- gpu_profiler.py: GPUProfiler
- optimization_strategy.py: OptimizationStrategy
- stream_manager.py: StreamManager
- batch_buffer_pool.py: BatchBufferPool
- onnx_inference_session.py: ONNXInferenceSession
- tensorrt_inference_session.py: TensorRTInferenceSession
- timing_cache_manager.py: TimingCacheManager
- cuda_graph_manager.py: CUDAGraphManager
- engine_metadata.py: EngineMetadata
"""
import logging
import os
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch

from .gpu_profiler import GPUProfiler
from .optimization_strategy import OptimizationStrategy, PRECISION_FP16, PRECISION_FP32
from .timing_cache_manager import TimingCacheManager
from .engine_metadata import EngineMetadataBuilder

logger = logging.getLogger(__name__)

# Logging and string constants
OPTIMIZATION_STRATEGY_MSG = "⚙️  Optimization Strategy:"

# Re-export for backwards compatibility
__all__ = [
    "ONNXOptimizer",
    "GPUProfiler",
    "OptimizationStrategy",
    "PRECISION_FP16",
    "PRECISION_FP32",
    "OPTIMIZATION_STRATEGY_MSG",
]


def _try_import_onnx(optimize: bool):
    """Try to import ONNX module."""
    if not optimize:
        return None

    try:
        import onnx as _onnx
        return _onnx
    except Exception as e:
        logger.warning(
            "ONNX package not usable for optimization/verification (missing optional deps like onnxscript)."
        )
        logger.debug(f"onnx import error: {e}")
        return None


def _optimize_onnx_model(export_path: str, onnx) -> None:
    """Optimize ONNX model with standard passes."""
    try:
        onnx_optimizer = importlib.import_module("onnx.optimizer")
    except Exception:
        onnx_optimizer = getattr(onnx, "optimizer", None)

    logger.info("🔧 Optimizing ONNX model...")
    onnx_model = onnx.load(export_path)

    # Apply standard optimizations
    passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_monotone_argmax',
        'eliminate_nop_pad',
        'eliminate_nop_transpose',
        'eliminate_unused_initializer',
        'extract_constant_to_initializer',
        'fuse_add_bias_into_conv',
        'fuse_bn_into_conv',
        'fuse_consecutive_concats',
        'fuse_consecutive_reduce_unsqueeze',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv',
        'fuse_transpose_into_gemm',
    ]

    try:
        if onnx_optimizer is not None and hasattr(onnx_optimizer, "optimize"):
            optimized_model = onnx_optimizer.optimize(onnx_model, passes)
            onnx.save(optimized_model, export_path)
            logger.info("✅ ONNX model optimized")
        else:
            logger.warning("ONNX optimizer not available; skipping optimization step")
    except Exception as e:
        logger.warning(f"ONNX optimization failed, continuing without optimization: {e}")


def _verify_onnx_model(export_path: str, onnx) -> None:
    """Verify ONNX model integrity."""
    try:
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model verification passed")
    except Exception as e:
        logger.warning(f"ONNX verification failed but export succeeded: {e}")


def _simplify_onnx_model(export_path: str, onnx) -> None:
    """Simplify ONNX model using onnxsim if available."""
    try:
        import onnxsim # pyright: ignore[reportMissingImports]

        logger.info("🔬 Attempting ONNX simplification with onnx-simplifier (onnxsim)...")
        onnx_model_for_sim = onnx.load(export_path)
        simplified_model_bool, simplified_model = onnxsim.simplify(onnx_model_for_sim)
        if simplified_model_bool:
            onnx.save(simplified_model, export_path)
            logger.info("✅ ONNX simplification succeeded and model was rewritten")
        else:
            logger.warning("⚠️ ONNX simplifier returned failure flag; keeping original model")
    except Exception as e:
        logger.debug(f"onnx-simplifier not available or simplification failed: {e}")


class ONNXOptimizer:
    """
    Handles ONNX model export and optimization for improved inference performance.
    
    Supports:
    - FP16 conversion for reduced memory and faster inference
    - TensorRT optimization (NVIDIA GPUs)
    - OpenVINO optimization (Intel CPUs)
    """
    
    @staticmethod
    def export_to_onnx(
        model: torch.nn.Module,
        export_path: str,
        input_shape: Tuple[int, ...],
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14,
        use_fp16: bool = True,
        optimize: bool = True,
        model_type: str = "default",
    ) -> bool:
        """
        Export PyTorch model to ONNX format with GPU-aware dynamic optimizations.
        
        Args:
            model: PyTorch model to export
            export_path: Path where ONNX model will be saved
            input_shape: Shape of input tensor (e.g., (1, 3, 224, 224))
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes configuration for variable batch sizes
            opset_version: ONNX opset version (will be overridden by strategy if provided)
            use_fp16: Convert model to FP16 precision (will be overridden by strategy if provided)
            optimize: Apply ONNX optimizations
            model_type: Type of model for strategy selection ("dinov3", "owlvit", "grounding_dino", "default")
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Profile GPU and get optimization strategy
            gpu_profile = GPUProfiler.profile_gpu()
            strategy = OptimizationStrategy.get_export_strategy(gpu_profile, model_type)
            
            logger.info("=" * 80)
            logger.info("DYNAMIC OPTIMIZATION STRATEGY")
            logger.info("=" * 80)
            logger.info(f"Model Type: {model_type}")
            logger.info("Strategy Summary:")
            logger.info(f"  - Device: {strategy.get('device', 'cpu')}")
            logger.info(f"  - Precision: {strategy.get('precision', PRECISION_FP32)}")
            logger.info(f"  - Batch Size: {strategy.get('batch_size', 1)}")
            logger.info(f"  - Use TensorRT: {strategy.get('use_tensorrt', False)}")
            logger.info(f"  - ONNX OpSet: {strategy.get('onnx_opset', opset_version)}")
            logger.info("=" * 80)
            
            # Override with strategy values
            opset_version = strategy.get("onnx_opset", opset_version)
            use_fp16 = strategy.get("use_fp16", use_fp16) and gpu_profile["available"]
            
            # Create directory if it doesn't exist
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🔄 Exporting model to ONNX: {export_path}")
            logger.info(f"   Input shape: {input_shape}")
            logger.info(f"   FP16: {use_fp16}, Optimize: {optimize}, OpSet: {opset_version}")
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            dummy_input = torch.randn(*input_shape, device=device)
            
            # Match input dtype to model dtype (avoid double conversion if model already FP16)
            if model_dtype == torch.float16:
                logger.info("   Model already in FP16, matching input dtype...")
                dummy_input = dummy_input.half()
            elif use_fp16 and device.type == "cuda":
                logger.info("   Converting model to FP16...")
                model = model.half()
                dummy_input = dummy_input.half()
            
            # Default names if not provided
            if input_names is None:
                input_names = ["input"]
            if output_names is None:
                output_names = ["output"]
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_input,),
                    export_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                )
            
            logger.info("✅ Model exported to ONNX successfully")
            
            # Optimize ONNX model (best-effort)
            onnx = _try_import_onnx(optimize)

            if optimize and onnx is not None:
                _optimize_onnx_model(export_path, onnx)

            # Verify the exported model (only if onnx is available)
            if onnx is not None:
                _verify_onnx_model(export_path, onnx)
                _simplify_onnx_model(export_path, onnx)
            
            # Get model size
            size_mb = Path(export_path).stat().st_size / (1024 * 1024)
            logger.info(f"📦 ONNX model size: {size_mb:.2f} MB")
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import ONNX dependencies: {e}")
            logger.error("Install with: pip install onnx onnxruntime")
            return False
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            return False
    
    @staticmethod
    def _configure_tensorrt_precision(config, strategy: Dict[str, Any], builder) -> None:
        """Configure TensorRT precision settings."""
        import tensorrt as trt
        
        precision = strategy.get("tensorrt_precision", PRECISION_FP16)
        if precision == PRECISION_FP16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16) # pyright: ignore[reportAttributeAccessIssue]
            logger.info("📊 FP16 precision: ✅ Enabled")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8) # pyright: ignore[reportAttributeAccessIssue]
            logger.info("📊 INT8 precision: ✅ Enabled")
        else:
            logger.info("📊 Precision: FP32 (default)")
    
    @staticmethod
    def _setup_optimization_profile(profile, network, strategy: Dict[str, Any]) -> None:
        """Setup optimization profile with batch dimensions."""
        batch_min = strategy.get("tensorrt_batch_min", 1)
        batch_opt = strategy.get("tensorrt_batch_opt", 8)
        batch_max = strategy.get("tensorrt_batch_max", 16)
        
        if network.num_inputs != 1:
            logger.warning(f"⚠️  Model has {network.num_inputs} inputs, expected 1")
        
        input_layer = network.get_input(0)
        input_name = input_layer.name
        input_shape_list = list(input_layer.shape)
        
        logger.info(f"📐 Input shape: {input_shape_list}")
        logger.info("📊 Batch optimization:")
        logger.info(f"  - Min: {batch_min}")
        logger.info(f"  - Opt: {batch_opt}")
        logger.info(f"  - Max: {batch_max}")
        
        if len(input_shape_list) == 4:
            _, c, h, w = input_shape_list
            profile.set_shape(input_name, (batch_min, c, h, w), (batch_opt, c, h, w), (batch_max, c, h, w))
        elif len(input_shape_list) == 3:
            _, seq_len, feat_dim = input_shape_list
            profile.set_shape(input_name, (batch_min, seq_len, feat_dim), (batch_opt, seq_len, feat_dim), (batch_max, seq_len, feat_dim))
        else:
            logger.warning(f"⚠️  Unexpected input shape: {input_shape_list}")
    
    @staticmethod
    def _configure_tactic_sources(config, gpu_profile: Dict[str, Any]) -> None:
        """Configure TensorRT tactic sources based on GPU capabilities."""
        import tensorrt as trt
        
        if gpu_profile["tensor_cores"]:
            config.set_tactic_sources(
                (1 << int(trt.TacticSource.CUBLAS)) | # pyright: ignore[reportAttributeAccessIssue]
                (1 << int(trt.TacticSource.CUBLAS_LT)) | # pyright: ignore[reportAttributeAccessIssue]
                (1 << int(trt.TacticSource.CUDNN)) # pyright: ignore[reportAttributeAccessIssue]
            )
            logger.info("🎯 Tactic sources: CUBLAS, CUBLAS_LT, CUDNN")
        else:
            config.set_tactic_sources(1 << int(trt.TacticSource.CUDNN)) # pyright: ignore[reportAttributeAccessIssue]
            logger.info("🎯 Tactic sources: CUDNN")
    
    @staticmethod
    def optimize_with_tensorrt_dynamic(
        onnx_path: str,
        output_path: Optional[str] = None,
        model_type: str = "default",
        use_timing_cache: bool = True,
        save_metadata: bool = True,
    ) -> bool:
        """
        Optimize ONNX model with TensorRT using GPU-aware dynamic settings.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine (if None, saves as .engine next to onnx)
            model_type: Type of model for strategy selection ("dinov3", "owlvit", "grounding_dino")
            use_timing_cache: Use persistent timing cache for faster rebuilds
            save_metadata: Save engine metadata for validation
            
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            import tensorrt as trt
            
            # Resolve absolute path to help TensorRT locate external data files
            onnx_abs_path = os.path.abspath(onnx_path)
            onnx_dir = os.path.dirname(onnx_abs_path)
            onnx_filename = os.path.basename(onnx_abs_path)
            
            gpu_profile = GPUProfiler.profile_gpu()
            
            if not gpu_profile["available"]:
                logger.warning("❌ TensorRT optimization requires CUDA GPU")
                return False
            
            strategy = OptimizationStrategy.get_export_strategy(gpu_profile, model_type)
            
            if not strategy.get("use_tensorrt", False):
                logger.warning(f"⚠️  TensorRT optimization not recommended for {model_type}")
                return False
            
            if output_path is None:
                output_path = onnx_abs_path.replace('.onnx', '.engine')
            else:
                output_path = os.path.abspath(output_path)
            
            ONNXOptimizer._log_tensorrt_config(gpu_profile, strategy, model_type)
            
            # Initialize timing cache manager
            timing_cache_manager = TimingCacheManager() if use_timing_cache else None
            
            # Initialize metadata builder
            metadata_builder = None
            if save_metadata:
                metadata_builder = EngineMetadataBuilder(model_type, output_path)
                metadata_builder.set_onnx_source(onnx_abs_path)
                metadata_builder.set_precision(strategy.get("tensorrt_precision", "fp32"))
                metadata_builder.set_batch_profile(
                    strategy.get("tensorrt_batch_min", 1),
                    strategy.get("tensorrt_batch_opt", 8),
                    strategy.get("tensorrt_batch_max", 16),
                )
            
            # Change CWD to ONNX directory so TRT can find external .data file
            original_cwd = os.getcwd()
            try:
                os.chdir(onnx_dir)
                logger.debug(f"Changed CWD to {onnx_dir} for TensorRT parsing")
                
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # pyright: ignore[reportAttributeAccessIssue]
                builder = trt.Builder(TRT_LOGGER) # pyright: ignore[reportAttributeAccessIssue]
                network = builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # pyright: ignore[reportAttributeAccessIssue]
                )
                parser = trt.OnnxParser(network, TRT_LOGGER) # pyright: ignore[reportAttributeAccessIssue]
                
                logger.info("🔍 Parsing ONNX model...")
                # Parse using filename since we are in the directory
                if not ONNXOptimizer._parse_onnx_model(parser, onnx_filename):
                    return False
            finally:
                os.chdir(original_cwd)
                
            config = builder.create_builder_config()
            ONNXOptimizer._configure_builder(config, strategy, builder)
            
            # Apply timing cache if available
            timing_cache = None
            if timing_cache_manager:
                timing_cache = timing_cache_manager.get_cache_for_builder(config)
                if timing_cache:
                    config.set_timing_cache(timing_cache, ignore_mismatch=True)
            
            profile = builder.create_optimization_profile()
            ONNXOptimizer._setup_optimization_profile(profile, network, strategy)
            config.add_optimization_profile(profile)
            
            ONNXOptimizer._configure_tactic_sources(config, gpu_profile)
            
            logger.info("🔨 Building TensorRT engine (this may take several minutes)...")
            engine = builder.build_serialized_network(network, config)
            
            if engine is None:
                logger.error("❌ Failed to build TensorRT engine")
                return False
            
            # Save timing cache for future builds
            if timing_cache_manager and timing_cache:
                timing_cache_manager.save_cache_from_builder(timing_cache)
            
            success = ONNXOptimizer._save_trt_engine(engine, output_path)
            
            # Save metadata
            if success and metadata_builder:
                metadata = metadata_builder.build()
                metadata.save()
            
            return success
            
        except ImportError as e:
            logger.error(f"❌ TensorRT not available: {e}")
            logger.error("Install TensorRT from: https://developer.nvidia.com/tensorrt")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to optimize with TensorRT: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _log_tensorrt_config(gpu_profile: Dict[str, Any], strategy: Dict[str, Any], model_type: str) -> None:
        """Log TensorRT configuration."""
        logger.info("=" * 80)
        logger.info("TENSORRT DYNAMIC OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Model: {model_type}")
        logger.info(f"GPU: {gpu_profile['device']}")
        logger.info(f"VRAM: {gpu_profile['vram_gb']:.1f} GB")
        logger.info(f"Compute Capability: {gpu_profile['compute_capability_str']}")
        logger.info(f"Tensor Cores: {'✅' if gpu_profile['tensor_cores'] else '❌'}")
        logger.info("")
        logger.info("TensorRT Settings:")
        logger.info(f"  - Precision: {strategy.get('tensorrt_precision', PRECISION_FP16)}")
        logger.info(f"  - Workspace: {strategy.get('tensorrt_workspace_mb', 2048)} MB")
        logger.info(f"  - Min Batch: {strategy.get('tensorrt_batch_min', 1)}")
        logger.info(f"  - Opt Batch: {strategy.get('tensorrt_batch_opt', 8)}")
        logger.info(f"  - Max Batch: {strategy.get('tensorrt_batch_max', 16)}")
        logger.info("=" * 80)
    
    @staticmethod
    def _parse_onnx_model(parser, onnx_path: str) -> bool:
        """Parse ONNX model and return success status."""
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.error("Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    logger.error(f"  {parser.get_error(error)}")
                return False
        logger.info("✅ ONNX model parsed successfully")
        return True
    
    @staticmethod
    def _configure_builder(config, strategy: Dict[str, Any], builder) -> None:
        """Configure TensorRT builder with dynamic settings."""
        import tensorrt as trt
        
        workspace_mb = strategy.get("tensorrt_workspace_mb", 2048)
        workspace_bytes = workspace_mb * (1024 ** 2)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes) # pyright: ignore[reportAttributeAccessIssue]
        logger.info(f"💾 Workspace: {workspace_mb} MB")
        
        config.builder_optimization_level = 4
        logger.info("⚙️  Builder optimization level: 4 (latency optimized)")
        
        ONNXOptimizer._configure_tensorrt_precision(config, strategy, builder)
    
    @staticmethod
    def _save_trt_engine(engine, output_path: str) -> bool:
        """Save TensorRT engine to file."""
        logger.info(f"💾 Saving engine to: {output_path}")
        with open(output_path, 'wb') as f:
            f.write(engine)
        
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info("✅ TensorRT engine saved successfully")
        logger.info(f"📦 Engine size: {size_mb:.2f} MB")
        logger.info("=" * 80)
        return True
    
    @staticmethod
    def optimize_with_tensorrt(
        onnx_path: str,
        output_path: Optional[str] = None,
        fp16_mode: bool = True,
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,  # 1GB
    ) -> bool:
        """
        Optimize ONNX model with TensorRT for NVIDIA GPUs (legacy method).
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine (if None, saves as .trt next to onnx)
            fp16_mode: Enable FP16 precision
            max_batch_size: Maximum batch size
            workspace_size: Maximum workspace size in bytes
            
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            import tensorrt as trt
            
            if output_path is None:
                output_path = onnx_path.replace('.onnx', '.trt')
            
            logger.info(f"🚀 Optimizing with TensorRT: {onnx_path}")
            logger.info(f"   FP16: {fp16_mode}, Max batch: {max_batch_size}")
            
            # Create TensorRT builder and network
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # pyright: ignore[reportAttributeAccessIssue]
            builder = trt.Builder(TRT_LOGGER) # pyright: ignore[reportAttributeAccessIssue]
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # pyright: ignore[reportAttributeAccessIssue]
            )
            parser = trt.OnnxParser(network, TRT_LOGGER) # pyright: ignore[reportAttributeAccessIssue]
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX file")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size) # pyright: ignore[reportAttributeAccessIssue]
            
            # Enable FP16 if supported and requested
            if fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16) # pyright: ignore[reportAttributeAccessIssue]
                logger.info("   ✅ FP16 mode enabled")
            else:
                logger.warning("   ⚠️  FP16 not available or not requested")
            
            # Build engine
            logger.info("   Building TensorRT engine (this may take a few minutes)...")
            engine = builder.build_serialized_network(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine)
            
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"✅ TensorRT engine saved: {output_path} ({size_mb:.2f} MB)")
            
            return True
            
        except ImportError as e:
            logger.error(f"TensorRT not available: {e}")
            logger.error("Install TensorRT from: https://developer.nvidia.com/tensorrt")
            return False
        except Exception as e:
            logger.error(f"Failed to optimize with TensorRT: {e}")
            return False
    
    @staticmethod
    def optimize_with_openvino(
        onnx_path: str,
        output_dir: Optional[str] = None,
        fp16_mode: bool = True,
    ) -> bool:
        """
        Optimize ONNX model with OpenVINO for Intel CPUs.
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Directory to save OpenVINO model (if None, creates folder next to onnx)
            fp16_mode: Enable FP16 precision
            
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            from openvino.runtime import Core
            from openvino.tools import mo # pyright: ignore[reportAttributeAccessIssue]
            
            if output_dir is None:
                output_dir = str(Path(onnx_path).parent / "openvino")
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🚀 Optimizing with OpenVINO: {onnx_path}")
            logger.info(f"   Output dir: {output_dir}")
            logger.info(f"   FP16: {fp16_mode}")
            
            # Convert ONNX to OpenVINO IR
            model_name = Path(onnx_path).stem
            output_model = str(Path(output_dir) / model_name)
            
            # Model Optimizer arguments
            mo_args = {
                'input_model': onnx_path,
                'output_model': output_model,
                'compress_to_fp16': fp16_mode,
            }
            
            logger.info("   Converting to OpenVINO IR format...")
            ov_model = mo.convert_model(**mo_args)
            
            # Save the model
            from openvino.runtime import serialize
            serialize(ov_model, f"{output_model}.xml")
            
            logger.info(f"✅ OpenVINO model saved: {output_model}.xml")
            
            # Get model size
            xml_size = Path(f"{output_model}.xml").stat().st_size / (1024 * 1024)
            bin_size = Path(f"{output_model}.bin").stat().st_size / (1024 * 1024)
            total_size = xml_size + bin_size
            
            logger.info(f"📦 OpenVINO model size: {total_size:.2f} MB (xml: {xml_size:.2f} MB, bin: {bin_size:.2f} MB)")
            
            return True
            
        except ImportError as e:
            logger.error(f"OpenVINO not available: {e}")
            logger.error("Install with: pip install openvino openvino-dev")
            return False
        except Exception as e:
            logger.error(f"Failed to optimize with OpenVINO: {e}")
            return False
    
    @staticmethod
    def get_onnx_providers(provider: str = "auto") -> list[str]:
        """
        Get available ONNX Runtime execution providers.
        
        Args:
            provider: Provider preference ('auto', 'tensorrt', 'cuda', 'cpu', 'openvino')
            
        Returns:
            List of execution providers in priority order
        """
        try:
            import onnxruntime as ort
            
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            if provider == "auto":
                # Auto-select best available provider
                if "TensorrtExecutionProvider" in available_providers:
                    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                elif "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                elif "OpenVINOExecutionProvider" in available_providers:
                    providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
            
            elif provider == "tensorrt":
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            
            elif provider == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            
            elif provider == "openvino":
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            
            else:  # cpu or unknown
                providers = ["CPUExecutionProvider"]
            
            # Filter to only available providers
            providers = [p for p in providers if p in available_providers]
            
            if not providers:
                logger.warning("No requested providers available, falling back to CPU")
                providers = ["CPUExecutionProvider"]
            
            logger.info(f"Using ONNX providers: {providers}")
            return providers
            
        except ImportError:
            logger.error("onnxruntime not installed")
            return ["CPUExecutionProvider"]
        except Exception as e:
            logger.error(f"Error getting ONNX providers: {e}")
            return ["CPUExecutionProvider"]
