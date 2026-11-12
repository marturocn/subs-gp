"""
TensorRT Inference Session - Infrastructure Layer

Lightweight TensorRT engine runner that deserializes a saved engine and 
provides a `run(inputs)` method similar to ONNXInferenceSession.
"""
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TensorRTInferenceSession:
    """
    Lightweight TensorRT engine runner that deserializes a saved engine and
    provides a `run(inputs)` method similar to ONNXInferenceSession.

    This implementation is best-effort: it requires `tensorrt` and `torch`.
    If those packages are unavailable the constructor will raise a RuntimeError
    with guidance to install TensorRT/PyTorch.
    """

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt

            self.trt = trt
            self.torch = torch
            self.np = np

            logger.info(f"Loading TensorRT engine: {engine_path}")
            
            # Initialize CUDA context via PyTorch if needed
            if torch.cuda.is_available():
                torch.cuda.init()
            
            # Load engine using TRT Runtime
            # Note: We don't need pycuda.autoinit anymore as we rely on PyTorch's context
            with open(engine_path, 'rb') as f:
                self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) # pyright: ignore[reportAttributeAccessIssue]
                engine = self.runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            self.engine = engine
            self.context = engine.create_execution_context()
            
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context (GPU OOM or invalid engine?)")

            # Prepare bindings metadata
            self.input_names: list[str] = []
            self.output_names: list[str] = []
            self.binding_to_dtype: Dict[str, type] = {}
            self.binding_to_idx: Dict[str, int] = {}

            # Handle different TensorRT API versions
            try:
                # TensorRT 8.0+
                num_bindings = self.engine.num_io_tensors
                get_binding_name = self.engine.get_tensor_name
                binding_is_input = lambda idx: self.engine.get_tensor_mode(get_binding_name(idx)) == trt.TensorIOMode.INPUT # pyright: ignore[reportAttributeAccessIssue]
                get_binding_dtype = lambda idx: self.engine.get_tensor_dtype(get_binding_name(idx))
            except AttributeError:
                # TensorRT 7.x fallback
                num_bindings = self.engine.num_bindings
                get_binding_name = self.engine.get_binding_name
                binding_is_input = self.engine.binding_is_input
                get_binding_dtype = self.engine.get_binding_dtype

            # Store helper functions as instance methods
            self._get_binding_name = get_binding_name
            self._binding_is_input = binding_is_input
            self._get_binding_dtype = lambda idx: trt.nptype(get_binding_dtype(idx))
            self._num_bindings = num_bindings

            for idx in range(num_bindings):
                name = get_binding_name(idx)
                self.binding_to_idx[name] = idx
                is_input = binding_is_input(idx)
                dtype = trt.nptype(get_binding_dtype(idx))
                self.binding_to_dtype[name] = dtype
                if is_input:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

            logger.info(f"✅ TensorRT engine loaded (PyTorch backend). Inputs: {self.input_names}, Outputs: {self.output_names}")

        except ImportError as e:
            logger.error(f"TensorRT/PyTorch not available: {e}")
            raise RuntimeError("TensorRT and PyTorch are required to run a serialized engine.")
        except Exception as e:
            logger.error(f"Failed to create TensorRTInferenceSession: {e}")
            raise RuntimeError(f"Failed to create TensorRTInferenceSession: {e}")

    def get_input_dtype(self) -> str:
        """Return a string representation for compatibility with existing code."""
        if not self.input_names:
            return 'float32'
        dt = self.binding_to_dtype.get(self.input_names[0], self.np.float32)
        return 'float16' if dt == self.np.float16 else 'float32'

    def get_batch_config(self) -> Optional[Dict[str, Any]]:
        """
        Get batch size configuration from the TensorRT engine's optimization profile.
        
        Returns:
            Dictionary with 'min_batch_size', 'max_batch_size', and 'optimal_batch_size',
            or None if unable to retrieve.
        """
        try:
            if not self.input_names:
                return None
            
            # Get the first input tensor's name
            input_name = self.input_names[0]
            profile_idx = 0
            
            min_shape = None
            opt_shape = None
            max_shape = None
            
            # Method 1: Modern TensorRT (get_tensor_profile_shape)
            # Signature: get_tensor_profile_shape(self, tensor_name: str, profile_index: int)
            if hasattr(self.engine, "get_tensor_profile_shape"):
                try:
                    shapes = self.engine.get_tensor_profile_shape(input_name, profile_idx)
                    if shapes and len(shapes) == 3:
                        min_shape, opt_shape, max_shape = shapes
                except Exception as e:
                    logger.debug(f"Method 1 (get_tensor_profile_shape) failed: {e}")

            # Method 2: Legacy TensorRT (get_profile_shape) with Binding Index
            # Signature: get_profile_shape(self, profile_index: int, binding_index: int)
            if min_shape is None and hasattr(self.engine, "get_profile_shape"):
                try:
                    if input_name in self.binding_to_idx:
                        binding_idx = self.binding_to_idx[input_name]
                        shapes = self.engine.get_profile_shape(profile_idx, binding_idx)
                        if shapes and len(shapes) == 3:
                            min_shape, opt_shape, max_shape = shapes
                except Exception as e:
                    logger.debug(f"Method 2 (get_profile_shape w/ idx) failed: {e}")
            
            # Method 3: Legacy TensorRT (get_profile_shape) with Name (some python bindings)
            if min_shape is None and hasattr(self.engine, "get_profile_shape"):
                try:
                    # Note: accessing by name is deprecated/removed in newer TRT but might work in older
                    shapes = self.engine.get_profile_shape(profile_idx, input_name)
                    if shapes and len(shapes) == 3:
                        min_shape, opt_shape, max_shape = shapes
                except Exception as e:
                    logger.debug(f"Method 3 (get_profile_shape w/ name) failed: {e}")

            if min_shape is not None and opt_shape is not None and max_shape is not None:
                # Extract batch dimension (first dimension)
                min_batch = min_shape[0]
                opt_batch = opt_shape[0]
                max_batch = max_shape[0]
                
                logger.info(f"✅ TensorRT batch constraints found: min={min_batch}, opt={opt_batch}, max={max_batch}")
                
                return {
                    'min_batch_size': min_batch,
                    'optimal_batch_size': opt_batch,
                    'max_batch_size': max_batch
                }
            else:
                logger.warning("⚠️  Could not retrieve profile shapes using any known TensorRT API method.")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get batch config from TensorRT engine: {e}")
            return None

    def _prepare_inputs(self, inputs: Dict[str, np.ndarray], bindings: list) -> list:
        """Helper to prepare input tensors."""
        input_tensors = []
        
        for name, array in inputs.items():
            if name not in self.binding_to_idx:
                continue
            
            # Convert to torch tensor
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)
                
            tensor = torch.from_numpy(array)
            
            if torch.cuda.is_available():
                tensor = tensor.cuda(non_blocking=True)
            
            idx = self.binding_to_idx[name]
            
            # Set dynamic shape
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except AttributeError:
                self.context.set_binding_shape(idx, tuple(tensor.shape))
            
            bindings[idx] = tensor.data_ptr()
            input_tensors.append(tensor)
            
        return input_tensors

    def _prepare_outputs(self, bindings: list) -> list:
        """Helper to prepare output tensors."""
        output_tensors = []
        
        for name in self.output_names:
            idx = self.binding_to_idx[name]
            
            # Get output shape
            try:
                dims = self.context.get_tensor_shape(name)
            except AttributeError:
                dims = self.context.get_binding_shape(idx)
                
            shape = tuple(dims)
            dtype_np = self.binding_to_dtype[name]
            
            # Map dtype
            if dtype_np == np.float32:
                dtype = torch.float32
            elif dtype_np == np.float16:
                dtype = torch.float16
            elif dtype_np == np.int32:
                dtype = torch.int32
            elif dtype_np == np.int64:
                dtype = torch.int64
            elif dtype_np == bool:
                dtype = torch.bool
            else:
                dtype = torch.float32
            
            tensor = torch.empty(shape, dtype=dtype, device=torch.device("cuda"))
            bindings[idx] = tensor.data_ptr()
            output_tensors.append(tensor)
            
        return output_tensors

    def run(self, inputs: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Run inference using the deserialized TensorRT engine with PyTorch backend.
        
        Args:
            inputs: dict mapping input name to numpy array

        Returns:
            list of numpy arrays corresponding to outputs in engine order
        """
        try:
            bindings = [0] * self._num_bindings
            
            # 1. Prepare Inputs
            # input_tensors kept alive to prevent GC
            input_tensors = self._prepare_inputs(inputs, bindings)
            _ = input_tensors # suppress unused variable warning
            
            # 2. Prepare Outputs
            output_tensors = self._prepare_outputs(bindings)
            
            # 3. Execute Async
            # PyTorch uses its own stream, we hook into it
            stream = torch.cuda.current_stream()
            
            executed = False
            
            # Try execute_async_v2 (Standard for explicit batch)
            if hasattr(self.context, 'execute_async_v2'):
                self.context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
                executed = True
            
            # Try execute_v2 (Synchronous fallback)
            elif hasattr(self.context, 'execute_v2'):
                self.context.execute_v2(bindings=bindings)
                executed = True
                
            # Try execute_async (Legacy implicit batch)
            elif hasattr(self.context, 'execute_async'):
                # Note: execute_async requires batch_size as first arg in legacy mode
                # We inferred batch_size from inputs[0]
                batch_size = inputs[list(inputs.keys())[0]].shape[0] if inputs else 1
                self.context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.cuda_stream)
                executed = True
            
            if not executed:
                logger.error(f"Available context methods: {dir(self.context)}")
                raise RuntimeError("No suitable execution method found (execute_async_v2, execute_v2, execute_async)")
            
            # 4. Synchronize and Copy back
            stream.synchronize()
            
            outputs = [tensor.cpu().numpy() for tensor in output_tensors]
            return outputs

        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}")
            raise RuntimeError(f"TensorRT inference failed: {e}")
