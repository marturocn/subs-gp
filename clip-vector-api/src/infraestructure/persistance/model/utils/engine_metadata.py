"""
Engine Metadata - Infrastructure Layer

Stores and validates TensorRT engine metadata to prevent incompatible usage.
"""
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EngineMetadata:
    """
    Metadata for a serialized TensorRT engine.
    
    Stored alongside the engine file to validate compatibility
    before loading, preventing cryptic runtime errors.
    """
    # Engine info
    engine_file: str
    model_name: str
    
    # Build environment
    trt_version: str
    cuda_version: str
    gpu_name: str
    compute_capability: str
    
    # Build configuration
    precision: str  # fp16, fp32, int8
    batch_min: int
    batch_opt: int
    batch_max: int
    
    # Timestamps
    build_date: str
    build_duration_seconds: float
    
    # File info
    engine_size_bytes: int
    onnx_source: Optional[str] = None
    
    def save(self, metadata_path: Optional[str] = None) -> bool:
        """
        Save metadata to JSON file.
        
        Args:
            metadata_path: Path to save metadata (if None, uses engine_file + .meta.json)
            
        Returns:
            True if save successful
        """
        if metadata_path is None:
            metadata_path = f"{self.engine_file}.meta.json"
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"✅ Engine metadata saved: {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save engine metadata: {e}")
            return False
    
    @classmethod
    def load(cls, metadata_path: str) -> Optional["EngineMetadata"]:
        """
        Load metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            EngineMetadata instance or None if load fails
        """
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load engine metadata: {e}")
            return None
    
    @classmethod
    def load_for_engine(cls, engine_path: str) -> Optional["EngineMetadata"]:
        """
        Load metadata for an engine file.
        
        Args:
            engine_path: Path to engine file (.engine or .trt)
            
        Returns:
            EngineMetadata instance or None if not found
        """
        metadata_path = f"{engine_path}.meta.json"
        if not os.path.exists(metadata_path):
            return None
        return cls.load(metadata_path)
    
    def is_compatible(self) -> Tuple[bool, str]:
        """
        Check if this engine is compatible with the current system.
        
        Returns:
            Tuple of (is_compatible, reason_if_not)
        """
        try:
            import tensorrt as trt
            import torch
            
            # Check TensorRT version
            current_trt = trt.__version__
            if self.trt_version != current_trt:
                # TensorRT minor version mismatch might be ok, major = bad
                self_major = self.trt_version.split('.')[0]
                curr_major = current_trt.split('.')[0]
                if self_major != curr_major:
                    return False, f"TensorRT major version mismatch: engine={self.trt_version}, system={current_trt}"
                logger.warning(f"TensorRT minor version mismatch: engine={self.trt_version}, system={current_trt}")
            
            # Check GPU
            if torch.cuda.is_available():
                current_gpu = torch.cuda.get_device_name(0)
                current_cc = torch.cuda.get_device_capability(0)
                current_cc_str = f"{current_cc[0]}.{current_cc[1]}"
                
                # Different GPU is usually fine if compute capability matches
                if self.compute_capability != current_cc_str:
                    return False, f"Compute capability mismatch: engine={self.compute_capability}, system={current_cc_str}"
                
                if self.gpu_name != current_gpu:
                    logger.warning(f"GPU name differs: engine={self.gpu_name}, system={current_gpu}")
            
            # Check engine file exists
            if not os.path.exists(self.engine_file):
                return False, f"Engine file not found: {self.engine_file}"
            
            return True, "Compatible"
            
        except Exception as e:
            return False, f"Compatibility check failed: {e}"
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"TensorRT Engine: {self.model_name}\n"
            f"  File: {self.engine_file}\n"
            f"  Built: {self.build_date}\n"
            f"  TensorRT: {self.trt_version}\n"
            f"  GPU: {self.gpu_name} (CC {self.compute_capability})\n"
            f"  Precision: {self.precision}\n"
            f"  Batch: {self.batch_min}-{self.batch_opt}-{self.batch_max}\n"
            f"  Size: {self.engine_size_bytes / (1024*1024):.1f} MB"
        )


class EngineMetadataBuilder:
    """Builder for creating EngineMetadata during engine build."""
    
    def __init__(self, model_name: str, engine_path: str):
        """
        Initialize metadata builder.
        
        Args:
            model_name: Name of the model
            engine_path: Path where engine will be saved
        """
        self.model_name = model_name
        self.engine_path = engine_path
        self.start_time = datetime.now()
        self.onnx_source: Optional[str] = None
        self.precision = "fp32"
        self.batch_min = 1
        self.batch_opt = 1
        self.batch_max = 1
    
    def set_onnx_source(self, onnx_path: str) -> "EngineMetadataBuilder":
        """Set the ONNX source file."""
        self.onnx_source = onnx_path
        return self
    
    def set_precision(self, precision: str) -> "EngineMetadataBuilder":
        """Set precision (fp16, fp32, int8)."""
        self.precision = precision
        return self
    
    def set_batch_profile(self, batch_min: int, batch_opt: int, batch_max: int) -> "EngineMetadataBuilder":
        """Set batch size profile."""
        self.batch_min = batch_min
        self.batch_opt = batch_opt
        self.batch_max = batch_max
        return self
    
    def build(self) -> EngineMetadata:
        """
        Build the metadata object.
        
        Should be called after engine build completes.
        """
        import torch
        
        # Get system info
        try:
            import tensorrt as trt
            trt_version = trt.__version__
        except Exception:
            trt_version = "unknown"
        
        cuda_version = torch.version.cuda or "unknown"
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            compute_capability = f"{cc[0]}.{cc[1]}"
        else:
            gpu_name = "unknown"
            compute_capability = "0.0"
        
        # Calculate build duration
        build_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Get engine size
        if os.path.exists(self.engine_path):
            engine_size = os.path.getsize(self.engine_path)
        else:
            engine_size = 0
        
        return EngineMetadata(
            engine_file=self.engine_path,
            model_name=self.model_name,
            trt_version=trt_version,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            compute_capability=compute_capability,
            precision=self.precision,
            batch_min=self.batch_min,
            batch_opt=self.batch_opt,
            batch_max=self.batch_max,
            build_date=self.start_time.isoformat(),
            build_duration_seconds=build_duration,
            engine_size_bytes=engine_size,
            onnx_source=self.onnx_source,
        )


def validate_engine_before_load(engine_path: str) -> Tuple[bool, Optional[EngineMetadata], str]:
    """
    Validate a TensorRT engine before loading.
    
    Args:
        engine_path: Path to engine file
        
    Returns:
        Tuple of (is_valid, metadata, reason)
    """
    # Check engine file exists
    if not os.path.exists(engine_path):
        return False, None, f"Engine file not found: {engine_path}"
    
    # Try to load metadata
    metadata = EngineMetadata.load_for_engine(engine_path)
    
    if metadata is None:
        # No metadata file - engine may still work but we can't validate
        logger.warning(f"No metadata found for engine: {engine_path}")
        return True, None, "No metadata available (cannot validate)"
    
    # Check compatibility
    is_compatible, reason = metadata.is_compatible()
    
    if not is_compatible:
        logger.warning(f"Engine incompatible: {reason}")
        return False, metadata, reason
    
    logger.info(f"✅ Engine validated: {metadata.model_name}")
    return True, metadata, "Compatible"
