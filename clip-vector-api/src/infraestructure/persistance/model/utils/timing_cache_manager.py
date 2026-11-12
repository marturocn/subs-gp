"""
Timing Cache Manager - Infrastructure Layer

Manages persistent timing cache for TensorRT to speed up engine builds.
Reduces build time from ~5-10 minutes to seconds on repeated builds.
"""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TimingCacheInfo:
    """Metadata about the timing cache."""
    trt_version: str
    gpu_name: str
    compute_capability: str
    created_at: str
    last_used: str
    cache_size_bytes: int


class TimingCacheManager:
    """
    Manages persistent timing cache for TensorRT.
    
    The timing cache stores optimal kernel tactics found during engine building,
    allowing subsequent builds to skip the expensive search process.
    
    This can reduce engine build times from 5-10 minutes to just seconds.
    """
    
    def __init__(self, cache_dir: str = "./trt_cache"):
        """
        Initialize timing cache manager.
        
        Args:
            cache_dir: Directory to store timing cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "timing_cache.bin"
        self.metadata_file = self.cache_dir / "timing_cache_meta.json"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🕐 TimingCacheManager initialized: {self.cache_dir}")
    
    def _get_current_gpu_info(self) -> tuple[str, str]:
        """Get current GPU name and compute capability."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cc = torch.cuda.get_device_capability(0)
                return gpu_name, f"{cc[0]}.{cc[1]}"
        except Exception:
            pass
        return "unknown", "0.0"
    
    def _get_trt_version(self) -> str:
        """Get TensorRT version string."""
        try:
            import tensorrt as trt
            return trt.__version__
        except Exception:
            return "unknown"
    
    def _load_metadata(self) -> Optional[TimingCacheInfo]:
        """Load cache metadata from disk."""
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return TimingCacheInfo(**data)
        except Exception as e:
            logger.warning(f"Failed to load timing cache metadata: {e}")
            return None
    
    def _save_metadata(self, info: TimingCacheInfo) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'trt_version': info.trt_version,
                    'gpu_name': info.gpu_name,
                    'compute_capability': info.compute_capability,
                    'created_at': info.created_at,
                    'last_used': info.last_used,
                    'cache_size_bytes': info.cache_size_bytes,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save timing cache metadata: {e}")
    
    def is_cache_valid(self) -> bool:
        """
        Check if existing cache is valid for current system.
        
        Cache is invalid if:
        - TensorRT version changed
        - GPU changed
        - Cache file is corrupted
        """
        if not self.cache_file.exists():
            logger.info("🕐 No timing cache found, will create new one")
            return False
        
        metadata = self._load_metadata()
        if metadata is None:
            logger.info("🕐 Timing cache metadata missing, invalidating cache")
            return False
        
        # Check TensorRT version
        current_trt = self._get_trt_version()
        if metadata.trt_version != current_trt:
            logger.info(f"🕐 TensorRT version changed ({metadata.trt_version} -> {current_trt}), invalidating cache")
            return False
        
        # Check GPU
        current_gpu, current_cc = self._get_current_gpu_info()
        if metadata.gpu_name != current_gpu or metadata.compute_capability != current_cc:
            logger.info(f"🕐 GPU changed ({metadata.gpu_name} -> {current_gpu}), invalidating cache")
            return False
        
        logger.info(f"✅ Timing cache valid (created: {metadata.created_at}, size: {metadata.cache_size_bytes / 1024:.1f} KB)")
        return True
    
    def load_cache(self) -> bytes:
        """
        Load existing timing cache or return empty bytes for new cache.
        
        Returns:
            Cache data as bytes
        """
        if self.is_cache_valid():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = f.read()
                    logger.info(f"🕐 Loaded timing cache: {len(cache_data)} bytes")
                    
                    # Update last_used
                    metadata = self._load_metadata()
                    if metadata:
                        metadata.last_used = datetime.now().isoformat()
                        self._save_metadata(metadata)
                    
                    return cache_data
            except Exception as e:
                logger.warning(f"Failed to load timing cache: {e}")
        
        # Return empty bytes for new cache
        return b""
    
    def save_cache(self, cache_data: bytes) -> bool:
        """
        Save timing cache to disk.
        
        Args:
            cache_data: Raw cache data from TensorRT
            
        Returns:
            True if save successful
        """
        try:
            # Save cache data
            with open(self.cache_file, 'wb') as f:
                f.write(cache_data)
            
            # Save metadata
            gpu_name, compute_cc = self._get_current_gpu_info()
            now = datetime.now().isoformat()
            
            metadata = TimingCacheInfo(
                trt_version=self._get_trt_version(),
                gpu_name=gpu_name,
                compute_capability=compute_cc,
                created_at=now,
                last_used=now,
                cache_size_bytes=len(cache_data),
            )
            self._save_metadata(metadata)
            
            logger.info(f"✅ Timing cache saved: {len(cache_data)} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save timing cache: {e}")
            return False
    
    def invalidate(self) -> None:
        """Invalidate and remove existing cache."""
        try:
            if self.cache_file.exists():
                os.remove(self.cache_file)
            if self.metadata_file.exists():
                os.remove(self.metadata_file)
            logger.info("🕐 Timing cache invalidated")
        except Exception as e:
            logger.warning(f"Failed to invalidate timing cache: {e}")
    
    def get_cache_for_builder(self, config):
        """
        Get or create timing cache for TensorRT builder config.
        
        Args:
            config: TensorRT builder config
            
        Returns:
            Timing cache object ready to use
        """
        try:
            import tensorrt as trt
            
            existing_cache = self.load_cache()
            
            if existing_cache:
                # Create cache from existing data
                timing_cache = config.create_timing_cache(existing_cache)
                logger.info("🕐 Using existing timing cache for faster build")
            else:
                # Create empty cache
                timing_cache = config.create_timing_cache(b"")
                logger.info("🕐 Creating new timing cache")
            
            return timing_cache
            
        except Exception as e:
            logger.warning(f"Failed to create timing cache: {e}")
            return None
    
    def save_cache_from_builder(self, timing_cache) -> bool:
        """
        Save timing cache from TensorRT builder after build completes.
        
        Args:
            timing_cache: TensorRT timing cache object
            
        Returns:
            True if save successful
        """
        try:
            serialized = timing_cache.serialize()
            
            # TensorRT returns IHostMemory object, need to convert to bytes
            # IHostMemory has a 'bytes()' method or can be accessed via buffer protocol
            if hasattr(serialized, 'tobytes'):
                # For newer TensorRT versions
                cache_data = serialized.tobytes()
            elif hasattr(serialized, '__bytes__'):
                cache_data = bytes(serialized)
            else:
                # Try direct memoryview conversion
                cache_data = bytes(memoryview(serialized))
            
            return self.save_cache(cache_data)
        except Exception as e:
            logger.warning(f"Failed to serialize timing cache: {e}")
            return False

