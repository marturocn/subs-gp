"""
Batch Buffer Pool - Infrastructure Layer

Pre-allocated buffer pool for efficient batch processing.
Avoids repeated GPU memory allocations.
"""
import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BatchBufferPool:
    """
    Pool of pre-allocated buffers for efficient batch processing.
    Avoids repeated GPU memory allocations.
    """
    
    def __init__(self, 
                 batch_size: int, 
                 input_shape: Tuple[int, ...],
                 dtype: type = np.float32,
                 num_buffers: int = 3):
        """
        Initialize buffer pool.
        
        Args:
            batch_size: Batch size
            input_shape: Input shape (without batch dim)
            dtype: Data type
            num_buffers: Number of buffers to pre-allocate
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.dtype = dtype
        self.buffer_shape = (batch_size,) + input_shape
        
        # Pre-allocate GPU buffers if CUDA is available
        self.gpu_buffers: list[torch.Tensor] = []
        self.cpu_buffers: list[torch.Tensor | np.ndarray] = []
        self.buffer_index = 0
        
        if torch.cuda.is_available():
            for _ in range(num_buffers):
                # GPU buffer
                gpu_buffer = torch.empty(self.buffer_shape, dtype=torch.float32, device='cuda')
                self.gpu_buffers.append(gpu_buffer)
                
                # CPU pinned buffer (faster for transfers)
                cpu_buffer = torch.empty(self.buffer_shape, dtype=torch.float32, 
                                        pin_memory=True)
                self.cpu_buffers.append(cpu_buffer)
        else:
            for _ in range(num_buffers):
                cpu_buffer = np.empty(self.buffer_shape, dtype=dtype)
                self.cpu_buffers.append(cpu_buffer)
        
        memory_per_buffer = np.prod(self.buffer_shape) * np.dtype(dtype).itemsize / (1024**2)
        
        logger.info("🧠 BufferPool initialized:")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Shape: {self.buffer_shape}")
        logger.info(f"   Buffers: {num_buffers}")
        logger.info(f"   Memory per buffer: {memory_per_buffer:.1f} MB")
    
    def get_buffer(self) -> torch.Tensor | np.ndarray:
        """Get next buffer from pool (round-robin)."""
        if self.gpu_buffers:
            buffer = self.gpu_buffers[self.buffer_index % len(self.gpu_buffers)]
        else:
            buffer = self.cpu_buffers[self.buffer_index % len(self.cpu_buffers)]
        
        self.buffer_index += 1
        return buffer
    
    def get_pinned_buffer(self) -> torch.Tensor | np.ndarray:
        """Get pinned CPU buffer (optimal for GPU transfers)."""
        buffer = self.cpu_buffers[self.buffer_index % len(self.cpu_buffers)]
        self.buffer_index += 1
        return buffer
    
    def reset(self) -> None:
        """Reset buffer index to start."""
        self.buffer_index = 0
    
    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU buffers are available."""
        return len(self.gpu_buffers) > 0
    
    @property
    def total_memory_mb(self) -> float:
        """Get total memory used by all buffers in MB."""
        num_total = len(self.gpu_buffers) + len(self.cpu_buffers)
        memory_per = np.prod(self.buffer_shape) * np.dtype(self.dtype).itemsize
        return (num_total * memory_per) / (1024 ** 2)
