"""
Stream Manager - Infrastructure Layer

Manages CUDA Streams for asynchronous and parallel execution.
Significantly improves throughput in batch processing.
"""
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class StreamManager:
    """
    Manages CUDA Streams for asynchronous and parallel execution.
    Significantly improves throughput in batch processing.
    """
    
    def __init__(self, num_streams: int = 2):
        """
        Initialize stream manager.
        
        Args:
            num_streams: Number of streams to create
        """
        self.num_streams = num_streams
        self.streams: list[torch.cuda.Stream] = []
        self.current_stream_idx = 0
        
        if torch.cuda.is_available():
            for _ in range(num_streams):
                stream = torch.cuda.Stream()
                self.streams.append(stream)
            
            logger.info(f"🌊 StreamManager initialized: {num_streams} CUDA Streams")
        else:
            logger.warning("⚠️  CUDA not available, StreamManager disabled")
    
    def get_stream(self) -> torch.cuda.Stream:
        """Get next stream (round-robin)."""
        if not self.streams:
            return torch.cuda.default_stream()
        
        stream = self.streams[self.current_stream_idx % len(self.streams)]
        self.current_stream_idx += 1
        return stream
    
    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        if torch.cuda.is_available():
            for stream in self.streams:
                stream.synchronize()
    
    def synchronize_stream(self, idx: int) -> None:
        """Synchronize specific stream."""
        if torch.cuda.is_available() and 0 <= idx < len(self.streams):
            self.streams[idx].synchronize()
    
    def get_current_stream_index(self) -> int:
        """Get current stream index."""
        return self.current_stream_idx % len(self.streams) if self.streams else 0
    
    def reset(self) -> None:
        """Reset stream index to start."""
        self.current_stream_idx = 0
    
    @property
    def is_available(self) -> bool:
        """Check if stream manager has active streams."""
        return len(self.streams) > 0
