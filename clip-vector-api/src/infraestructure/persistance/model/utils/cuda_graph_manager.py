"""
CUDA Graph Manager - Infrastructure Layer

Manages CUDA Graphs for reduced kernel launch overhead.
Provides ~10-30% speedup by capturing and replaying GPU operations.
"""
import logging
from typing import Callable, Optional, Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CUDAGraphManager:
    """
    Manages CUDA Graphs for optimized inference.
    
    CUDA Graphs capture a sequence of GPU operations and replay them with 
    minimal CPU overhead. This is especially beneficial for:
    - Fixed batch size workloads
    - Repetitive inference patterns
    - Low-latency requirements
    
    Provides ~10-30% speedup by eliminating kernel launch overhead.
    """
    
    def __init__(self):
        """Initialize CUDA Graph manager."""
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_exec: Optional[Any] = None
        self.captured = False
        self.warmup_completed = False
        
        # Captured tensors (kept alive for graph replay)
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        
        # Capture settings
        self.capture_batch_size: Optional[int] = None
        self.capture_input_shape: Optional[tuple] = None
        
        if torch.cuda.is_available():
            logger.info("📊 CUDAGraphManager initialized")
        else:
            logger.warning("⚠️ CUDA not available, CUDAGraphManager disabled")
    
    @property
    def is_available(self) -> bool:
        """Check if CUDA graphs are available."""
        return torch.cuda.is_available()
    
    @property
    def is_captured(self) -> bool:
        """Check if a graph has been captured."""
        return self.captured and self.graph is not None
    
    def warmup(self, inference_fn: Callable, sample_input: torch.Tensor, num_warmup: int = 3) -> None:
        """
        Warmup the inference function before capture.
        
        CUDA graphs require the operations to be "warmed up" first to ensure
        consistent memory patterns.
        
        Args:
            inference_fn: Function that takes input tensor and returns output
            sample_input: Sample input tensor with desired shape
            num_warmup: Number of warmup iterations
        """
        if not self.is_available:
            return
        
        logger.info(f"🔥 Warming up for CUDA graph capture ({num_warmup} iterations)...")
        
        with torch.no_grad():
            for i in range(num_warmup):
                _ = inference_fn(sample_input)
                torch.cuda.synchronize()
        
        self.warmup_completed = True
        logger.info("✅ Warmup completed")
    
    def capture(
        self, 
        inference_fn: Callable, 
        sample_input: torch.Tensor,
        auto_warmup: bool = True,
    ) -> bool:
        """
        Capture inference operations into a CUDA graph.
        
        After capture, use replay() for faster execution.
        
        Args:
            inference_fn: Function that takes input tensor and returns output
            sample_input: Sample input tensor with desired shape
            auto_warmup: Automatically run warmup before capture
            
        Returns:
            True if capture successful
        """
        if not self.is_available:
            logger.warning("CUDA not available, cannot capture graph")
            return False
        
        try:
            # Warmup if needed
            if auto_warmup and not self.warmup_completed:
                self.warmup(inference_fn, sample_input)
            
            logger.info(f"📊 Capturing CUDA graph (input shape: {sample_input.shape})...")
            
            # Store capture settings
            self.capture_batch_size = sample_input.shape[0]
            self.capture_input_shape = tuple(sample_input.shape)
            
            # Create static tensors for graph
            self.static_input = sample_input.clone()
            
            # Create the graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Capture the operations
            with torch.no_grad():
                with torch.cuda.graph(self.graph):
                    self.static_output = inference_fn(self.static_input)
            
            self.captured = True
            logger.info(f"✅ CUDA graph captured successfully (batch_size={self.capture_batch_size})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to capture CUDA graph: {e}")
            self.reset()
            return False
    
    def replay(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Replay captured graph with new input data.
        
        The input tensor must have the same shape as the capture input.
        
        Args:
            input_tensor: Input tensor (same shape as capture)
            
        Returns:
            Output tensor, or None if replay fails
        """
        if not self.is_captured:
            logger.warning("No graph captured, cannot replay")
            return None
        
        if input_tensor.shape != self.capture_input_shape:
            logger.warning(
                f"Input shape mismatch: got {input_tensor.shape}, "
                f"expected {self.capture_input_shape}. Falling back to direct execution."
            )
            return None
        
        try:
            # Copy input data to static input
            self.static_input.copy_(input_tensor)
            
            # Replay the graph
            self.graph.replay()
            
            # Return a clone of the output (static_output will be reused)
            return self.static_output.clone()
            
        except Exception as e:
            logger.error(f"CUDA graph replay failed: {e}")
            return None
    
    def reset(self) -> None:
        """Reset the graph manager, releasing captured graph."""
        self.graph = None
        self.graph_exec = None
        self.captured = False
        self.warmup_completed = False
        self.static_input = None
        self.static_output = None
        self.capture_batch_size = None
        self.capture_input_shape = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("📊 CUDA graph manager reset")
    
    def get_capture_info(self) -> Dict[str, Any]:
        """Get information about captured graph."""
        return {
            'is_captured': self.is_captured,
            'batch_size': self.capture_batch_size,
            'input_shape': self.capture_input_shape,
            'warmup_completed': self.warmup_completed,
        }


class CUDAGraphPool:
    """
    Pool of CUDA graphs for different batch sizes.
    
    Maintains multiple captured graphs optimized for different batch sizes,
    selecting the best one at runtime.
    """
    
    def __init__(self, batch_sizes: list[int] = None):
        """
        Initialize graph pool.
        
        Args:
            batch_sizes: List of batch sizes to pre-capture graphs for
        """
        self.graphs: Dict[int, CUDAGraphManager] = {}
        self.batch_sizes = batch_sizes or [1, 4, 8, 16]
        
        logger.info(f"📊 CUDAGraphPool initialized for batch sizes: {self.batch_sizes}")
    
    def capture_all(self, inference_fn: Callable, input_shape_template: tuple) -> int:
        """
        Capture graphs for all configured batch sizes.
        
        Args:
            inference_fn: Inference function
            input_shape_template: Input shape without batch dim, e.g. (3, 224, 224)
            
        Returns:
            Number of graphs successfully captured
        """
        if not torch.cuda.is_available():
            return 0
        
        captured_count = 0
        
        for batch_size in self.batch_sizes:
            try:
                # Create sample input
                input_shape = (batch_size,) + input_shape_template
                sample_input = torch.randn(input_shape, device='cuda')
                
                # Create and capture graph
                graph_manager = CUDAGraphManager()
                if graph_manager.capture(inference_fn, sample_input):
                    self.graphs[batch_size] = graph_manager
                    captured_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to capture graph for batch_size={batch_size}: {e}")
        
        logger.info(f"📊 Captured {captured_count}/{len(self.batch_sizes)} graphs")
        return captured_count
    
    def get_graph(self, batch_size: int) -> Optional[CUDAGraphManager]:
        """Get graph for specific batch size."""
        return self.graphs.get(batch_size)
    
    def replay(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Replay appropriate graph for input tensor.
        
        Falls back to None if no matching graph.
        """
        batch_size = input_tensor.shape[0]
        graph = self.graphs.get(batch_size)
        
        if graph:
            return graph.replay(input_tensor)
        
        return None
    
    def reset_all(self) -> None:
        """Reset all captured graphs."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        logger.info("📊 All CUDA graphs reset")
