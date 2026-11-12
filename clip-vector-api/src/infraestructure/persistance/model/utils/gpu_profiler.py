"""
GPU Profiler - Infrastructure Layer

Profiles GPU capabilities for dynamic optimization decisions.
"""
import logging
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)


def _classify_gpu_family(compute_capability: int) -> str:
    """Classify GPU family by compute capability."""
    if compute_capability >= 9:
        return "Hopper"  # RTX 6000 Ada, L40S, etc.
    elif compute_capability == 8:
        return "Ampere"  # RTX 3090, A100, etc.
    elif compute_capability == 7:
        return "Volta"   # V100, RTX Titan, etc.
    else:
        return "Legacy"  # Older GPUs


def _estimate_cuda_cores(cc_major: int, _: int, gpu_name: str) -> int:
    """Estimate CUDA cores based on compute capability and GPU name."""
    # Common GPU models and their CUDA core counts
    gpu_cores_map = {
        "RTX 4090": 16384,
        "RTX 3090": 10496,
        "RTX 3080": 8704,
        "RTX 3070": 5888,
        "A100": 6912,
        "A30": 3456,
        "V100": 5120,
        "L40S": 18176,
    }
    
    for model, cores in gpu_cores_map.items():
        if model in gpu_name:
            return cores
    
    # Fallback estimates by compute capability
    fallback = {
        9: 14080,  # Hopper
        8: 8192,   # Ampere
        7: 5120,   # Volta
        6: 2560,   # Pascal
    }
    
    return fallback.get(cc_major, 1024)


class GPUProfiler:
    """Profiles GPU capabilities for dynamic optimization."""
    
    @staticmethod
    def profile_gpu() -> Dict[str, Any]:
        """
        Profile GPU capabilities and compute optimal settings.
        
        Returns:
            Dictionary with GPU profile information
        """
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device": "cpu",
                "compute_capability": None,
                "vram_gb": 0,
                "tensor_cores": False,
                "max_threads_per_block": 0,
                "cuda_cores": 0,
            }
        
        device_idx = 0
        compute_cap = torch.cuda.get_device_capability(device_idx)
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        vram_gb = total_memory / (1024 ** 3)
        
        # Determine GPU family and capabilities
        gpu_name = torch.cuda.get_device_name(device_idx)
        cc_major, cc_minor = compute_cap
        
        # GPU families and their characteristics
        has_tensor_cores = cc_major >= 7  # Volta and newer
        has_rt_cores = cc_major >= 8     # Ampere and newer
        
        # Estimate CUDA cores based on compute capability
        cuda_cores = _estimate_cuda_cores(cc_major, cc_minor, gpu_name)
        
        profile = {
            "available": True,
            "device": torch.cuda.get_device_name(device_idx),
            "device_idx": device_idx,
            "compute_capability": compute_cap,
            "compute_capability_str": f"{cc_major}.{cc_minor}",
            "vram_gb": vram_gb,
            "vram_bytes": total_memory,
            "tensor_cores": has_tensor_cores,
            "rt_cores": has_rt_cores,
            "cuda_cores": cuda_cores,
            "gpu_family": _classify_gpu_family(cc_major),
        }
        
        logger.info("🧠 GPU Profile:")
        logger.info(f"   Device: {profile['device']}")
        logger.info(f"   Compute Capability: {profile['compute_capability_str']}")
        logger.info(f"   VRAM: {vram_gb:.1f} GB")
        logger.info(f"   CUDA Cores: {cuda_cores}")
        logger.info(f"   Tensor Cores: {'✅' if has_tensor_cores else '❌'}")
        logger.info(f"   RT Cores: {'✅' if has_rt_cores else '❌'}")
        
        return profile
