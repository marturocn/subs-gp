
import torch
import sys
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_checkpoint")

def verify_checkpoint():
    checkpoint_path = "models/grounding_dino_optimized.pt"
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
        
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # Check keys
        keys = list(checkpoint.keys())
        logger.info(f"Checkpoint keys count: {len(keys)}")
        logger.info(f"First 10 keys: {keys[:10]}")
        
        # Check for _orig_mod prefix
        orig_mod_keys = [k for k in keys if "_orig_mod." in k]
        
        if orig_mod_keys:
            logger.error(f"❌ Found {len(orig_mod_keys)} keys with '_orig_mod.' prefix!")
            logger.error(f"Sample: {orig_mod_keys[:5]}")
            return False
            
        logger.info("✅ No '_orig_mod.' prefixes found. Checkpoint structure looks correct.")
        
        # Check precision
        is_fp16 = any(v.dtype == torch.float16 for v in checkpoint.values() if torch.is_tensor(v) and v.is_floating_point())
        logger.info(f"Precision check: {'FP16 (Correct)' if is_fp16 else 'FP32'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify checkpoint: {e}")
        return False

if __name__ == "__main__":
    success = verify_checkpoint()
    sys.exit(0 if success else 1)
