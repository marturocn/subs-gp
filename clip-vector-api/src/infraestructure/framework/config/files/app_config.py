"""
Application configuration module.
"""
import os
from typing import Dict, Any


def app_config() -> Dict[str, Any]:
    """
    Load application configuration from environment variables.
    
    Returns:
        Dict containing application configuration
    """
    return {
        "app": {
            "env": os.getenv("LOCAL"),
            "stage": os.getenv("STAGE"),
            "log_level": os.getenv("LOG_LEVEL"),
            "custom_domain": os.getenv("CUSTOM_DOMAIN"),
            "API_PORT": os.getenv("API_PORT"),
        },
        "faiss": {
            "shared_memory_name": os.getenv("SHARED_MEMORY_NAME"),
            "shared_memory_size": os.getenv("SHARED_MEMORY_SIZE"),
        }
    }
