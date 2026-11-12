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
            "vector_dimension": int(os.getenv("VECTOR_DIMENSION", "1024")),
        }
    }
