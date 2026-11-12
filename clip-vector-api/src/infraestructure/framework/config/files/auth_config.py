"""
Authentication configuration module.
"""
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def auth_config() -> Dict[str, Any]:
    """
    Load authentication configuration from environment variables.
    
    Returns:
        Dict containing authentication configuration
    """
    basic_credentials_env = os.getenv("AUTH_BASIC_CREDENTIALS", "[]")
    
    try:
        basic_credentials = json.loads(basic_credentials_env)
        if not isinstance(basic_credentials, list):
            raise ValueError("AUTH_BASIC_CREDENTIALS must be a JSON array")
    except ValueError as e:
        print(f"Error parsing AUTH_BASIC_CREDENTIALS: {e}")
        basic_credentials = []
    
    return {
        "auth": {
            "basic_credentials": basic_credentials
        }
    }
