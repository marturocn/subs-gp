"""
Database configuration utilities.
"""
import os
from typing import Dict, Any, Optional


def get_env_credentials() -> Optional[Dict[str, Any]]:
    """
    Get database credentials from environment variables.
    Returns None if AWS_USE_SECRETS_MANAGER is enabled.
    """
    if os.getenv("AWS_USE_SECRETS_MANAGER", "false").lower() == "true":
        return None
    
    return {
        "type": os.getenv("DB_TYPE", "mysql"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "name": os.getenv("DB_NAME", "auth_db"),
        "user": os.getenv("DB_USER", "root"),
        "pass": os.getenv("DB_PASS", "root"),
    }
