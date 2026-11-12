"""
Configuration registry for loading all configuration modules.
"""
from .app_config import app_config
from .auth_config import auth_config
from typing import List, Callable, Dict, Any


# Registry of configuration loaders
# Note: Database config is now handled by DbGetConfigService with Secrets Manager support
CONFIG_LOADERS: List[Callable[[], Dict[str, Any]]] = [
    app_config,
    auth_config,
]


def load_all_configs() -> Dict[str, Any]:
    """
    Load all configurations and merge them into a single dictionary.
    
    Returns:
        Dict containing all merged configurations
    """
    merged_config: Dict[str, Any] = {}
    for loader in CONFIG_LOADERS:
        config = loader()
        merged_config.update(config)
    return merged_config

