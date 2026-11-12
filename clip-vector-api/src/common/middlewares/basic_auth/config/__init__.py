"""
Basic Auth configuration module.
"""
from src.infraestructure.framework.config.config_root_module import ConfigRootModule, get_auth_config
from src.infraestructure.framework.config.services.auth_config_service import AuthConfigService


def get_basic_auth_config() -> AuthConfigService:
    """
    Get the basic authentication configuration service.
    
    Returns:
        AuthConfigService instance with basic auth configuration
    """
    return get_auth_config()
