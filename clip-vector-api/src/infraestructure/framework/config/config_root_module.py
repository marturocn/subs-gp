
from functools import lru_cache
from .files.config_registry import load_all_configs
from .services.app_config_service import AppConfigService
from .services.auth_config_service import AuthConfigService
from typing import Dict, Any


class ConfigRootModule:
    """
    Root configuration module that manages all configuration services.
    """

    def __init__(self: 'ConfigRootModule') -> None:
        """Initialize the configuration root module."""
        self._config: Dict[str, Any] = load_all_configs()
        self._app_config_service: AppConfigService | None = None
        self._auth_config_service: AuthConfigService | None = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the loaded configuration."""
        return self._config
    
    def get_app_config_service(self) -> AppConfigService:
        """
        Get the application configuration service instance.
        
        Returns:
            AppConfigService instance
        """
        if self._app_config_service is None:
            self._app_config_service = AppConfigService(self._config)
        return self._app_config_service
    
    def get_auth_config_service(self) -> AuthConfigService:
        """
        Get the authentication configuration service instance.
        
        Returns:
            AuthConfigService instance
        """
        if self._auth_config_service is None:
            self._auth_config_service = AuthConfigService(self._config)
        return self._auth_config_service


# Global instance (singleton pattern)
@lru_cache(maxsize=1)
def get_config_root_module() -> ConfigRootModule:
    """
    Get the singleton instance of ConfigRootModule.
    
    Returns:
        ConfigRootModule instance
    """
    return ConfigRootModule()


# Convenience functions for direct access to services
def get_app_config() -> AppConfigService:
    """
    Get the application configuration service.
    
    Returns:
        AppConfigService instance
    """
    return get_config_root_module().get_app_config_service()


def get_auth_config() -> AuthConfigService:
    """
    Get the authentication configuration service.
    
    Returns:
        AuthConfigService instance
    """
    return get_config_root_module().get_auth_config_service()
