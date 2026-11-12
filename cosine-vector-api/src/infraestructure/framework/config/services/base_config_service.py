"""
Base configuration service for common configuration functionality.
"""
from abc import ABC
from typing import Any, Dict


class BaseConfigService(ABC):
    """
    Abstract base class for configuration services.
    Provides common functionality for accessing configuration values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base config service.
        
        Args:
            config: The configuration dictionary
        """
        self.config = config
    
    @property
    def prefix(self) -> str:
        """
        Get the configuration prefix for this service.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must define the prefix property")
    
    def get(self, key: str) -> Any:
        """
        Get a configuration value by key, prefixed with the service prefix.
        
        Args:
            key: The configuration key (without prefix)
            
        Returns:
            The configuration value
            
        Raises:
            KeyError: If the configuration key is not found
        """
        full_key = f"{self.prefix}.{key}"
        prefix_config = self.config.get(self.prefix, {})
        
        if key not in prefix_config:
            raise KeyError(f"Configuration key '{full_key}' not found")
            
        return prefix_config[key]
    
    def get_optional(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key, returning a default if not found.
        
        Args:
            key: The configuration key (without prefix)
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default
        """
        try:
            return self.get(key)
        except KeyError:
            return default
