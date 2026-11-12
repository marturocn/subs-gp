"""
Application configuration service.
"""
from .base_config_service import BaseConfigService
from typing import Dict, Any


class AppConfigService(BaseConfigService):
    """
    Service for accessing application configuration values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the app config service.
        
        Args:
            config: The configuration dictionary
        """
        super().__init__(config)
    
    @property
    def prefix(self) -> str:
        """Get the configuration prefix for app settings."""
        return "app"
    
    @property
    def env(self) -> str:
        """Get the environment setting."""
        return str(self.get("env"))
    
    @property
    def is_dev(self) -> bool:
        """Check if the environment is development."""
        return self.env == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if the environment is production."""
        return self.env == "production"
    
    @property
    def node_env(self) -> str:
        """Get the Node.js environment setting."""
        return str(self.get("node"))
    
    @property
    def stage(self) -> str:
        """Get the stage setting."""
        return str(self.get("stage"))
    
    @property
    def log_level(self) -> str:
        """Get the log level setting."""
        return str(self.get("log_level"))
    
    @property
    def api_port(self) -> int:
        """Get the API port setting."""
        return int(self.get("API_PORT"))

    @property
    def is_custom_domain_enabled(self) -> bool:
        """Check if custom domain is enabled."""
        custom_domain = str(self.get_optional("custom_domain", "")).upper()
        return custom_domain == "TRUE"

    @property
    def faiss_vector_dimension(self) -> int:
        """Get the FAISS vector dimension."""
        faiss_config = self.config.get("faiss", {})
        return int(faiss_config.get("vector_dimension", 1024))
