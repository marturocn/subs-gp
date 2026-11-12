"""
Authentication configuration service.
"""
from .base_config_service import BaseConfigService
from typing import Dict, Any, List


class AuthConfigService(BaseConfigService):
    """
    Service for accessing authentication configuration values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the authentication config service.
        
        Args:
            config: The configuration dictionary
        """
        super().__init__(config)
    
    @property
    def prefix(self) -> str:
        """Get the configuration prefix for authentication settings."""
        return "auth"
    
    @property
    def BASIC_CREDENTIALS(self) -> List[Dict[str, str]]:
        """Get the basic authentication credentials."""
        return list(self.get("basic_credentials"))
    
    def get_credential_by_username(self, username: str) -> Dict[str, str] | None:
        """
        Get credentials by username.
        
        Args:
            username: The username to search for
            
        Returns:
            Dict containing username and password, or None if not found
        """
        for credential in self.BASIC_CREDENTIALS:
            if credential.get("username") == username:
                return credential
        return None
    
    def validate_credentials(self, username: str, password: str) -> bool:
        """
        Validate if the provided username and password are correct.
        
        Args:
            username: The username to validate
            password: The password to validate
            
        Returns:
            True if credentials are valid, False otherwise
        """
        credential = self.get_credential_by_username(username)
        if credential is None:
            return False
        return credential.get("password") == password
