"""
AWS Secrets Manager interface.
"""
from abc import ABC, abstractmethod
from reactivex import Observable


class ISecretManager(ABC):
    """Abstract interface for AWS Secrets Manager."""
    
    @abstractmethod
    def get_secret(self, secret_id: str) -> Observable[str]:
        """
        Get a secret value from AWS Secrets Manager.
        
        Args:
            secret_id: The ID or ARN of the secret
            
        Returns:
            Observable[str]: An observable that emits the secret string value
        """
        pass
