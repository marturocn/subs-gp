"""
AWS Secrets Manager Interface.

Defines the abstract base class for secret management services.
"""
from abc import ABC, abstractmethod
from reactivex import Observable


class ISecretManager(ABC):
    """
    Abstract interface for AWS Secrets Manager implementations.
    
    Supports both local boto3 and Lambda extension implementations.
    """
    
    @abstractmethod
    def get_secret(self, secret_name: str) -> Observable[str]:
        """
        Fetch a secret value from AWS Secrets Manager.
        
        Args:
            secret_name: The name or ARN of the secret to retrieve.
            
        Returns:
            Observable that emits the secret value as a string.
        """
        pass
