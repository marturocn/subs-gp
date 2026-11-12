"""
AWS Secrets Manager Lambda Layer service implementation.
Uses HTTP endpoint from AWS Parameters and Secrets Lambda Extension.
"""
import os
from typing import Any, Dict
import httpx
from reactivex import Observable, create


from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_interface import ISecretManager


class SecretManagerLayerService(ISecretManager):
    """
    Lambda Layer implementation of Secrets Manager.
    Uses the AWS Parameters and Secrets Lambda Extension HTTP endpoint.
    """
    
    def __init__(self, endpoint: str = "http://localhost:2773", session_token: str = ""):
        """
        Initialize the Lambda Layer service.
        
        Args:
            endpoint: Base endpoint for the Lambda extension (default: http://localhost:2773)
            session_token: AWS session token for the extension
        """
        self.endpoint = endpoint
        self.session_token = session_token or os.getenv("AWS_SESSION_TOKEN", "")
    
    def get_secret(self, secret_id: str) -> Observable[str]:
        """
        Get a secret value from AWS Secrets Manager via Lambda Extension.
        
        Args:
            secret_id: The ID or ARN of the secret
            
        Returns:
            Observable[str]: An observable that emits the secret string value
        """
        def subscribe(observer, scheduler=None):
            try:
                url = f"{self.endpoint}/secretsmanager/get?secretId={secret_id}"
                headers: Dict[str, str] = {}
                
                if self.session_token:
                    headers["X-Aws-Parameters-Secrets-Token"] = self.session_token
                
                with httpx.Client() as client:
                    response = client.get(url, headers=headers, timeout=10.0)
                    response.raise_for_status()
                    
                    data = response.json()
                    secret_string = data.get("SecretString", "")
                    
                    observer.on_next(secret_string)
                    observer.on_completed()
                    
            except httpx.HTTPError as e:
                observer.on_error(e)
            except Exception as e:
                observer.on_error(e)
        
        return create(subscribe)
