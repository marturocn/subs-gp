"""
AWS Secrets Manager local service implementation.
Uses boto3 to directly connect to AWS Secrets Manager.
"""
import os
from typing import Any, Dict, Optional
import boto3
from botocore.exceptions import ClientError
from reactivex import Observable, create, of


from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_interface import ISecretManager


class SecretManagerLocalService(ISecretManager):
    """
    Local implementation of Secrets Manager using boto3 SDK.
    Used for development environments with AWS credentials.
    """
    
    def __init__(
        self,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None
    ):
        """
        Initialize the Secrets Manager client.
        
        Args:
            region: AWS region (defaults to AWS_REGION env var)
            access_key_id: AWS access key (defaults to AWS_ACCESS_KEY_ID env var)
            secret_access_key: AWS secret key (defaults to AWS_SECRET_ACCESS_KEY env var)
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        # Build credentials config
        credentials: Dict[str, Any] = {}
        if access_key_id and secret_access_key:
            credentials["aws_access_key_id"] = access_key_id
            credentials["aws_secret_access_key"] = secret_access_key
        elif os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            credentials["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
            credentials["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        self.client = boto3.client(
            "secretsmanager",
            region_name=self.region,
            **credentials
        )
    
    def get_secret(self, secret_id: str) -> Observable[str]:
        """
        Get a secret value from AWS Secrets Manager.
        
        Args:
            secret_id: The ID or ARN of the secret
            
        Returns:
            Observable[str]: An observable that emits the secret string value
        """
        def subscribe(observer, scheduler=None):
            try:
                response = self.client.get_secret_value(SecretId=secret_id)
                secret_string = response.get("SecretString", "")
                observer.on_next(secret_string)
                observer.on_completed()
            except ClientError as e:
                observer.on_error(e)
            except Exception as e:
                observer.on_error(e)
        
        return create(subscribe)
