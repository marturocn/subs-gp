"""
AWS Secrets Manager Local Service.

Boto3 implementation for local development and EC2/ECS environments.
"""
import json
import os
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError
from reactivex import Observable, of, throw

from .secret_manager_interface import ISecretManager


class SecretManagerLocalService(ISecretManager):
    """
    AWS Secrets Manager implementation using boto3.
    
    Best suited for:
    - Local development with AWS credentials
    - EC2/ECS instances with IAM roles
    """
    
    def __init__(self, region: str | None = None):
        """
        Initialize the boto3 Secrets Manager client.
        
        Args:
            region: AWS region name. Defaults to AWS_REGION env var or us-east-1.
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of boto3 client."""
        if self._client is None:
            self._client = boto3.client(
                'secretsmanager',
                region_name=self.region
            )
        return self._client
    
    def get_secret(self, secret_name: str) -> Observable[str]:
        """
        Fetch a secret from AWS Secrets Manager using boto3.
        
        Args:
            secret_name: The name or ARN of the secret.
            
        Returns:
            Observable emitting the secret string value.
        """
        def _fetch_secret() -> Observable[str]:
            try:
                response = self.client.get_secret_value(SecretId=secret_name)
                
                if 'SecretString' in response:
                    return of(response['SecretString'])
                else:
                    # Binary secret - decode to string
                    import base64
                    decoded = base64.b64decode(response['SecretBinary']).decode('utf-8')
                    return of(decoded)
                    
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ResourceNotFoundException':
                    return throw(ValueError(f"Secret '{secret_name}' not found"))
                elif error_code == 'AccessDeniedException':
                    return throw(PermissionError(f"Access denied to secret '{secret_name}'"))
                else:
                    return throw(e)
            except Exception as e:
                return throw(e)
        
        return _fetch_secret()
    
    def get_secret_as_dict(self, secret_name: str) -> Observable[Dict[str, Any]]:
        """
        Fetch and parse a JSON secret.
        
        Args:
            secret_name: The name or ARN of the secret.
            
        Returns:
            Observable emitting the parsed JSON as a dictionary.
        """
        return self.get_secret(secret_name).pipe(
            lambda obs: obs.pipe(
                lambda o: o.pipe(
                    lambda x: of(json.loads(x.value)) if hasattr(x, 'value') else of(json.loads(x))
                )
            )
        )
