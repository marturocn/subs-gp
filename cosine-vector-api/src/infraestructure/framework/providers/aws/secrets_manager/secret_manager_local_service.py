"""
AWS Secrets Manager Local Service.

Boto3 implementation for local development and EC2/ECS environments.
"""
import json
import os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from reactivex import Observable, of, throw

from .secret_manager_interface import ISecretManager


def _region_from_arn(secret_id: str) -> Optional[str]:
    """Return the region of a Secrets Manager ARN, or None for a plain secret name."""
    if not secret_id.startswith("arn:"):
        return None
    parts = secret_id.split(":")
    return parts[3] if len(parts) > 3 and parts[3] else None


class SecretManagerLocalService(ISecretManager):
    """
    AWS Secrets Manager implementation using boto3.

    Credentials come from boto3's default chain (the host IAM role on EC2/ECS).
    """

    def __init__(self, region: str | None = None):
        """
        Initialize the boto3 Secrets Manager client.

        Args:
            region: AWS region name. Defaults to AWS_REGION env var or us-east-1.
                An ARN secret id is always fetched from the ARN's own region.
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self._clients: Dict[str, Any] = {}

    def _client_for(self, secret_id: str):
        """Lazy boto3 client for the secret's region (ARN region, else self.region)."""
        region = _region_from_arn(secret_id) or self.region
        if region not in self._clients:
            self._clients[region] = boto3.client(
                'secretsmanager',
                region_name=region
            )
        return self._clients[region]

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
                response = self._client_for(secret_name).get_secret_value(SecretId=secret_name)
                
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
