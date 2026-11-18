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


def _region_from_arn(secret_id: str) -> Optional[str]:
    """Return the region of a Secrets Manager ARN, or None for a plain secret name."""
    if not secret_id.startswith("arn:"):
        return None
    parts = secret_id.split(":")
    return parts[3] if len(parts) > 3 and parts[3] else None


class SecretManagerLocalService(ISecretManager):
    """
    Implementation of Secrets Manager using boto3 SDK.
    Credentials come from boto3's default chain (the host IAM role on EC2/ECS).
    """

    def __init__(self, region: Optional[str] = None):
        """
        Initialize the Secrets Manager client.

        Args:
            region: AWS region (defaults to AWS_REGION env var).
                An ARN secret id is always fetched from the ARN's own region.
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self._clients: Dict[str, Any] = {}

    def _client_for(self, secret_id: str):
        """Lazy boto3 client for the secret's region (ARN region, else self.region)."""
        region = _region_from_arn(secret_id) or self.region
        if region not in self._clients:
            self._clients[region] = boto3.client("secretsmanager", region_name=region)
        return self._clients[region]

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
                response = self._client_for(secret_id).get_secret_value(SecretId=secret_id)
                secret_string = response.get("SecretString", "")
                observer.on_next(secret_string)
                observer.on_completed()
            except ClientError as e:
                observer.on_error(e)
            except Exception as e:
                observer.on_error(e)

        return create(subscribe)
