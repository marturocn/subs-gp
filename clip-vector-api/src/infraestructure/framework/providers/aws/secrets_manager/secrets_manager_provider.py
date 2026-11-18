"""
Dishka provider for AWS Secrets Manager.
"""
import os
from dishka import Provider, provide, Scope

from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_interface import ISecretManager
from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_local_service import SecretManagerLocalService


class SecretsManagerProvider(Provider):
    """
    Provider for AWS Secrets Manager dependencies.
    Uses boto3 with the default credential chain (host IAM role).
    """

    @provide(scope=Scope.APP)
    def get_secret_manager(self) -> ISecretManager:
        """
        Provide Secrets Manager implementation.

        Returns:
            ISecretManager: boto3-based implementation
        """
        return SecretManagerLocalService(region=os.getenv("AWS_REGION"))
