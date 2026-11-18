"""
AWS Secrets Manager Dishka Provider.

Provides dependency injection for ISecretManager implementations.
"""
import os

from dishka import Provider, provide, Scope

from .secret_manager_interface import ISecretManager
from .secret_manager_local_service import SecretManagerLocalService


class SecretsManagerProvider(Provider):
    """
    Dishka provider for AWS Secrets Manager services.

    Provides SecretManagerLocalService (boto3, host IAM role) when
    AWS_USE_SECRETS_MANAGER=true, otherwise a no-op implementation.
    """

    @provide(scope=Scope.APP)
    def provide_secret_manager(self) -> ISecretManager:
        """
        Provide the appropriate ISecretManager implementation.

        Returns:
            ISecretManager: boto3 service, or a no-op when Secrets Manager is disabled.
        """
        return self.get_secret_manager()

    def get_secret_manager(self) -> ISecretManager:
        """
        Get the appropriate ISecretManager implementation.
        Non-Dishka helper method for direct usage.

        Returns:
            ISecretManager: boto3 service, or a no-op when Secrets Manager is disabled.
        """
        # Check if AWS Secrets Manager is enabled
        use_secrets_manager = os.getenv("AWS_USE_SECRETS_MANAGER", "false").lower()

        if use_secrets_manager != "true":
            # Return a no-op implementation or None-safe wrapper
            return _NoOpSecretManager()

        region = os.getenv("AWS_REGION", "us-east-1")
        return SecretManagerLocalService(region=region)


class _NoOpSecretManager(ISecretManager):
    """
    No-op implementation when Secrets Manager is disabled.
    
    Always returns an error Observable indicating SM is not configured.
    """
    
    def get_secret(self, secret_name: str):
        from reactivex import throw
        return throw(RuntimeError(
            "AWS Secrets Manager is not enabled. "
            "Set AWS_USE_SECRETS_MANAGER=true to enable."
        ))


# Convenience export
DISHKA_SECRETS_MANAGER_PROVIDER = SecretsManagerProvider()
