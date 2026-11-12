"""
AWS Secrets Manager Dishka Provider.

Provides dependency injection for ISecretManager implementations.
"""
import os

from dishka import Provider, provide, Scope

from .secret_manager_interface import ISecretManager
from .secret_manager_local_service import SecretManagerLocalService
from .secret_manager_layer_service import SecretManagerLayerService


class SecretsManagerProvider(Provider):
    """
    Dishka provider for AWS Secrets Manager services.
    
    Automatically detects the environment and provides the appropriate
    ISecretManager implementation:
    - Lambda environment: SecretManagerLayerService (HTTP extension)
    - Other environments: SecretManagerLocalService (boto3)
    """
    
    @provide(scope=Scope.APP)
    def provide_secret_manager(self) -> ISecretManager:
        """
        Provide the appropriate ISecretManager implementation.
        
        Returns:
            ISecretManager: Lambda layer service if in Lambda, otherwise boto3 service.
        """
        return self.get_secret_manager()
    
    def get_secret_manager(self) -> ISecretManager:
        """
        Get the appropriate ISecretManager implementation.
        Non-Dishka helper method for direct usage.
        
        Returns:
            ISecretManager: Lambda layer service if in Lambda, otherwise boto3 service.
        """
        # Check if AWS Secrets Manager is enabled
        use_secrets_manager = os.getenv("AWS_USE_SECRETS_MANAGER", "false").lower()
        
        if use_secrets_manager != "true":
            # Return a no-op implementation or None-safe wrapper
            return _NoOpSecretManager()
        
        # Detect Lambda environment
        if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            return SecretManagerLayerService()
        
        # Default to boto3 for local/EC2/ECS
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

