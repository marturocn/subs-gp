"""
Dishka provider for AWS Secrets Manager.
"""
import os
from dishka import Provider, provide, Scope

from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_interface import ISecretManager
from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_local_service import SecretManagerLocalService
from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_layer_service import SecretManagerLayerService


class SecretsManagerProvider(Provider):
    """
    Provider for AWS Secrets Manager dependencies.
    Automatically selects the appropriate implementation based on environment.
    """
    
    @provide(scope=Scope.APP)
    def get_secret_manager(self) -> ISecretManager:
        """
        Provide Secrets Manager implementation.
        
        Returns:
            ISecretManager: Local implementation for dev, Layer implementation for Lambda
        """
        # Check if running in AWS Lambda environment
        if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            # Lambda environment - use Layer service
            endpoint = os.getenv("AWS_SM_ENDPOINT", "http://localhost:2773")
            session_token = os.getenv("AWS_SESSION_TOKEN", "")
            return SecretManagerLayerService(endpoint=endpoint, session_token=session_token)
        else:
            # Local/dev environment - use direct boto3
            region = os.getenv("AWS_REGION")
            access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            return SecretManagerLocalService(
                region=region,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key
            )
