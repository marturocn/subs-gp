"""
AWS Secrets Manager module.
"""
from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_interface import ISecretManager
from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_local_service import SecretManagerLocalService
from src.infraestructure.framework.providers.aws.secrets_manager.secret_manager_layer_service import SecretManagerLayerService
from src.infraestructure.framework.providers.aws.secrets_manager.secrets_manager_provider import SecretsManagerProvider

# Dishka providers list
DISHKA_AWS_PROVIDERS = [
    SecretsManagerProvider(),
]

__all__ = [
    "ISecretManager",
    "SecretManagerLocalService",
    "SecretManagerLayerService",
    "SecretsManagerProvider",
    "DISHKA_AWS_PROVIDERS",
]
