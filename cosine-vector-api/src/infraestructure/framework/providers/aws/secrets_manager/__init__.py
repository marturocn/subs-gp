"""
AWS Secrets Manager module.

Provides interfaces and implementations for AWS Secrets Manager integration.
"""
from .secret_manager_interface import ISecretManager
from .secret_manager_local_service import SecretManagerLocalService
from .secret_manager_layer_service import SecretManagerLayerService
from .secrets_manager_provider import SecretsManagerProvider, DISHKA_SECRETS_MANAGER_PROVIDER


__all__ = [
    "ISecretManager",
    "SecretManagerLocalService",
    "SecretManagerLayerService",
    "SecretsManagerProvider",
    "DISHKA_SECRETS_MANAGER_PROVIDER",
]
