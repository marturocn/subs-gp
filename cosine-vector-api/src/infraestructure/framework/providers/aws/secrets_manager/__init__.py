"""
AWS Secrets Manager module.

Provides interfaces and implementations for AWS Secrets Manager integration.
"""
from .secret_manager_interface import ISecretManager
from .secret_manager_local_service import SecretManagerLocalService
from .secrets_manager_provider import SecretsManagerProvider, DISHKA_SECRETS_MANAGER_PROVIDER


__all__ = [
    "ISecretManager",
    "SecretManagerLocalService",
    "SecretsManagerProvider",
    "DISHKA_SECRETS_MANAGER_PROVIDER",
]
