"""
AWS providers module.

Provides AWS service integrations.
"""
from .secrets_manager import (
    ISecretManager,
    SecretManagerLocalService,
    SecretManagerLayerService,
    SecretsManagerProvider,
    DISHKA_SECRETS_MANAGER_PROVIDER,
)


# All AWS-related Dishka providers
AWS_PROVIDERS = [
    DISHKA_SECRETS_MANAGER_PROVIDER,
]


__all__ = [
    "ISecretManager",
    "SecretManagerLocalService", 
    "SecretManagerLayerService",
    "SecretsManagerProvider",
    "DISHKA_SECRETS_MANAGER_PROVIDER",
    "AWS_PROVIDERS",
]
