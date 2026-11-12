"""
Proveedores globales para FastAPI.
"""
from .global_filters_provider import configure_global_exception_handlers
from .logger.logger import logger
from .aws import AWS_PROVIDERS


# Combined list of all service providers for DI container
DI_SERVICES_PROVIDERS = [
    *AWS_PROVIDERS,
]


__all__ = [
    "configure_global_exception_handlers",
    "logger",
    "AWS_PROVIDERS",
    "DI_SERVICES_PROVIDERS",
]
