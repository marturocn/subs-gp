"""
Infraestructure module
"""
# Expose global logger (Imported first to prevent circular imports in DB configs)
from .framework.providers.logger.logger import logger

from .db import db_connection, DISHKA_DB_PROVIDER
from .framework.config import config_root_module
from .framework.config.files import app_config, config_registry
from .framework.config.services import app_config_service, base_config_service
from .framework.filters import server_exception_filter, unknown_exception_filter
from .framework.providers import global_filters_provider
from .persistance import persistance_module
from .persistance.task import task_persistence_module
from .persistance.task.models import task_model
from .persistance.task.repositories import task_impl_repository

__all__ = [
    "db_connection",
    "DISHKA_DB_PROVIDER",
    "config_root_module",
    "app_config",
    "config_registry",
    "app_config_service",
    "base_config_service",
    "server_exception_filter",
    "unknown_exception_filter",
    "global_filters_provider",
    "persistance_module",
    "task_persistence_module",
    "task_model",
    "task_impl_repository",
    "logger"
]