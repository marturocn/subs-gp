"""
Application module
"""
from dishka import make_async_container
from src.api import API_PROVIDERS
from src.application.vector.use_cases import DISHKA_VECTOR_PROVIDERS
from src.infraestructure.persistance.persistance_module import PERSISTENCE_PROVIDERS
from src.application.task.use_cases import DISHKA_TASK_PROVIDERS
from .task import (
    CreateTaskUseCaseImpl,
    GetTasksUseCaseImpl,
    UpdateTaskUseCaseImpl,
    CreateTaskProvider,
    GetTasksProvider,
    UpdateTaskProvider,
)
from src.infraestructure.framework.providers import DI_SERVICES_PROVIDERS

APPLICATION_PROVIDERS = [
    *DISHKA_TASK_PROVIDERS,
    *DISHKA_VECTOR_PROVIDERS,  # New vector application provider
    *PERSISTENCE_PROVIDERS,
    *DI_SERVICES_PROVIDERS,
    *API_PROVIDERS
]

# Instancias de tus providers existentes
application_container = make_async_container(
    *APPLICATION_PROVIDERS
)

__all__ = [
    "CreateTaskUseCaseImpl",
    "GetTasksUseCaseImpl",
    "UpdateTaskUseCaseImpl",
    "CreateTaskProvider",
    "GetTasksProvider", 
    "UpdateTaskProvider",
    "application_container",
]