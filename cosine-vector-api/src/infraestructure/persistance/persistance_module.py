from infraestructure.persistance import BASE_ENTITY_PROVIDERS
from src.infraestructure.db import DISHKA_DB_PROVIDER
from .task.task_persistence_module import TASK_ENTITY_PROVIDERS

PERSISTENCE_PROVIDERS = [
    DISHKA_DB_PROVIDER,  # Database connections and entities
    *TASK_ENTITY_PROVIDERS,
    *BASE_ENTITY_PROVIDERS,
]

__all__ = [
    "PERSISTENCE_PROVIDERS",
]
