# Import modules to ensure decorators are executed
# This will trigger the @api_module decorators and populate REGISTERED_PROVIDERS
from .task import TaskController  # Import task module (to trigger decorator)
from .vector import VectorController  # Import vector module (to trigger decorator)
from src.common.core.module_registry import REGISTERED_PROVIDERS

API_PROVIDERS = [
    *REGISTERED_PROVIDERS
]

__all__ = [
    "API_PROVIDERS",
    "TaskController",
    "VectorController",
]