"""
Use cases module for task application layer.
"""
from dishka import make_async_container
from .create_task_impl import CreateTaskUseCaseImpl, CreateTaskProvider
from .get_tasks_impl import GetTasksUseCaseImpl, GetTasksProvider
from .update_task_impl import UpdateTaskUseCaseImpl, UpdateTaskProvider


# Dishka providers list
DISHKA_TASK_PROVIDERS = [
    CreateTaskProvider(),
    GetTasksProvider(),
    UpdateTaskProvider(),
]

__all__ = [
    "CreateTaskUseCaseImpl",
    "GetTasksUseCaseImpl",
    "UpdateTaskUseCaseImpl",
    "CreateTaskProvider",
    "GetTasksProvider", 
    "UpdateTaskProvider",
    "DISHKA_TASK_PROVIDERS",
]
