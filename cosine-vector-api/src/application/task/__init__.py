"""
Task application module.
"""
from .use_cases import (
    CreateTaskUseCaseImpl,
    GetTasksUseCaseImpl,
    UpdateTaskUseCaseImpl,
    CreateTaskProvider,
    GetTasksProvider,
    UpdateTaskProvider
)

from .use_cases.create_task_impl.dto.requests import CreateTaskRequestDTO
from .use_cases.create_task_impl.dto.responses import TaskResponseDTO, TaskListResponseDTO
from .use_cases.get_tasks_impl.dto.requests import GetTasksRequestDTO
from .use_cases.update_task_impl.dto.requests import UpdateTaskRequestDTO

__all__ = [
    # Use Case Implementations
    "CreateTaskUseCaseImpl",
    "GetTasksUseCaseImpl",
    "UpdateTaskUseCaseImpl",
    # Dishka Providers
    "CreateTaskProvider",
    "GetTasksProvider",
    "UpdateTaskProvider",
    # DTOs for API
    "CreateTaskRequestDTO",
    "GetTasksRequestDTO",
    "UpdateTaskRequestDTO",
    "TaskResponseDTO",
    "TaskListResponseDTO",
]
