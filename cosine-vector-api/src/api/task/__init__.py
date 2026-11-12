"""
Task API module.
"""
from http import HTTPMethod
from dishka import Provider, Scope, provide
from src.common.middlewares.basic_auth.basic_auth_middleware import BasicAuthMiddleware
from src.common.core.module_registry import api_module
from .task_controller import TaskController
from src.domain import (
    GetTasksUseCase,
    CreateTaskUseCase,
    UpdateTaskUseCase
)

class TaskControllerProvider(Provider):
    scope = Scope.REQUEST

    @provide(scope=Scope.REQUEST)
    def get_task_controller(
        self,
        get_tasks_use_case: GetTasksUseCase,
        create_task_use_case: CreateTaskUseCase,
        update_task_use_case: UpdateTaskUseCase
    ) -> TaskController: # pyright: ignore[reportGeneralTypeIssues]
        print("🔎 TaskControllerProvider: creando TaskController")
        return TaskController(get_tasks_use_case, create_task_use_case, update_task_use_case)

__all__ = [
    "TaskController",
    "TaskControllerProvider",
]

# @api_module(
#     middlewares={
#         TaskController: [BasicAuthMiddleware]
#         }
# )
# class TaskControllerModule:
#     provider = TaskControllerProvider()
#     controller = TaskController

