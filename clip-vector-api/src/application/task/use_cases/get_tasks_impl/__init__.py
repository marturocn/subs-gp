"""
Get tasks use case provider module.
"""
from dishka import Provider, provide, Scope
from src.domain import GetTasksUseCase
from src.domain.task.repositories.task_repository import TaskRepository
from .get_tasks_usecase_impl import GetTasksUseCaseImpl


class GetTasksProvider(Provider):
    """
    Dishka provider for GetTasks use case dependencies.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_get_tasks_use_case(self, repository: TaskRepository) -> GetTasksUseCase:
        """Proporciona el caso de uso para obtener tareas."""
        return GetTasksUseCaseImpl(repository)


__all__ = [
    "GetTasksUseCaseImpl",
    "GetTasksProvider",
]
