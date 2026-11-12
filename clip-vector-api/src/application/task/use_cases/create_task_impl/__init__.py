"""
Create task use case provider module.
"""
from dishka import Provider, provide, Scope
from src.domain import CreateTaskUseCase
from src.domain.task.repositories.task_repository import TaskRepository
from .create_task_usecase_impl import CreateTaskUseCaseImpl


class CreateTaskProvider(Provider):
    """
    Dishka provider for CreateTask use case dependencies.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_create_task_use_case(self, repository: TaskRepository) -> CreateTaskUseCase:
        """Proporciona el caso de uso para crear tareas."""
        return CreateTaskUseCaseImpl(repository)


__all__ = [
    "CreateTaskUseCaseImpl",
    "CreateTaskProvider",
]
