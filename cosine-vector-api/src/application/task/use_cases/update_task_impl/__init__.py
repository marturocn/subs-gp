"""
Update task use case provider module.
"""
from dishka import Provider, provide, Scope
from src.domain import UpdateTaskUseCase
from src.domain.task.repositories.task_repository import TaskRepository
from .update_task_usecase_impl import UpdateTaskUseCaseImpl


class UpdateTaskProvider(Provider):
    """
    Dishka provider for UpdateTask use case dependencies.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_update_task_use_case(self, repository: TaskRepository) -> UpdateTaskUseCase:
        """Proporciona el caso de uso para actualizar tareas."""
        return UpdateTaskUseCaseImpl(repository)


__all__ = [
    "UpdateTaskUseCaseImpl",
    "UpdateTaskProvider",
]
