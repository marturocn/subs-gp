"""
Use case for updating tasks.
"""
from abc import ABC, abstractmethod
from reactivex import Observable

from src.domain.task.models.entities.task import Task
from src.common.core import UseCase
from .commands import UpdateTaskCommand


class UpdateTaskUseCase(UseCase[UpdateTaskCommand, Task], ABC):
    """
    Abstract use case for updating an existing task.
    """
    
    @abstractmethod
    def execute(self, command: UpdateTaskCommand) -> Observable[Task]:
        """
        Executes the use case to update a task.
        
        Args:
            command: Command with the data of the task to update
        
        Returns:
            Observable[Task]: Observable that emits the updated task
        """
        pass
