"""
Use case for retrieving all tasks.
"""
from abc import ABC, abstractmethod
from reactivex import Observable
from src.common.core import UseCase
from .responses import GetTasksResponse


class GetTasksUseCase(UseCase[None, GetTasksResponse], ABC):
    """
    Abstract use case for retrieving all tasks.
    """
    
    @abstractmethod
    def execute(self, command: None) -> Observable[GetTasksResponse]:
        """
        Executes the use case to retrieve all tasks.
        
        Args:
            command: No command required (None)
        
        Returns:
            Observable[GetTasksResponse]: Observable that emits the list of tasks
        """
        pass
