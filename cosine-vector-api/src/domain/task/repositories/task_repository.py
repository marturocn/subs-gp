"""
Abstract repository for tasks.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from reactivex import Observable # <--- THIS IS THE CORRECT IMPORT FOR REACTIVEX

from src.domain.task.models.entities.task import Task


class TaskRepository(ABC):
    """
    Abstract repository for task management.
    Defines basic CRUD operations for tasks.
    """
    
    @abstractmethod
    def create(self, task: Task) -> Observable[Task]:
        """
        Creates a new task.
        
        Args:
            task: The task to create
        
        Returns:
            Observable[Task]: Observable that emits the created task
        """
        pass
    
    @abstractmethod
    def find(self) -> Observable[List[Task]]:
        """
        Retrieves all tasks.
        
        Returns:
            Observable[List[Task]]: Observable that emits the list of tasks
        """
        pass
    
    @abstractmethod
    def updateById(self, id: str, task_update: Optional[Task]) -> Observable[Task]:
        """
        Updates a task by its ID.
        
        Args:
            id: ID of the task to update
            task_update: Dictionary with the fields to update
        
        Returns:
            Observable[Task]: Observable that emits the updated task
        """
        pass
