"""
Update task use case implementation.
"""
import reactivex # You might need this if you use reactivex.from_ or other reactivex top-level functions
from reactivex import Observable # <--- THIS IS THE CORRECT IMPORT FOR REACTIVEX
from reactivex import operators as op

from src.domain.task.use_cases.update_task_case import UpdateTaskUseCase, UpdateTaskCommand
from src.domain.task.repositories.task_repository import TaskRepository
from src.domain.task.models.entities.task import Task


class UpdateTaskUseCaseImpl(UpdateTaskUseCase):
    """
    Implementation of UpdateTaskUseCase.
    """
    
    def __init__(self, repository: TaskRepository):
        """
        Initialize the use case with a task repository.
        
        Args:
            repository: Task repository for data persistence
        """
        self.repository = repository
    
    def execute(self, command: UpdateTaskCommand) -> Observable[Task]:
        """
        Execute the update task use case.
        
        Args:
            command: Command with task update data
            
        Returns:
            Observable[Task]: Observable that emits the updated task
        """

        task_update = Task(
            id=command.id,
            title=command.title if command.title is not None else "",
            description=command.description if command.description is not None else "",
            is_completed=command.is_completed if command.is_completed is not None else False,
        )
        return self.repository.updateById(command.id, task_update)
