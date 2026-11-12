"""
SQLAlchemy implementation of TaskRepository.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Correct imports for the 'reactivex' library
import reactivex # Often useful to import the top-level package if you use reactivex.of, reactivex.throw
from reactivex import Observable, of, throw

from src.domain.task.repositories.task_repository import TaskRepository
from src.domain.task.models.entities.task import Task
from ..models.task_model import TaskModel
import uuid


class TaskImplRepository(TaskRepository):
    """
    SQLAlchemy implementation of TaskRepository.
    Handles task persistence using SQLAlchemy ORM.
    """
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session.
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session
    
    def create(self, task: Task) -> Observable[Task]:
        """
        Creates a new task in the database.
        
        Args:
            task: The task to create
            
        Returns:
            Observable[Task]: Observable that emits the created task
        """
        try:
            # Generate ID if not provided
            if not task.id:
                task.id = str(uuid.uuid4())
            
            # Convert domain entity to model
            task_model = TaskModel.from_domain(task)
            
            # Save to database
            self.session.add(task_model)
            self.session.commit()
            
            # Convert back to domain entity
            created_task = task_model.to_domain()
            
            # Use reactivex.of
            return of(created_task)
            
        except IntegrityError as e:
            self.session.rollback()
            # Use reactivex.throw
            return throw(Exception(f"Task creation failed: {str(e)}"))
        except Exception as e:
            self.session.rollback()
            # Use reactivex.throw
            return throw(e)
    
    def find(self) -> Observable[List[Task]]:
        """
        Retrieves all tasks from the database.
        
        Returns:
            Observable[List[Task]]: Observable that emits the list of tasks
        """
        try:
            # Query all tasks
            task_models = self.session.query(TaskModel).all()
            
            # Convert to domain entities
            tasks = [model.to_domain() for model in task_models]
            
            # Use reactivex.of
            return of(tasks)
            
        except Exception as e:
            # Use reactivex.throw
            return throw(e)

    def updateById(self, id: str, task_update: Optional[Task]) -> Observable[Task]:
        """
        Updates a task by its ID.
        
        Args:
            id: ID of the task to update
            task_update: Dictionary with the fields to update
            
        Returns:
            Observable[Task]: Observable that emits the updated task
        """
        try:
            # Check if task exists
            task_model = self.session.query(TaskModel).filter(TaskModel.id == id).first()
            
            if not task_model:
                # Use reactivex.throw
                return throw(Exception(f"Task with id {id} not found."))
            
            # Update fields
            # NOTE: This `vars(task_update).items()` might not be what you want
            # for Optional[Task]. It would iterate over the attributes of the Task object.
            # You might want to pass a dictionary or a specific DTO for updates.
            if task_update: # Ensure task_update is not None
                for field, value in vars(task_update).items():
                    if hasattr(task_model, field) and field != 'id': # Don't update the ID
                        setattr(task_model, field, value)
            
            # Save changes
            self.session.commit()
            
            # Convert to domain entity
            updated_task = task_model.to_domain()
            # Use reactivex.of
            return of(updated_task)
            
        except Exception as e:
            self.session.rollback()
            # Use reactivex.throw
            return throw(e)