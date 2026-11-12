"""
Task SQLAlchemy model for database persistence.
"""
from sqlalchemy import Boolean, Column, String
from sqlalchemy.ext.declarative import declarative_base
from src.domain.task.models.entities.task import Task
import uuid

# Crear una instancia local de Base por ahora
Base = declarative_base()


class TaskModel(Base):
    """
    SQLAlchemy model for Task entity.
    Maps Task domain entity to database table.
    """
    
    __tablename__ = 'tasks'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    description = Column(String(500), nullable=False)
    is_completed = Column(Boolean, default=False, nullable=False)
    
    def to_domain(self) -> Task:
        """
        Converts database model to domain entity.
        
        Returns:
            Task: Domain task entity
        """
        return Task(
            id=str(self.id),
            title=str(self.title),
            description=str(self.description),
            is_completed=bool(self.is_completed)
        )
    
    @classmethod
    def from_domain(cls, task: Task) -> 'TaskModel':
        """
        Creates database model from domain entity.
        
        Args:
            task: Domain task entity
        
        Returns:
            TaskModel: Database task model
        """
        return cls(
            id=task.id or str(uuid.uuid4()),
            title=task.title,
            description=task.description,
            is_completed=task.is_completed
        )
