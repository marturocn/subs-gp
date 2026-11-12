"""
Task persistence module.
Provides dependency injection configuration for task persistence layer.
"""
from sqlalchemy.orm import Session
from dishka import Provider, provide, Scope
from src.domain.task.repositories.task_repository import TaskRepository
from src.infraestructure.db.db_connection import get_db_session
from .repositories.task_impl_repository import TaskImplRepository




class TaskPersistenceProvider(Provider):
    """
    Dishka provider for task persistence layer dependencies.
    Uses the centralized database provider for session management.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_task_repository(self, session: Session) -> TaskRepository:
        """
        Proporciona el repositorio de tareas.
        Recibe la sesión directamente del DISHKA_DB_PROVIDER.
        """
        return TaskImplRepository(session)


TASK_ENTITY_PROVIDERS = [
    TaskPersistenceProvider(),  # Entities
]

