"""
Caso de uso para crear tareas.
"""
from abc import ABC, abstractmethod
from reactivex import Observable
from src.domain.task.models import Task
from src.common.core import UseCase
from .commands import CreateTaskCommand


class CreateTaskUseCase(UseCase[CreateTaskCommand, Task], ABC):
    """
    Caso de uso abstracto para crear una nueva tarea.
    """
    
    @abstractmethod
    def execute(self, command: CreateTaskCommand) -> Observable[Task]:
        """
        Ejecuta el caso de uso para crear una nueva tarea.
        
        Args:
            command: Comando con los datos de la tarea a crear
            
        Returns:
            Observable[Task]: Observable que emite la tarea creada
        """
        pass
