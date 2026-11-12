"""
Comando para crear una nueva tarea.
"""
from dataclasses import dataclass


@dataclass
class CreateTaskCommand:
    """
    Comando para crear una nueva tarea.
    """
    title: str
    description: str
    is_completed: bool
