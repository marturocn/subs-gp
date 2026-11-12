"""
DTO de respuesta para tareas.
"""
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.domain.task.models.entities.task import Task


class TaskResponseDto(BaseModel):
    """
    DTO para respuestas de tareas.
    """
    id: Optional[str] = Field(description="ID único de la tarea")
    title: Optional[str] = Field(description="Título de la tarea")
    description: str = Field(description="Descripción de la tarea")
    is_completed: bool = Field(description="Estado de completado de la tarea")

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @classmethod
    def from_domain(cls, task: "Task") -> "TaskResponseDto":
        """
        Convierte una entidad del dominio a DTO de respuesta.
        """
        return cls(
            id=task.id,
            title=task.title,
            description=task.description,
            is_completed=task.is_completed
        )
