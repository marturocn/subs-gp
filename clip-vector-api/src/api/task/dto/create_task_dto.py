from pydantic import BaseModel, Field, field_validator

# DTO para creación
class CreateTaskDto(BaseModel):
    title: str = Field(min_length=3, max_length=200, description="Título de la tarea")
    description: str = Field(min_length=5, max_length=1000, description="Descripción de la tarea")
    is_completed: bool = Field(default=False, description="Estado de completado de la tarea")

    model_config = {
        "populate_by_name": True,
    }

