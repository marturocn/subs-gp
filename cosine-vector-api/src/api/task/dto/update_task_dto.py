from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from src.common.exceptions.generic_exception_enum import GenericExceptionsEnum, format_exception_message

# DTO para actualización (ya estaba bien, pero lo unificamos aquí para tener todo completo)
class UpdateTaskDto(BaseModel):
    title: Optional[str] = Field(default=None, min_length=3)
    description: Optional[str] = Field(default=None, min_length=5)
    is_completed: Optional[bool] = Field(default=None)

    model_config = {
        "populate_by_name": True,
    }
