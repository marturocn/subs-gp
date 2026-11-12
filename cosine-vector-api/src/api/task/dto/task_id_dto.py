from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from src.common.exceptions.generic_exception_enum import GenericExceptionsEnum, format_exception_message

class TaskIdDto(BaseModel):
    id: int = Field(..., description="ID de la tarea")

    model_config = {
        "populate_by_name": True,
    }
