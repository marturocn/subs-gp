"""
DTO for piece ID validation in vector operations.
"""
from pydantic import BaseModel, Field


class PieceIdDto(BaseModel):
    """
    DTO for validating piece ID parameter.
    """
    id: str = Field(..., description="ID de la pieza artística", max_length=60)

    model_config = {
        "populate_by_name": True,
    }
