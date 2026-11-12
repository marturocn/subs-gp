"""
DTO for piece ID validation in vector operations.
"""
from pydantic import BaseModel, Field


class PieceIdDto(BaseModel):
    """
    DTO for validating piece ID parameter.
    """
    id: int = Field(..., description="ID de la pieza artística")

    model_config = {
        "populate_by_name": True,
    }
