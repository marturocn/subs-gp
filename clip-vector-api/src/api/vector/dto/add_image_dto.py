"""
DTO for search image validation in vector operations.
"""
from pydantic import BaseModel, Field, ConfigDict
from fastapi import UploadFile
from typing import Any


class AddImageDto(BaseModel):
    """
    DTO for validating add image file.
    """
    file: UploadFile = Field(..., description="Imagen a añadir al índice FAISS")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True  # Permite tipos arbitrarios como UploadFile
    )