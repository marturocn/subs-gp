"""
DTO for search image validation in vector operations.
"""
from pydantic import BaseModel, Field, ConfigDict
from fastapi import UploadFile
from typing import Any


class SearchImageDto(BaseModel):
    """
    DTO for validating search image file.
    """
    file: UploadFile = Field(..., description="Imagen a buscar en el índice FAISS")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True  # Permite tipos arbitrarios como UploadFile
    )