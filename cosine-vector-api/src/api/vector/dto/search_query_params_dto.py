"""
DTO for search query parameters validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class SearchQueryParamsDto(BaseModel):
    """
    DTO for validating search query parameters.
    """
    threshold: Optional[float] = Field(
        0.5,
        description="Valor mínimo de similitud para incluir un resultado (rango: 0.0 a 1.0). Por defecto 0.5.",
        ge=0.0,
        le=1.0
    )
    
    maxResults: Optional[int] = Field(
        None,
        description="Número máximo de resultados a retornar (debe ser >= 0)",
        ge=0
    )
    
    enableObjectDetector: Optional[bool] = Field(
        True,
        description="Habilita/deshabilita la detección de objetos. Si es true, usa search_vectors_operator (piece_item_vector). Si es false, usa búsqueda simple por coseno (piece_vector)."
    )
    
    model_config = ConfigDict(
        populate_by_name=True
    )
    