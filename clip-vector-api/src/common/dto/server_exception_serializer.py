"""
Serializer for server exceptions.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class ServerParameterSerializer(BaseModel):
    """
    Serializador para parámetros de error.
    """
    name: str
    value: Any
    
    class Config:
        from_attributes = True


class ErrorDetailSerializer(BaseModel):
    """
    Serializador para los detalles del error.
    """
    diagnosticInformation: Optional[str] = None
    parameters: Optional[List[ServerParameterSerializer]] = None
    
    class Config:
        from_attributes = True
        # Permitir campos adicionales
        extra = "allow"


class ErrorSerializer(BaseModel):
    """
    Serializador para la estructura de error.
    """
    code: int
    message: str
    detail: ErrorDetailSerializer
    
    class Config:
        from_attributes = True


class ServerExceptionSerializer(BaseModel):
    """
    Serializer for server exception responses.
    """
    
    status: int
    meta: Dict[str, Any]
    error: ErrorSerializer
    
    class Config:
        from_attributes = True
