"""
Modelo para respuestas del servidor.
"""
from http import HTTPStatus
from typing import Any, Dict, Generic, TypeVar, Optional, Union
from enum import IntEnum



# Tipos genéricos para los datos y metadatos
TData = TypeVar('TData')
TMeta = TypeVar('TMeta', bound=Dict[str, Any])


class ServerResponseModel(Generic[TData, TMeta]):
    """
    Modelo genérico para respuestas del servidor equivalente al modelo TypeScript.
    """
    
    def __init__(
        self, 
        data: TData, 
        meta: Optional[Union[TMeta, Dict[str, Any]]] = None, 
        status: HTTPStatus = HTTPStatus.OK
    ):
        """
        Inicializa el modelo de respuesta del servidor.
        
        Args:
            data: Los datos de la respuesta
            meta: Metadatos adicionales
            status: Código de estado HTTP
        """
        self.status = status
        self.meta = meta if meta is not None else {}
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el modelo a diccionario.
        
        Returns:
            Diccionario con los datos del modelo
        """
        return {
            "status": self.status.value,
            "meta": self.meta,
            "data": self.data
        }
    
    def set_status(self, status: HTTPStatus) -> "ServerResponseModel[TData, TMeta]":
        """
        Establece el código de estado.
        
        Args:
            status: Código de estado HTTP
            
        Returns:
            La instancia actual para encadenamiento de métodos
        """
        self.status = status
        return self
    
    def set_meta(self, meta: TMeta) -> "ServerResponseModel[TData, TMeta]":
        """
        Establece los metadatos.
        
        Args:
            meta: Metadatos a establecer
            
        Returns:
            La instancia actual para encadenamiento de métodos
        """
        self.meta = meta
        return self
    
    def set_data(self, data: TData) -> "ServerResponseModel[TData, TMeta]":
        """
        Establece los datos.
        
        Args:
            data: Datos a establecer
            
        Returns:
            La instancia actual para encadenamiento de métodos
        """
        self.data = data
        return self
    
    def __repr__(self) -> str:
        """Representación string del objeto."""
        return f"ServerResponseModel(status={self.status}, meta={self.meta}, data={self.data})"
    
    def __eq__(self, other: object) -> bool:
        """Comparación de igualdad."""
        if not isinstance(other, ServerResponseModel):
            return False
        return (
            self.status == other.status and 
            self.meta == other.meta and 
            self.data == other.data
        )
