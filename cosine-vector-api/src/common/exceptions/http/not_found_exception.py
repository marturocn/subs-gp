"""
Excepción personalizada para recursos no encontrados.
"""
from typing import Optional, Dict, Any
from ..custom_server_exception import CustomServerException
from ..codes_exception import CodeException
from ..messages_exception import MessageException


class NotFoundException(CustomServerException):
    """
    Excepción para cuando un recurso no es encontrado.
    """
    
    def __init__(
        self, 
        resource_name: str = "Recurso", 
        resource_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None
    ):
        message = f"{resource_name}"
        if resource_id:
            message += f" con ID {resource_id}"
        message += " no encontrado"
        
        super().__init__(
            code=CodeException.NOT_FOUND,
            message=message,
            status_code=404
        )
        
        if detail:
            self.set_detail(detail)
        
        # Agregar información específica del recurso
        self.set_detail({
            "resource_name": resource_name,
            "resource_id": resource_id
        })
