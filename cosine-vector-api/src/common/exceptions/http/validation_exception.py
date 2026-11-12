"""
Excepción personalizada para errores de validación.
"""
from typing import Optional, Dict, Any, List
from ..custom_server_exception import CustomServerException, ServerParameterModel
from ..codes_exception import CodeException
from ..messages_exception import MessageException


class ValidationException(CustomServerException):
    """
    Excepción para errores de validación de datos.
    """
    
    def __init__(
        self, 
        message: str = "Error de validación",
        field_errors: Optional[List[Dict[str, str]]] = None,
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            code=CodeException.VALIDATION_ERROR,
            message=message,
            status_code=400
        )
        
        if detail:
            self.set_detail(detail)
        
        if field_errors:
            self.set_detail({"field_errors": field_errors})
