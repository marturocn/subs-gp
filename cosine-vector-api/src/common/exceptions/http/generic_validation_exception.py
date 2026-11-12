"""
Excepción personalizada para errores de validación genérica.
"""
from typing import Optional, Dict, Any, List
from ..custom_server_exception import CustomServerException, ServerParameterModel

class GenericValidationException(CustomServerException):
    """
    Excepción personalizada de ejemplo.
    """
    
    def __init__(
        self, 
        message: Optional[str] = "Invalid generic exception",
        detail: Optional[Dict[str, Any]] = None,
        parameters: Optional[List[ServerParameterModel]] = None
    ):
        super().__init__(
            code=1001,  # Código de error personalizado
            message=message if message is not None else "Invalid generic exception",
            status_code=400
        )
        
        if detail:
            self.set_detail(detail)
        
        if parameters:
            self.set_parameters(parameters)

