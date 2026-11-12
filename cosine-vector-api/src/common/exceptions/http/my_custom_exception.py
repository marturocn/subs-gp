"""
Excepción personalizada ejemplo - Mi excepción personalizada.
"""
from typing import Optional, Dict, Any, List

from common.exceptions.custom_server_exception import CustomServerException, ServerParameterModel

class MyCustomException(CustomServerException):
    """
    Excepción personalizada de ejemplo.
    """
    
    def __init__(
        self, 
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        parameters: Optional[List[ServerParameterModel]] = None
    ):
        super().__init__(
            code=1001,  # Código de error personalizado
            message=message,
            status_code=400
        )
        
        if detail:
            self.set_detail(detail)
        
        if parameters:
            self.set_parameters(parameters)
