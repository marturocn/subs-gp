"""
Excepción base personalizada que NO hereda de HTTPException.
"""
from typing import Any, Dict, Optional, List
import os
import json


class ServerParameterModel:
    """
    Modelo para parámetros de error.
    """
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el parámetro a diccionario."""
        return {
            "name": self.name,
            "value": self.value
        }


class CustomServerException(Exception):
    """
    Excepción base personalizada que NO hereda de HTTPException.
    Esto evita que FastAPI interfiera con nuestro manejo de excepciones.
    """
    
    def __init__(
        self,
        code: int,
        message: str,
        status_code: int = 500,
        diagnostic_information: Optional[Any] = None,
        headers: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        
        self.status_code = status_code
        self.headers = headers
        
        # Initialize with explicit type hint for mypy
        self.meta: Dict[str, Any] = {}
        self.error: Dict[str, Any] = {
            "code": code,
            "message": message,
            "detail": {}  # Initialize detail as a Dict[str, Any]
        }
        
        # Solo incluir información de diagnóstico en modo debug
        if os.getenv('LOG_LEVEL') == 'DEBUG2' and diagnostic_information:
            # Ensure 'detail' is treated as a Dict[str, Any]
            detail_dict: Dict[str, Any] = self.error["detail"]
            detail_dict["diagnosticInformation"] = json.dumps(diagnostic_information) if not isinstance(diagnostic_information, str) else str(diagnostic_information)
    
    def get_status(self) -> int:
        """Get the HTTP status code."""
        return self.status_code
    
    def set_code(self, code: int) -> "CustomServerException":
        """Set the error code."""
        self.error["code"] = code
        return self
    
    def set_message(self, message: str) -> "CustomServerException":
        """Set the exception message."""
        self.error["message"] = message
        return self
    
    def set_diagnostic_information(self, stack: Any) -> "CustomServerException":
        """Set diagnostic information."""
        if os.getenv('LOG_LEVEL') == 'DEBUG':
            # Ensure 'detail' is treated as a Dict[str, Any]
            detail_dict: Dict[str, Any] = self.error["detail"]
            detail_dict["diagnosticInformation"] = json.dumps(stack) if not isinstance(stack, str) else str(stack)
        return self
    
    def set_meta(self, meta: Dict[str, Any]) -> "CustomServerException":
        """Set metadata."""
        self.meta = meta
        return self
    
    def set_detail(self, detail: Dict[str, Any]) -> "CustomServerException":
        """Set error details."""
        # Ensure 'detail' is treated as a Dict[str, Any] before updating
        current_detail: Dict[str, Any] = self.error["detail"]
        self.error["detail"] = {**current_detail, **detail}
        return self
    
    def set_parameters(self, parameters: List[ServerParameterModel]) -> "CustomServerException":
        """Set error parameters."""
        # Ensure 'detail' is treated as a Dict[str, Any]
        detail_dict: Dict[str, Any] = self.error["detail"]
        detail_dict["parameters"] = [param.to_dict() for param in parameters]
        return self
    
    def get_response(self) -> Dict[str, Any]:
        """
        Obtiene la respuesta estructurada.
        """
        return {
            "status": self.get_status(),
            "meta": self.meta or {},
            "error": {
                "code": self.error["code"],
                "message": self.error["message"],
                "detail": self.error["detail"]
            }
        }
