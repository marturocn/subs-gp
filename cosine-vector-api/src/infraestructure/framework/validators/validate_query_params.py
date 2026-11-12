from fastapi import Depends, Query, Request
from pydantic import BaseModel, ValidationError
from typing import Type, Any, Optional
import inspect
from src.common.exceptions.http.generic_validation_exception import GenericValidationException

# Constante para evitar duplicación
INVALID_VALUE_MESSAGE = "Invalid value"


class ValidateQueryParams:
    """
    Validador para query parameters usando Pydantic models.
    Similar a ValidateParams pero para query strings en lugar de path params.
    """
    
    def __init__(self, dto_class: Type[BaseModel], description: Optional[str] = None):
        self.dto_class = dto_class
        self.description = description or f"Query parameters for {dto_class.__name__}"

    def __call__(self) -> Any:
        """
        Crea un dependency para FastAPI que valida query parameters.
        Usa Request para capturar los query params sin validación previa de FastAPI.
        """
        validator = self
        dto_class = self.dto_class
        model_fields = self.dto_class.model_fields
        
        # Crear función de dependencia que usa Request para obtener query params
        def dependency(request: Request) -> BaseModel:
            try:
                # Obtener query params como dict
                query_params = dict(request.query_params)
                
                # Filtrar solo los campos que están en el modelo
                filtered_params = {}
                for field_name in model_fields.keys():
                    if field_name in query_params:
                        filtered_params[field_name] = query_params[field_name]
                
                # Dejar que Pydantic haga la validación y conversión de tipos
                return dto_class(**filtered_params)
                
            except ValidationError as exc:
                raise validator._handle_validation_error(exc)
        
        return Depends(dependency)
    
    def _handle_validation_error(self, exc: ValidationError) -> GenericValidationException:
        """Maneja errores de validación de Pydantic"""
        error = exc.errors()[0]
        field = error.get("loc", ["unknown"])[-1]
        msg = error.get("msg", INVALID_VALUE_MESSAGE)
        error_type = error.get("type", "unknown")
        
        # Personalizar mensajes según el tipo de error
        if "int_parsing" in error_type:
            reason = "Input should be a valid integer, unable to parse string as an integer"
        elif "float_parsing" in error_type:
            reason = "Input should be a valid float, unable to parse string as a number"
        elif "greater_than_equal" in error_type:
            limit = error.get("ctx", {}).get("ge", "")
            reason = f"Input should be greater than or equal to {limit}"
        elif "less_than_equal" in error_type:
            limit = error.get("ctx", {}).get("le", "")
            reason = f"Input should be less than or equal to {limit}"
        else:
            reason = msg
        
        detail = {
            "field": field,
            "reason": reason
        }
        raise GenericValidationException(detail=detail)
