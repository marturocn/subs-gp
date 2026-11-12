from fastapi import Depends, Request, Path
from pydantic import BaseModel
from typing import Type, Any
import inspect
from src.common.exceptions.http.generic_validation_exception import GenericValidationException

# Constante para evitar duplicación
INVALID_VALUE_MESSAGE = "Invalid value"

class ValidateParams:
    def __init__(self, dto_class: Type[BaseModel]):
        self.dto_class = dto_class

    def __call__(self)-> Any:
        # Para DTOs con un solo campo, crear parámetro que aparezca en Swagger
        model_fields = self.dto_class.model_fields
        
        if len(model_fields) == 1:
            field_name, field_info = next(iter(model_fields.items()))
            description = field_info.description or f"Parámetro {field_name}"
            
            # Crear una función dinámicamente con el nombre correcto del parámetro
            def create_dependency():
                # Usar exec para crear dinámicamente una función con el parámetro correcto
                func_code = f"""
def dependency({field_name}: Any = Path(..., description="{description}")):
    try:
        return dto_class({field_name}={field_name})
    except Exception as exc:
        return validator._handle_validation_error(exc)
"""
                namespace = {
                    'Any': Any,
                    'Path': Path,
                    'dto_class': self.dto_class,
                    'validator': self
                }
                exec(func_code, namespace)
                return namespace['dependency']
            
            return Depends(create_dependency())
        else:
            # Para múltiples campos, usar Request
            def dependency(request: Request):
                return self._validate_request(request)
            return Depends(dependency)
    
    def _validate_request(self, request: Request):
        """Valida los parámetros de la request"""
        data = request.path_params
        try:
            return self.dto_class(**data)
        except Exception as exc:
            return self._handle_validation_error(exc)
    
    def _handle_validation_error(self, exc: Exception):
        """Maneja errores de validación"""
        from pydantic import ValidationError
        if isinstance(exc, ValidationError): # pyright: ignore[reportArgumentType]
            error = exc.errors()[0]
            field = error.get("loc", ["unknown"])[-1]
            reason = error.get("msg", INVALID_VALUE_MESSAGE)
            error_type = error.get("type", "unknown")
            detail = {
                "field": field,
                "reason": reason or error_type
            }
            raise GenericValidationException(detail=detail)
        else:
            raise exc
