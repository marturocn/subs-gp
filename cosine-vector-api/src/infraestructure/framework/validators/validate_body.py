

# Decorador tipo clase para validación de DTOs
from fastapi import Depends, Request
from src.common.exceptions.http.generic_validation_exception import GenericValidationException


class ValidateBody:
    def __init__(self, dto_class):
        self.dto_class = dto_class

    def __call__(self):
        async def dependency(request: Request):
            data = await request.json()
            try:
                return self.dto_class(**data)
            except Exception as exc:
                from pydantic import ValidationError
                if isinstance(exc, ValidationError):
                    error = exc.errors()[0]
                    field = error.get("loc", ["unknown"])[-1]
                    reason = error.get("msg", "Invalid value")
                    detail = {
                        "field": field,
                        "reason": reason
                    }
                    raise GenericValidationException(detail=detail)
                else:
                    raise exc
        return Depends(dependency)