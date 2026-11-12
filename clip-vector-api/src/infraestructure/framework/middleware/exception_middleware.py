"""
Middleware personalizado para manejo de excepciones.
"""
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.common.exceptions.custom_server_exception import CustomServerException
from src.common.exceptions.unknown_exception import UnknownException
import traceback

logger = logging.getLogger(__name__)


class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """
    Middleware global para capturar todas las excepciones.
    Se ejecuta antes que otros middlewares para asegurar que capture todas las excepciones.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        print(f"🔥 init: Procesando request: {request.url}")
        """
        Maneja las excepciones de forma global.
        """
        try:
            logger.info(f"🔥 GlobalExceptionMiddleware: Procesando request: {request.url}")
            response = await call_next(request)
            return response
        except CustomServerException as exc:
            print(f"🔥 GlobalExceptionMiddleware: Capturada CustomServerException: {type(exc).__name__}")
            logger.info(f"🔥 GlobalExceptionMiddleware: Capturada CustomServerException: {type(exc).__name__}")
            
            try:
                response_data = exc.get_response()
                return JSONResponse(
                    status_code=exc.get_status(),
                    content=response_data
                )
            except Exception as serialization_error:
                print(f"🔥 Error al serializar CustomServerException: {serialization_error}")
                logger.error(f"🔥 Error al serializar CustomServerException: {serialization_error}")
                # Fallback a respuesta simple
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": 500,
                        "meta": {},
                        "error": {
                            "code": 9999,
                            "message": "Error interno del servidor",
                            "detail": {"error": str(serialization_error)}
                        }
                    }
                )
        except Exception as exc:
            print(f"🔥 GlobalExceptionMiddleware: Capturada Exception: {type(exc).__name__}: {str(exc)}")
            print(f"🔥 MRO de la excepción: {[cls.__name__ for cls in type(exc).__mro__]}")
            logger.error(f"🔥 GlobalExceptionMiddleware: Capturada Exception: {type(exc).__name__}: {str(exc)}")
            
            # Verificar si es una CustomServerException que no fue capturada por el primer except
            if isinstance(exc, CustomServerException):
                print("🔥 La excepción ES una CustomServerException pero no fue capturada en el primer bloque")
                try:
                    response_data = exc.get_response()
                    return JSONResponse(
                        status_code=exc.get_status(),
                        content=response_data
                    )
                except Exception as serialization_error:
                    print(f"🔥 Error al serializar en fallback: {serialization_error}")
            
            # Crear una UnknownException
            try:
                stack_trace = f"{type(exc).__name__}: {str(exc)}"
                unknown_exc = UnknownException(
                    detail={"stack": stack_trace}
                )
                response_data = unknown_exc.get_response()
                
                return JSONResponse(
                    status_code=unknown_exc.get_status(),
                    content=response_data
                )
            except Exception as unknown_error:
                print(f"🔥 Error al crear UnknownException: {unknown_error}")
                # Respuesta de emergencia
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": 500,
                        "meta": {},
                        "error": {
                            "code": 9999,
                            "message": "Error crítico del servidor",
                            "detail": {"original_error": str(exc), "unknown_error": str(unknown_error)}
                        }
                    }
                )
