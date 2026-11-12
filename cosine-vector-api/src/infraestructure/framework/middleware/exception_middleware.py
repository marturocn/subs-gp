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
        """
        Maneja las excepciones de forma global.
        """
        try:
            logger.info(f"🔥 GlobalExceptionMiddleware: Procesando request: {request.method} {request.url}")
            
            # Log básico del request sin consumir el body
            logger.info(f"🔥 Request Method: {request.method}")
            logger.info(f"🔥 Request URL: {request.url}")
            logger.info(f"🔥 Request Headers: User-Agent={request.headers.get('user-agent', 'unknown')}, Content-Type={request.headers.get('content-type', 'unknown')}")
            
            # NO leer el body aquí para evitar consumir el stream
            # El body será leído por el endpoint correspondiente
            
            response = await call_next(request)
            
            # Verificar detalles de la respuesta antes de enviarla
            status_code = getattr(response, 'status_code', 'unknown')
            content_type = getattr(response, 'media_type', 'unknown')
            
            logger.info(f"✅ GlobalExceptionMiddleware: Request completado - Status: {status_code}, Content-Type: {content_type}")
            
            # Verificar si la respuesta tiene contenido
            if hasattr(response, 'body'):
                try:
                    body_length = len(getattr(response, 'body', b''))
                    logger.info(f"📦 GlobalExceptionMiddleware: Response body length: {body_length} bytes")
                except Exception:
                    logger.info("📦 GlobalExceptionMiddleware: Could not determine response body length")
            
            return response
        except CustomServerException as exc:
            print(f"🔥 GlobalExceptionMiddleware: Capturada CustomServerException: {type(exc).__name__}")
            logger.info(f"🔥 GlobalExceptionMiddleware: Capturada CustomServerException: {type(exc).__name__}")
            logger.error(f"🔥 Full traceback for CustomServerException: {traceback.format_exc()}")
            
            try:
                response_data = exc.get_response()
                logger.info(f"🔥 Serializando CustomServerException con respuesta: {response_data}")
                return JSONResponse(
                    status_code=exc.get_status(),
                    content=response_data
                )
            except (ValueError, TypeError, AttributeError) as serialization_error:
                print(f"🔥 Error al serializar CustomServerException: {serialization_error}")
                logger.error(f"🔥 Error al serializar CustomServerException: {serialization_error}")
                logger.error(f"🔥 Full traceback for serialization error: {traceback.format_exc()}")
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
            logger.error(f"🔥 Full traceback: {traceback.format_exc()}")
            
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
                logger.error(f"🔥 Error al crear UnknownException: {unknown_error}")
                logger.error(f"🔥 Full traceback for unknown error: {traceback.format_exc()}")
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
