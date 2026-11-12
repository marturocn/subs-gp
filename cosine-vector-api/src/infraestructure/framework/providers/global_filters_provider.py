"""
Configuración de filtros globales para FastAPI.
"""
import logging
from fastapi import FastAPI, Request, Response

from common.exceptions.custom_server_exception import CustomServerException
from src.infraestructure.framework.filters.server_exception_filter import server_exception_handler
from src.infraestructure.framework.filters.unknown_exception_filter import unknown_exception_handler

logger = logging.getLogger(__name__)


def configure_global_exception_handlers(
        request: Request,
        exception_model: CustomServerException
) -> Response:
    """
    Configura los manejadores de excepciones globales para la aplicación FastAPI.
    
    Args:
        app: La instancia de la aplicación FastAPI
    """
    logger.info("🔧 Configurando manejadores de excepciones globales...")
        
    
    # Manejador para excepciones personalizadas del servidor
    exception = server_exception_handler(
        request,
        exception_model
    )

    
    logger.info("🎯 Todos los manejadores de excepciones configurados correctamente")

    return exception
