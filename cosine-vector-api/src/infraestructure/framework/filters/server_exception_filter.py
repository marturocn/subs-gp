"""
Server exception handler for FastAPI.
"""
import logging
from typing import Union
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from common.exceptions.custom_server_exception import CustomServerException

logger = logging.getLogger(__name__)


def server_exception_handler(_: Request, exc: CustomServerException) -> Response:
    """
    Custom server exception handler.
    
    Args:
        _: The HTTP request (required by FastAPI but not used)
        exc: The server exception
        
    Returns:
        JSONResponse with the serialized exception following the format
    """
    logger.debug(f"🔧 ServerExceptionFilter: Processing exception: {type(exc).__name__}")

    response_data = exc.get_response()

    logger.debug(f"🔥 ServerExceptionFilter: Generated response: {response_data}")

    return JSONResponse(
        status_code=exc.get_status(),
        content=response_data
    )
