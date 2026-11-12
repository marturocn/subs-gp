"""
Unknown exception handler for FastAPI.
"""
import logging
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from src.common.exceptions.unknown_exception import UnknownException

logger = logging.getLogger(__name__)

def unknown_exception_handler(_: Request, raw_exception: Exception) -> Response:
    """
    Handler for unknown or unhandled exceptions.
    
    Args:
        _: The HTTP request (required by FastAPI but not used)
        raw_exception: The unhandled exception
        
    Returns:
        JSONResponse with the serialized exception following the format
    """
    logger.error(f"� UnknownExceptionFilter: Unhandled exception: {type(raw_exception).__name__}: {str(raw_exception)}")
    
    # Create a server exception based on the raw exception
    # Use str(raw_exception) to include the original error message
    stack_trace = f"Exception: {str(raw_exception)}\n{traceback.format_exc()}"
    exception = UnknownException(detail={"stack": stack_trace})
    
    response_data = exception.get_response()
    
    logger.error(f"🔥 UnknownExceptionFilter: Generated response: {response_data}")
    
    return JSONResponse(
        status_code=exception.get_status(),
        content=response_data
    )
