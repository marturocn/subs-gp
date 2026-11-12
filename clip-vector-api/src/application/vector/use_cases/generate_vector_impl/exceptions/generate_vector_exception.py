"""
Exception for generate vector use case operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class GenerateVectorException(CustomServerException):
    """
    Exception raised when there's an error during generate vector operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during vector generation.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during vector generation."
        super().__init__(
            code=3005,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during vector generation"}
        self.set_detail(detail)
