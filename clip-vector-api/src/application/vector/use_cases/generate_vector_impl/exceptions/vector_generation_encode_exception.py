"""
Exception for vector generation encoding operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class VectorGenerationEncodeException(CustomServerException):
    """
    Exception raised when there's an error during vector generation encoding operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during vector generation encoding.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during vector generation encoding."
        super().__init__(
            code=3006,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during vector generation encoding"}
        self.set_detail(detail)
