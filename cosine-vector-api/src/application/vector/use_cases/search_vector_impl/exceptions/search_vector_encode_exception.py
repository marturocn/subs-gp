"""
Exception for vector search encoding operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class SearchVectorEncodeException(CustomServerException):
    """
    Exception raised when there's an error during vector search encoding operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during vector search encoding.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during vector search encoding."
        super().__init__(
            code=3011,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during vector search encoding"}
        self.set_detail(detail)
