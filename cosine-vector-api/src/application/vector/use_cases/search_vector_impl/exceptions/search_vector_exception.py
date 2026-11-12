"""
Search Vector Exceptions
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class SearchVectorException(CustomServerException):
    """
    Exception raised when there's an error during vector search operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during vector search.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during vector search."
        super().__init__(
            code=3003,
            message=safe_message,
            status_code=400
        )
        if detail is None:
            detail = {"reason": "Error occurred during vector search"}
        self.set_detail(detail)
