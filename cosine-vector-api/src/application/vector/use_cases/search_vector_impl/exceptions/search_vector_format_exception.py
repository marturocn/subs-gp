"""
Exception for vector search result formatting operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class SearchVectorFormatException(CustomServerException):
    """
    Exception raised when there's an error during vector search result formatting operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during search result formatting.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during search result formatting."
        super().__init__(
            code=3012,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during search result formatting"}
        self.set_detail(detail)
