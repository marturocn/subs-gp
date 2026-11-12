"""
Exception for add vector use case operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class AddVectorException(CustomServerException):
    """
    Exception raised when there's an error during add vector operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during add vector operation.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during add vector operation."
        super().__init__(
            code=3004,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during add vector operation"}
        self.set_detail(detail)
