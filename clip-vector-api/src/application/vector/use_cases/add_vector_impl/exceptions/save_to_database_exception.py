"""
Exception for database save operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class SaveToDatabaseException(CustomServerException):
    """
    Exception raised when there's an error during database save operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during database save operation.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during database save operation."
        super().__init__(
            code=3002,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during database save operation"}
        self.set_detail(detail)
