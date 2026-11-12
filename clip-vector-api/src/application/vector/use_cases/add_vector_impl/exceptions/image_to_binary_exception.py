
"""
Exception for image to binary conversion operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class ImageToBinaryException(CustomServerException):
    """
    Exception raised when there's an error during image to binary conversion.
    """

    def __init__(
        self,
        message: Optional[str] = "Invalid or unreadable image.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Invalid or unreadable image."
        super().__init__(
            code=3002,
            message=safe_message,
            status_code=400
        )
        if detail is None:
            detail = {"reason": "Invalid or unreadable image"}
        self.set_detail(detail)

