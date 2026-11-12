"""
General Load Vectors Exception
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class LoadVectorsDbException(CustomServerException):
    """
    Base exception for load vectors from database operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during vector loading operation.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during vector loading operation."
        super().__init__(
            code=5000,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during vector loading operation"}
        self.set_detail(detail)
