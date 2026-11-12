"""
Invalid Vector Format Exception for Load Vectors
"""
from typing import Optional, Dict, Any, List

from src.common.exceptions.custom_server_exception import CustomServerException


class LoadVectorsInvalidFormatException(CustomServerException):
    """
    Exception raised when vectors have invalid format or dimension.
    Code: 3004
    """

    def __init__(
        self,
        message: Optional[str] = "One or more vectors have invalid format or dimension.",
        detail: Optional[Dict[str, Any]] = None,
        invalid_ids: Optional[List[str]] = None
    ):
        safe_message = message if message is not None else "One or more vectors have invalid format or dimension."
        super().__init__(
            code=3004,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "One or more vectors with invalid shape"}
            if invalid_ids:
                detail["invalid_ids"] = invalid_ids
        self.set_detail(detail)
