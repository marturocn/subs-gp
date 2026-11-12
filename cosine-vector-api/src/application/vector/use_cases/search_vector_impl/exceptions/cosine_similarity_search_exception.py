"""
Exception for cosine similarity search operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class CosineSimilaritySearchException(CustomServerException):
    """
    Exception raised when there's an error during cosine similarity search operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during cosine similarity search.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during cosine similarity search."
        super().__init__(
            code=3010,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during cosine similarity search"}
        self.set_detail(detail)
