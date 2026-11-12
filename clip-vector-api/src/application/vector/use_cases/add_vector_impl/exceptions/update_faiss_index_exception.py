"""
Exception for FAISS index update operations.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class UpdateFaissIndexException(CustomServerException):
    """
    Exception raised when there's an error during FAISS index update operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during FAISS index update.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during FAISS index update."
        super().__init__(
            code=3003,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during FAISS index update"}
        self.set_detail(detail)
