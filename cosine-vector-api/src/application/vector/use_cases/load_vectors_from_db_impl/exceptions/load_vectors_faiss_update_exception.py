"""
FAISS Update Exception for Load Vectors
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class LoadVectorsFaissUpdateException(CustomServerException):
    """
    Exception raised when there's an error updating the FAISS index.
    Code: 1008
    """

    def __init__(
        self,
        message: Optional[str] = "Error updating FAISS index.",
        detail: Optional[Dict[str, Any]] = None,
        faiss_error: Optional[str] = None
    ):
        safe_message = message if message is not None else "Error updating FAISS index."
        super().__init__(
            code=1008,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "General FAISS index failure"}
            if faiss_error:
                detail["faiss_error"] = faiss_error
        self.set_detail(detail)
