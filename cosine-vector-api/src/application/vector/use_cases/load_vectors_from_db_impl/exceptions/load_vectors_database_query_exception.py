"""
Database Query Exception for Load Vectors
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class LoadVectorsDatabaseQueryException(CustomServerException):
    """
    Exception raised when there's an error querying the database for vectors.
    Code: 1005
    """

    def __init__(
        self,
        message: Optional[str] = "Error loading vectors from database.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error loading vectors from database."
        super().__init__(
            code=1005,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error querying the DB"}
        self.set_detail(detail)
