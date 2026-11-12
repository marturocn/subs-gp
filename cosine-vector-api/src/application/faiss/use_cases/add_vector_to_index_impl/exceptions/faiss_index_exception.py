from http import HTTPStatus
from typing import Optional, Dict, Any

from src.common import CustomServerException

class FaissIndexException(CustomServerException):
    def __init__(
        self,
        message: str = "Internal error adding vector to index.",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            code=3008,
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
        if detail is None:
            detail = {"reason": "Internal error with the index"}
        self.set_detail(detail)
