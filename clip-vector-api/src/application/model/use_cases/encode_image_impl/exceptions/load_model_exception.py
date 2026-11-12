from http import HTTPStatus
from typing import Optional, Dict, Any

from src.common import CustomServerException

class LoadModelException(CustomServerException):
    def __init__(
        self,
        message: Optional[str] = "Error loading the CLIP model.",
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            code=3009,
            message=message or "Error loading the CLIP model.",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
        if detail is None:
            detail = {"reason": "Internal error with the index"}
        self.set_detail(detail)
