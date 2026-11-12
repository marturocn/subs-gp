"""
Exception for unknown errors.
"""
from http import HTTPStatus
from typing import Any, Dict, List, Optional
from .custom_server_exception import CustomServerException, ServerParameterModel
from .codes_exception import CodeException
from .messages_exception import MessageException


class UnknownException(CustomServerException):

    def __init__(
        self,
        detail: Optional[Dict[str, Any]] = None,
        parameters: Optional[List[ServerParameterModel]] = None
    ):
        super().__init__(
            code=CodeException.UNKNOWN_ERROR,  # Código de error para validación genérica
            message=MessageException.UNKNOWN_ERROR,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
        if detail:
            self.set_detail(detail)
        if parameters:
            self.set_parameters(parameters)
    