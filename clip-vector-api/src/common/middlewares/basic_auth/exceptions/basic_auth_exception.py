
from http import HTTPStatus
from typing import Optional, Dict, Any
from src.common.exceptions.custom_server_exception import CustomServerException

class BasicAuthException(CustomServerException):
	def __init__(
		self,
		message: str = "Invalid Basic Authentication credentials.",
		detail: Optional[Dict[str, Any]] = None
	):
		super().__init__(
			code=1004,
			message=message,
			status_code=HTTPStatus.UNAUTHORIZED
		)
		if detail is None:
			detail = {}
		self.set_detail(detail)
