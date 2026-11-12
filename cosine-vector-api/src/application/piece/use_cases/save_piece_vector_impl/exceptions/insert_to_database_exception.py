from http import HTTPStatus
from typing import Optional, Dict, Any

from src.common import CustomServerException

class InsertToDatabaseException(CustomServerException):
	
	def __init__(
		self,
		message: str = "Failed to store vector in database.",
		detail: Optional[Dict[str, Any]] = None
	):
		super().__init__(
			code=3003,
			message=message,
			status_code=HTTPStatus.INTERNAL_SERVER_ERROR
		)
		if detail is None:
			detail = {"reason": "Failed to insert into DB"}
		self.set_detail(detail)


