"""
Specific exceptions for search vector operators.
"""
from typing import Optional, Dict, Any

from src.common.exceptions.custom_server_exception import CustomServerException


class SearchSimilarVectorsException(CustomServerException):
    """
    Exception raised when there's an error during similar vector search operations.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during similar vector search.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during similar vector search."
        super().__init__(
            code=3004,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during similar vector search"}
        self.set_detail(detail)


class CosineSimilarityException(CustomServerException):
    """
    Exception raised when there's an error during cosine similarity calculation.
    """

    def __init__(
        self,
        message: Optional[str] = "Error occurred during cosine similarity calculation.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Error occurred during cosine similarity calculation."
        super().__init__(
            code=3005,
            message=safe_message,
            status_code=500
        )
        if detail is None:
            detail = {"reason": "Error occurred during cosine similarity calculation"}
        self.set_detail(detail)


class NoSimilarVectorsFoundException(CustomServerException):
    """
    Exception raised when no similar vectors are found during search.
    """

    def __init__(
        self,
        message: Optional[str] = "No similar vectors found.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "No similar vectors found."
        super().__init__(
            code=3006,
            message=safe_message,
            status_code=404
        )
        if detail is None:
            detail = {"reason": "No similar vectors found in the index"}
        self.set_detail(detail)


class InvalidQueryVectorException(CustomServerException):
    """
    Exception raised when the query vector is invalid (e.g., zero magnitude).
    """

    def __init__(
        self,
        message: Optional[str] = "Invalid query vector.",
        detail: Optional[Dict[str, Any]] = None
    ):
        safe_message = message if message is not None else "Invalid query vector."
        super().__init__(
            code=3007,
            message=safe_message,
            status_code=400
        )
        if detail is None:
            detail = {"reason": "Query vector is invalid"}
        self.set_detail(detail)
