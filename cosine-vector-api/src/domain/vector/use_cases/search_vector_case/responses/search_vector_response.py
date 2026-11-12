"""
Response for search vector use case.
"""
from typing import List, Dict, Any
from common import ServerResponseModel


class SearchVectorResponse(ServerResponseModel[dict, dict]):
    """
    Response after successfully searching similar vectors.
    """
    def __init__(self, results: List[Dict[str, Any]]):
        data = {
            "results": results
        }
        super().__init__(data=data)
