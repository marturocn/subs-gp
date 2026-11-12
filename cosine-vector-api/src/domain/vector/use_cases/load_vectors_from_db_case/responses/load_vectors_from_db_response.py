"""
Response for load vectors from database use case.
"""
from common import ServerResponseModel


class LoadVectorsFromDbResponse(ServerResponseModel[dict, dict]):
    """
    Response after successfully loading vectors from database.
    """
    def __init__(self, message: str, loaded_count: int):
        data = {
            "message": message,
            "loaded_count": loaded_count
        }
        super().__init__(data=data)