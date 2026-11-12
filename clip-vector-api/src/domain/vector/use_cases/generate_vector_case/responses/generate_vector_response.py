from typing import List, Dict, Any
from common import ServerResponseModel


class GenerateVectorResponse(ServerResponseModel[dict, dict]):
    """
    Response after successfully generating vectors for piece and detected objects.
    """
    def __init__(self, piece_vector: List[float], objects: List[Dict[str, Any]]):
        data = {
            "vectors": {
                "piece": piece_vector,
                "objects": objects
            }
        }
        super().__init__(data=data)
