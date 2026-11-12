from typing import Optional, Any, Dict
from common import ServerResponseModel


class AddVectorResponse(ServerResponseModel[dict, dict]):
    """
    Response after successfully adding a vector.
    Ahora incluye opcionalmente `pieceVectorId` en el mensaje cuando esté disponible.
    """
    def __init__(self, piece_id: str, piece_vector_id: Optional[int] = None):
        message: Dict[str, Any] = {
            "message": "Vector added successfully",
            "pieceId": piece_id
        }

        # Incluir pieceVectorId si se proporciona
        if piece_vector_id is not None:
            message["pieceVectorId"] = piece_vector_id

        data = {"message": message}
        super().__init__(data=data)
