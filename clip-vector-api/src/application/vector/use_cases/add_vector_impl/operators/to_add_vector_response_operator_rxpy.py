"""
RxPY Operator for transforming final result to AddVectorResponse.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
from typing import Callable, Dict, Any
from reactivex import Observable
from reactivex.operators import map

from src.domain.vector.use_cases.add_vector_case.responses.add_vector_response import AddVectorResponse


def to_add_vector_response_operator(
    logger: logging.Logger
) -> Callable[[Observable], Observable[AddVectorResponse]]:
    """
    Operator para transformar el resultado final a AddVectorResponse.

    :param logger: Logger instance
    :return: Observable con AddVectorResponse
    """
    def operator(source: Observable) -> Observable[AddVectorResponse]:
        def transform(result: Dict[str, Any]) -> AddVectorResponse:
            piece_id = result["piece_id"]
            # num_vectors_added puede no existir en todos los pipelines; usar get
            num_vectors = result.get("num_vectors_added", None)

            # Obtener el piece_vector id guardado si está disponible
            saved_piece_vector = result.get("saved_piece_vector")
            piece_vector_id = None
            if isinstance(saved_piece_vector, dict):
                piece_vector_id = saved_piece_vector.get("id")

            logger.info(f"🎉 Pipeline completado exitosamente para piece_id={piece_id}")
            if num_vectors is not None:
                logger.info(f"   - {num_vectors} vectores guardados y agregados a FAISS")
            if piece_vector_id is not None:
                logger.info(f"   - pieceVectorId={piece_vector_id} guardado en DB")

            return AddVectorResponse(piece_id=piece_id, piece_vector_id=piece_vector_id)
        
        return source.pipe(
            map(transform)
        )
    return operator
