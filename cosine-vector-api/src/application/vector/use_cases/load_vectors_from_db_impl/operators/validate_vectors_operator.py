"""
RxPY Operator for validating vector format and dimension.
"""
import logging
from typing import List, Tuple, Callable, Dict
from reactivex import Observable
from reactivex import operators as ops

from src.domain.piece_vector.models.entities.piece_vector import PieceVector
from src.application.vector.use_cases.load_vectors_from_db_impl.exceptions.load_vectors_invalid_format_exception import LoadVectorsInvalidFormatException


logger = logging.getLogger(__name__)


def _validate_piece_vector_has_faiss_id(piece_vector: PieceVector) -> bool:
    """Check if piece vector has a valid faiss_id."""
    return piece_vector.faiss_id is not None


def _validate_piece_vector_has_data(piece_vector: PieceVector) -> bool:
    """Check if piece vector has valid vector data."""
    return piece_vector.vector and len(piece_vector.vector) > 0


def _process_valid_vector(piece_vector: PieceVector, dimension_errors: Dict[int, int]) -> Tuple[int, List[float]]:
    """Process a valid vector and track its dimension."""
    vector_dim = len(piece_vector.vector)
    if vector_dim not in dimension_errors:
        dimension_errors[vector_dim] = 0
    dimension_errors[vector_dim] += 1
    return (piece_vector.faiss_id, piece_vector.vector)


def _log_validation_statistics(validated_count: int, invalid_count: int, 
                               skipped_no_faiss_id: int, dimension_errors: Dict[int, int]) -> None:
    """Log validation statistics."""
    logger.info(f"✅ [VALIDATE_VECTORS_OP] Validated {validated_count} vectors")
    if invalid_count > 0:
        logger.warning(f"⚠️ [VALIDATE_VECTORS_OP] Skipped {invalid_count} invalid vectors (empty)")
    if skipped_no_faiss_id > 0:
        logger.warning(f"⚠️ [VALIDATE_VECTORS_OP] Skipped {skipped_no_faiss_id} vectors without faiss_id")
    
    if dimension_errors:
        logger.info("📊 [VALIDATE_VECTORS_OP] Vector dimensions found:")
        for dim, count in sorted(dimension_errors.items()):
            logger.info(f"   - Dimension {dim}: {count} vectors")


def validate_vectors_operator() -> Callable[[Observable], Observable]:
    """
    RxPY operator that validates piece vectors format and dimension.
    Converts PieceVector entities to (faiss_id, vector) tuples for FAISS rebuilding.
    
    Returns:
        RxPY operator function that transforms List[PieceVector] to List[Tuple[int, List[float]]]
        Output format: [(faiss_id, vector), ...]
    """
    
    def validate_vectors_step(piece_vectors: List[PieceVector]) -> List[Tuple[int, List[float]]]:
        """Validate vectors format and convert to (faiss_id, vector) tuples."""
        try:
            logger.info(f"🔍 [VALIDATE_VECTORS_OP] Validating {len(piece_vectors)} piece vectors...")
            
            validated_vectors = []
            invalid_count = 0
            skipped_no_faiss_id = 0
            dimension_errors: Dict[int, int] = {}
            
            for piece_vector in piece_vectors:
                # Validar que tenga faiss_id
                if not _validate_piece_vector_has_faiss_id(piece_vector):
                    skipped_no_faiss_id += 1
                    logger.debug(f"⏭️ [VALIDATE_VECTORS_OP] Skipping vector {piece_vector.id}: no faiss_id")
                    continue
                
                # Validar que tenga vector
                if not _validate_piece_vector_has_data(piece_vector):
                    invalid_count += 1
                    logger.warning(f"⚠️ [VALIDATE_VECTORS_OP] Invalid vector for faiss_id {piece_vector.faiss_id}: empty vector")
                    continue
                
                # Procesar vector válido
                validated_tuple = _process_valid_vector(piece_vector, dimension_errors)
                validated_vectors.append(validated_tuple)
            
            # Log de estadísticas
            _log_validation_statistics(len(validated_vectors), invalid_count, 
                                      skipped_no_faiss_id, dimension_errors)
            
            if len(validated_vectors) == 0:
                logger.warning("⚠️ [VALIDATE_VECTORS_OP] No valid vectors found with faiss_id after validation")
            
            return validated_vectors
            
        except Exception as e:
            logger.error(f"❌ [VALIDATE_VECTORS_OP] Validation error: {e}")
            raise LoadVectorsInvalidFormatException(
                detail={"reason": "Error validating vectors format", "error": str(e)}
            )
    
    return ops.map(validate_vectors_step)
