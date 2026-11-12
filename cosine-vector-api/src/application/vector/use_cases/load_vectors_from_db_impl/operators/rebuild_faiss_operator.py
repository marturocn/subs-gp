"""
RxPY Operator for rebuilding FAISS index with GPU support.
"""
import logging
from typing import List, Tuple, Callable
from reactivex import Observable
from reactivex import operators as ops

from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.application.vector.use_cases.load_vectors_from_db_impl.exceptions.load_vectors_faiss_update_exception import LoadVectorsFaissUpdateException


logger = logging.getLogger(__name__)


def rebuild_faiss_operator(faiss_repository: FaissIndexRepository) -> Callable[[Observable], Observable]:
    """
    RxPY operator that rebuilds FAISS index with vectors using GPU acceleration.
    
    Args:
        faiss_repository: FAISS repository for rebuilding the index
    
    Returns:
        RxPY operator function that transforms List[Tuple[int, List[float]]] to Observable[int] (loaded count)
        where tuples contain (faiss_id, vector)
    """
    
    def rebuild_faiss_step(vectors: List[Tuple[int, List[float]]]) -> Observable[int]:
        """
        Rebuild the FAISS index with the validated vectors using GPU.
        
        Flow:
        1. Create index from database vectors with their FAISS IDs
        2. Move to GPU for fast searches
        
        Returns:
            Observable[int]: Observable that emits the loaded count
        """
        try:
            logger.info(f"🔄 [REBUILD_FAISS_OP] Starting GPU rebuild with {len(vectors)} vectors...")
            
            # rebuild_index_from_vectors now returns Observable[int]
            return faiss_repository.rebuild_index_from_vectors(vectors).pipe(
                ops.map(lambda loaded_count: _log_and_return(loaded_count)),
                ops.catch(lambda e, _: _handle_rebuild_error(e))
            )
            
        except Exception as e:
            logger.error(f"❌ [REBUILD_FAISS_OP] FAISS GPU rebuild error: {e}")
            raise LoadVectorsFaissUpdateException(
                detail={
                    "reason": "FAISS index rebuild failure with GPU",
                    "faiss_error": str(e),
                    "persistence_strategy": "gpu_accelerated"
                },
                faiss_error=str(e)
            )
    
    return ops.flat_map(rebuild_faiss_step)


def _log_and_return(loaded_count: int) -> int:
    """Log the result and return the count."""
    if loaded_count > 0:
        logger.info(f"✅ [REBUILD_FAISS_OP] Successfully rebuilt FAISS index with GPU: {loaded_count} vectors")
        logger.info("🚀 [REBUILD_FAISS_OP] Index loaded to GPU for fast searches")
    else:
        logger.warning("⚠️ [REBUILD_FAISS_OP] Rebuilt index but no vectors were loaded")
    return loaded_count


def _handle_rebuild_error(e: Exception) -> Observable[int]:
    """Handle rebuild errors."""
    from reactivex import throw
    logger.error(f"❌ [REBUILD_FAISS_OP] FAISS GPU rebuild error: {e}")
    return throw(LoadVectorsFaissUpdateException(
        detail={
            "reason": "FAISS index rebuild failure with GPU",
            "faiss_error": str(e),
            "persistence_strategy": "gpu_accelerated"
        },
        faiss_error=str(e)
    ))
