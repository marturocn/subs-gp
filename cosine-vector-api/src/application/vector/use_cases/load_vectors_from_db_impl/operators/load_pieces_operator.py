"""
RxPY Operator for loading piece vectors from database.
"""
import logging
import asyncio
from typing import List, Callable
from reactivex import Observable
from reactivex import operators as ops

from src.domain.vector.use_cases.load_vectors_from_db_case import LoadVectorsFromDbCommand
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.domain.piece_vector.models.entities.piece_vector import PieceVector
from src.application.vector.use_cases.load_vectors_from_db_impl.exceptions.load_vectors_database_query_exception import LoadVectorsDatabaseQueryException


logger = logging.getLogger(__name__)


def load_piece_vectors_operator(piece_vector_repository: PieceVectorRepository) -> Callable[[Observable], Observable]:
    """
    RxPY operator that loads piece vectors from database.
    Now loads from piece_vector table with faiss_id mapping.
    
    Args:
        piece_vector_repository: Repository for piece vector operations
        
    Returns:
        RxPY operator function that transforms LoadVectorsFromDbCommand to List[PieceVector]
    """
    
    def load_piece_vectors_step(command: LoadVectorsFromDbCommand) -> List[PieceVector]:
        """Load piece vectors from database (piece_vector table)."""
        try:
            logger.info("📊 [LOAD_PIECE_VECTORS_OP] Loading piece vectors from piece_vector table...")
            
            # Use a simpler approach - create a sync wrapper in a new thread
            import asyncio
            import threading
            
            piece_vectors = []
            exception = None
            
            def run_async():
                nonlocal piece_vectors, exception
                try:
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        piece_vectors = loop.run_until_complete(piece_vector_repository.get_all_vectors())
                    finally:
                        loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_async)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            
            if piece_vectors is None:
                piece_vectors = []
            
            logger.info(f"✅ [LOAD_PIECE_VECTORS_OP] Loaded {len(piece_vectors)} piece vectors from piece_vector table")
            
            # Log algunos detalles de los vectores cargados
            if piece_vectors:
                vectors_with_faiss = [pv for pv in piece_vectors if pv.faiss_id is not None]
                vectors_without_faiss = [pv for pv in piece_vectors if pv.faiss_id is None]
                logger.info(f"🔢 [LOAD_PIECE_VECTORS_OP] Vectors with faiss_id: {len(vectors_with_faiss)}")
                if vectors_without_faiss:
                    logger.warning(f"⚠️ [LOAD_PIECE_VECTORS_OP] Vectors without faiss_id: {len(vectors_without_faiss)} (will be skipped)")
            
            return piece_vectors
            
        except Exception as e:
            logger.error(f"❌ [LOAD_PIECE_VECTORS_OP] Database query error: {e}")
            raise LoadVectorsDatabaseQueryException(
                detail={"reason": "Error querying piece_vector table", "error": str(e)}
            )
    
    return ops.map(load_piece_vectors_step)
