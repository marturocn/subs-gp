"""
Vector application module.
"""
import asyncio
from dishka import Provider, provide, Scope

from common.core.observable_resolver import resolve_observable_to_single_result
from domain.vector.use_cases.load_vectors_from_db_case.commands.load_vectors_from_db_command import LoadVectorsFromDbCommand
from src.domain.vector.use_cases.load_vectors_from_db_case import LoadVectorsFromDbUseCase, LoadVectorsFromDbResponse
from src.application.vector.use_cases.load_vectors_from_db_impl.load_vectors_from_db_use_case_impl import LoadVectorsFromDbUseCaseImpl
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.infraestructure import logger


class LoadVectorsFromDbProvider(Provider):
    """
    Provider for Vector application layer dependencies.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_load_vectors_from_db_use_case(
        self,
        faiss_index_repository: FaissIndexRepository,
        piece_vector_repository: PieceVectorRepository,
    ) -> LoadVectorsFromDbUseCase:
        """Provide load vectors from database use case implementation."""
        use_case = LoadVectorsFromDbUseCaseImpl(
            faiss_index_repository=faiss_index_repository,
            piece_vector_repository=piece_vector_repository,
        )
        return use_case
    
    @provide(scope=Scope.APP)
    async def load_vectors_on_startup(
        self,
        faiss_index_repository: FaissIndexRepository,
        piece_vector_repository: PieceVectorRepository,
    ) -> LoadVectorsFromDbResponse:
        """
        Load vectors from database on application startup.
        This method is executed once when the application starts.
        """
        logger.info("🔄 [STARTUP] Loading vectors from database on application startup...")
        try:
            result = await load_vectors_from_db_handler(
                faiss_index_repository=faiss_index_repository,
                piece_vector_repository=piece_vector_repository,
            )
            logger.info(f"✅ [STARTUP] Vectors loaded successfully: {result.data.get('loaded_count', 0)} vectors")
            return result
        except Exception as e:
            logger.error(f"⚠️  [STARTUP] Could not load vectors on startup: {str(e)}")
            logger.error(f"⚠️  [STARTUP] Application will continue without pre-loaded vectors")
            raise


async def load_vectors_from_db_handler(
    faiss_index_repository: FaissIndexRepository,
    piece_vector_repository: PieceVectorRepository,
) -> LoadVectorsFromDbResponse:
    """
    Handler to execute load vectors from database use case.
    
    Args:
        faiss_index_repository: FAISS index repository
        piece_vector_repository: Piece vector repository
        
    Returns:
        LoadVectorsFromDbResponse: Response with load status
    """
    logger.info("📥 [LOAD_VECTORS_HANDLER] Initiating load vectors from database...")
    
    try:
        # Create use case instance
        use_case = LoadVectorsFromDbUseCaseImpl(
            faiss_index_repository=faiss_index_repository,
            piece_vector_repository=piece_vector_repository,
        )
        
        # Create and execute command
        command = LoadVectorsFromDbCommand()
        result = await resolve_observable_to_single_result(use_case.execute(command))
        
        logger.info("✅ [LOAD_VECTORS_HANDLER] Successfully loaded vectors from database")
        return result
        
    except Exception as e:
        logger.error(f"❌ [LOAD_VECTORS_HANDLER] Error loading vectors: {str(e)}")
        raise

__all__ = [
    "LoadVectorsFromDbUseCaseImpl",
    "LoadVectorsFromDbProvider",
    "load_vectors_from_db_handler",
]
