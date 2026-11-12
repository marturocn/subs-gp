"""
Load Vectors From Database Use Case Implementation - Application Layer
"""
from reactivex import Observable, of
from reactivex import operators as ops

from src.domain.vector.use_cases.load_vectors_from_db_case import LoadVectorsFromDbUseCase, LoadVectorsFromDbCommand, LoadVectorsFromDbResponse
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.application.vector.use_cases.load_vectors_from_db_impl.operators.load_pieces_operator import load_piece_vectors_operator
from src.application.vector.use_cases.load_vectors_from_db_impl.operators.validate_vectors_operator import validate_vectors_operator
from src.application.vector.use_cases.load_vectors_from_db_impl.operators.rebuild_faiss_operator import rebuild_faiss_operator
from src.application.vector.use_cases.load_vectors_from_db_impl.operators.format_load_response_operator import format_load_response_operator
from src.application.vector.use_cases.load_vectors_from_db_impl.exceptions.load_vectors_db_exception import LoadVectorsDbException
from src.application.vector.use_cases.load_vectors_from_db_impl.exceptions.load_vectors_database_query_exception import LoadVectorsDatabaseQueryException
from src.application.vector.use_cases.load_vectors_from_db_impl.exceptions.load_vectors_invalid_format_exception import LoadVectorsInvalidFormatException
from src.application.vector.use_cases.load_vectors_from_db_impl.exceptions.load_vectors_faiss_update_exception import LoadVectorsFaissUpdateException
from src.infraestructure import logger


class LoadVectorsFromDbUseCaseImpl(LoadVectorsFromDbUseCase):
    """
    Implementation of the Load Vectors From Database use case using RxPY observables pipeline.
    Now uses the piece_vector table instead of piece.vector column.
    """
    
    def __init__(
        self,
        faiss_index_repository: FaissIndexRepository,
        piece_vector_repository: PieceVectorRepository,
    ):
        self._faiss_index_repository = faiss_index_repository
        self._piece_vector_repository = piece_vector_repository
    
    def execute(self, command: LoadVectorsFromDbCommand) -> Observable[LoadVectorsFromDbResponse]:
        """
        Execute the load vectors from database use case using RxPY pipeline with custom operators.
        
        Pipeline:
        1. Load all piece vectors from piece_vector table (with faiss_id, piece_id, vector)
        2. Validate vectors format and dimension
        3. Rebuild FAISS index with vectors:
           - Maps FAISS index (0,1,2...) -> faiss_id (from piece_vector.faiss_id)
           - faiss_id can be used to query piece_vector table and JOIN with piece table
           - This allows searches to return catalog_unique_id via the mapping chain:
             FAISS index -> faiss_id -> piece_vector -> piece -> catalog_unique_id
        4. Format response
        
        Args:
            command: Command to load vectors from database
            
        Returns:
            Observable[LoadVectorsFromDbResponse]: Observable that emits the load response
        """
        logger.info("🚀 [LOAD_VECTORS_FROM_DB_USE_CASE] Starting load vectors from database execution")
        logger.info(f"🔧 [LOAD_VECTORS_FROM_DB_USE_CASE] FAISS repository type: {type(self._faiss_index_repository).__name__}")
        logger.info(f"🔍 [LOAD_VECTORS_FROM_DB_USE_CASE] PieceVector repository type: {type(self._piece_vector_repository).__name__}")
        
        # Create command observable
        command_obs = of(command)
        
        # Clean RxPY pipeline using custom operators
        return command_obs.pipe(
            load_piece_vectors_operator(self._piece_vector_repository),
            validate_vectors_operator(),
            rebuild_faiss_operator(self._faiss_index_repository),
            format_load_response_operator()
        )
    
    def _handle_error(self, error: Exception, source: Observable) -> Observable[LoadVectorsFromDbResponse]:
        """Handle and transform errors appropriately."""
        if isinstance(error, (LoadVectorsDatabaseQueryException, LoadVectorsInvalidFormatException, LoadVectorsFaissUpdateException)):
            raise error
        
        logger.error(f"❌ [LOAD_VECTORS_FROM_DB_USE_CASE] Unexpected error: {error}")
        raise LoadVectorsDbException(
            message=f"Unexpected error in LoadVectorsFromDbUseCase: {str(error)}",
            detail={"error": str(error)}
        )
