"""
Search Vector Use Case Implementation - Application Layer
"""
from reactivex import Observable, of
from reactivex import operators as ops
from typing import List, Tuple

from src.domain.vector.use_cases.search_vector_case import SearchVectorUseCase, SearchVectorCommand, SearchVectorResponse
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.application.vector.use_cases.search_vector_impl.operators.encode_image_operator import encode_image_operator
from src.application.vector.use_cases.search_vector_impl.operators.search_vectors_operator import search_vectors_operator
from src.application.vector.use_cases.search_vector_impl.operators.simple_cosine_search_operator import simple_cosine_search_operator
from src.application.vector.use_cases.search_vector_impl.operators.format_results_operator import format_results_operator
from src.application.vector.use_cases.search_vector_impl.exceptions.search_vector_exception import SearchVectorException
from src.application.vector.use_cases.search_vector_impl.exceptions.search_vector_encode_exception import SearchVectorEncodeException
from src.application.vector.use_cases.search_vector_impl.exceptions.cosine_similarity_search_exception import CosineSimilaritySearchException
from src.application.vector.use_cases.search_vector_impl.exceptions.search_vector_format_exception import SearchVectorFormatException
from src.infraestructure import logger


class SearchVectorUseCaseImpl(SearchVectorUseCase):
    """
    Implementation of the Search Vector use case using RxPY observables pipeline.
    """
    
    def __init__(
        self,
        faiss_index_repository: FaissIndexRepository,
        piece_vector_repository: PieceVectorRepository,
    ):
        self._faiss_index_repository = faiss_index_repository
        self._piece_vector_repository = piece_vector_repository
    
    def execute(self, command: SearchVectorCommand) -> Observable[SearchVectorResponse]:
        """
        Execute the search vector use case using RxPY pipeline with custom operators.
        
        Args:
            command: Command with search data
            
        Returns:
            Observable[SearchVectorResponse]: Observable that emits the search response
        """
        enable_object_detector = command.enable_object_detector if command.enable_object_detector is not None else True
        
        logger.info("🚀 [SEARCH_VECTOR_USE_CASE] Starting search execution")
        logger.info(f"🔍 [SEARCH_VECTOR_USE_CASE] Command: threshold={command.threshold}, max_results={command.max_results}")
        logger.info(f"🎯 [SEARCH_VECTOR_USE_CASE] enableObjectDetector: {enable_object_detector}")
        logger.info(f"🔍 [SEARCH_VECTOR_USE_CASE] FAISS repository type: {type(self._faiss_index_repository).__name__}")
        
        # Debug FAISS repository state before starting
        if enable_object_detector:
            self._debug_faiss_repository()
        
        # Choose the appropriate search operator based on enable_object_detector
        if enable_object_detector:
            logger.info("🎯 [SEARCH_VECTOR_USE_CASE] Using search_vectors_operator (with object detection)")
            # Clean RxPY pipeline using search_vectors_operator (piece_item_vector table)
            return (
                of(command)
                .pipe(
                    encode_image_operator(),     # Step 1: Encode image to vector
                    search_vectors_operator(
                        self._faiss_index_repository,
                        self._piece_vector_repository,
                        command.k,
                        command.threshold,
                        command.max_results
                    ),  # Step 2: Search similar vectors with object detection
                    format_results_operator(),                              # Step 3: Format results
                    ops.catch(self._handle_error)                          # Error handling
                )
            )
        else:
            logger.info("🎯 [SEARCH_VECTOR_USE_CASE] Using simple_cosine_search_operator (without object detection)")
            
            # Calculate appropriate k for FAISS search
            # Since each piece can have multiple angles (e.g., 5 angles per piece),
            # we need to retrieve more candidates to ensure diversity
            # Strategy: k = max_results * 10 (to get ~10 angles per desired result)
            max_results = command.max_results or 10
            calculated_k = max_results * 10  # Get 10x more candidates for diversity
            
            logger.info(f"📊 [SEARCH_VECTOR_USE_CASE] Calculated k={calculated_k} (max_results={max_results} * 10)")
            
            # Clean RxPY pipeline using simple_cosine_search_operator (piece_vector table only)
            return (
                of(command)
                .pipe(
                    encode_image_operator(),     # Step 1: Encode image to vector
                    simple_cosine_search_operator(
                        self._faiss_index_repository,
                        self._piece_vector_repository,
                        calculated_k,
                        command.threshold or 0.35,
                        command.max_results
                    ),  # Step 2: Simple cosine similarity search with FAISS + piece_vector filter
                    format_results_operator(),                              # Step 3: Format results
                    ops.catch(self._handle_error)                          # Error handling
                )
            )
    
    def _debug_faiss_repository(self) -> None:
        """Debug FAISS repository state for troubleshooting."""
        try:
            logger.info("🔍 [DEBUG_FAISS] Checking FAISS repository state...")
            
            def _subscribe_sync(observable):
                """Helper to subscribe to Observable and get value synchronously."""
                result = []
                error_holder = []
                
                def on_next(value):
                    result.append(value)
                
                def on_error(e):
                    error_holder.append(e)
                
                observable.subscribe(on_next=on_next, on_error=on_error)
                
                if error_holder:
                    raise error_holder[0]
                
                return result[0] if result else []
            
            # Check if repository has get_all_vectors method
            if hasattr(self._faiss_index_repository, 'get_all_vectors'):
                all_vectors_obs = self._faiss_index_repository.get_all_vectors()
                all_vectors = _subscribe_sync(all_vectors_obs)
                logger.info(f"🔍 [DEBUG_FAISS] Total vectors in repository: {len(all_vectors)}")
                
                if all_vectors:
                    first_piece_id, first_vector = all_vectors[0]
                    logger.info(f"🔍 [DEBUG_FAISS] First vector: piece_id={first_piece_id}, dim={len(first_vector)}")
                    
                    # Show first few piece IDs
                    piece_ids = [piece_id for piece_id, _ in all_vectors[:3]]
                    logger.info(f"🔍 [DEBUG_FAISS] Sample piece IDs: {piece_ids}")
                else:
                    logger.warning("⚠️ [DEBUG_FAISS] No vectors found in repository")
            else:
                logger.warning("⚠️ [DEBUG_FAISS] get_all_vectors method not available")
            
        except Exception as e:
            logger.error(f"❌ [DEBUG_FAISS] Error debugging FAISS repository: {e}")
    
    def _handle_error(self, error: Exception, source: Observable) -> Observable[SearchVectorResponse]:
        logger.error(f"Unexpected error in SearchVectorUseCase: {error}")
        """Handle and transform errors appropriately."""
        if isinstance(error, (SearchVectorEncodeException, CosineSimilaritySearchException, SearchVectorFormatException)):
            raise error
        
        raise SearchVectorException(
            message=f"Unexpected error in SearchVectorUseCase: {str(error)}",
            detail={"error": str(error)}
        )
