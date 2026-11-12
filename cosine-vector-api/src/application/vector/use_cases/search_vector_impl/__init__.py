"""
Search Vector Use Case Provider - Application Layer
"""
from dishka import Provider, provide, Scope

from src.domain.vector.use_cases.search_vector_case import SearchVectorUseCase
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from .search_vector_use_case_impl import SearchVectorUseCaseImpl


class SearchVectorProvider(Provider):
    """
    Dishka provider for SearchVector use case dependencies.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_search_vector_use_case(
        self, 
        faiss_index_repository: FaissIndexRepository,
        piece_vector_repository: PieceVectorRepository
    ) -> SearchVectorUseCase:
        """Proporciona el caso de uso para buscar vectores similares."""
        return SearchVectorUseCaseImpl(faiss_index_repository, piece_vector_repository)


__all__ = [
    "SearchVectorUseCaseImpl",
    "SearchVectorProvider",
]
