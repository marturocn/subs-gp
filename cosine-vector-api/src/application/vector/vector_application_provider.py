"""
Vector application module.
"""
from dishka import Provider, provide, Scope

from src.domain.vector.use_cases.add_vector_usecase import AddVectorUseCase
from src.application.vector.use_cases.add_vector_impl.add_vector_use_case_impl import AddVectorUseCaseImpl
from src.domain.piece.repositories.piece_repository import PieceRepository
from src.domain.model.repositories.clip_model_repository import ClipModelRepository
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository


class VectorApplicationProvider(Provider):
    """
    Provider for Vector application layer dependencies.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_add_vector_use_case(
        self,
        piece_repository: PieceRepository,
        clip_model_repository: ClipModelRepository,
        faiss_index_repository: FaissIndexRepository,
    ) -> AddVectorUseCase:
        """Provide add vector use case implementation."""
        return AddVectorUseCaseImpl(
            piece_repository=piece_repository,
            clip_model_repository=clip_model_repository,
            faiss_index_repository=faiss_index_repository,
        )
