"""
Piece persistence module.
"""
from dishka import Provider, provide, Scope

from src.domain.piece.repositories.piece_repository import PieceRepository
from src.infraestructure.persistance.piece.repositories.piece_repository_impl import PieceRepositoryImpl
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.infraestructure.persistance.piece.repositories.piece_vector_repository_impl import PieceVectorRepositoryImpl


class PiecePersistenceProvider(Provider):
    """
    Provider for Piece persistence dependencies.
    """
    
    @provide(scope=Scope.APP)
    def get_piece_repository(self) -> PieceRepository:
        """Provide piece repository implementation."""
        return PieceRepositoryImpl()
    
    @provide(scope=Scope.APP)
    def get_piece_vector_repository(self) -> PieceVectorRepository:
        """Provide piece vector repository implementation."""
        return PieceVectorRepositoryImpl()
