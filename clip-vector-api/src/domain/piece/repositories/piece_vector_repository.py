"""
Domain repository interface for PieceVector operations.
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from src.infraestructure.persistance.piece.models.piece_vector_model import PieceVectorModel
from ..models.entities.piece_vector import PieceVector
from sqlalchemy.orm import Query


class PieceVectorRepository(ABC):
    """
    Abstract repository for PieceVector operations.
    """
    

    @abstractmethod
    def create_query_builder(self) -> Observable[Query[PieceVectorModel]]:
        """
        Create a query builder for complex queries.

        Returns:
            Observable emitting a SQLAlchemy Select query.
        """
        pass
