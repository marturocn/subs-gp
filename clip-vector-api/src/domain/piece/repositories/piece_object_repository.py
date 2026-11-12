"""
Piece Object Repository Interface - Domain Layer
"""
from abc import ABC, abstractmethod
from reactivex import Observable


class PieceObjectRepository(ABC):
    """
    Repository interface for PieceObject operations.
    Provides reactive patterns for database operations.
    """

    @abstractmethod
    def create_query_builder(self) -> Observable:
        """
        Create a reactive query builder for PieceObject operations.
        
        Returns:
            Observable: Observable that emits a query builder for PieceObject
        """
        pass
