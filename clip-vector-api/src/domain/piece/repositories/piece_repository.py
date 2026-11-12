"""
Domain repository interface for Piece operations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from reactivex import Observable

from src.infraestructure.persistance.piece.models.piece_model import PieceModel
from ..models.entities.piece import Piece
from sqlalchemy.orm import Query

class PieceRepository(ABC):
    """
    Abstract repository for Piece operations.
    """
    
    @abstractmethod
    async def save_vector(self, piece_id: str, vector: List[float]) -> None:
        """
        Save or update vector for a piece.
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The vector representation of the image
        """
        pass

    @abstractmethod
    def create_query_builder(self) -> Observable[Query[PieceModel]]:
        """
        Create a query builder for complex queries.

        Returns:
            Observable emitting a SQLAlchemy Select query.
        """
        pass
    
    @abstractmethod
    def save_vector_sync(self, piece_id: str, vector: List[float]) -> None:
        """
        Save or update vector for a piece (synchronous).
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The vector representation of the image
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, piece_id: str) -> Optional[Piece]:
        """
        Get a piece by its ID.
        
        Args:
            piece_id: The unique identifier of the piece
            
        Returns:
            The piece if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def exists(self, piece_id: str) -> bool:
        """
        Check if a piece exists.
        
        Args:
            piece_id: The unique identifier of the piece
            
        Returns:
            True if the piece exists, False otherwise
        """
        pass
