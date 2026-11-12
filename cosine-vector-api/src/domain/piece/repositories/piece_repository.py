"""
Domain repository interface for Piece operations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.entities.piece import Piece


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

    @abstractmethod
    async def get_all_with_vectors(self) -> List[Piece]:
        """
        Get all pieces that have vectors in the database.
        
        Returns:
            List of pieces that have non-null vectors
        """
        pass
