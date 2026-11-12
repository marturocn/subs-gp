"""
Domain repository interface for PieceVector operations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ..models.entities.piece_vector import PieceVector


class PieceVectorRepository(ABC):
    """
    Abstract repository for PieceVector operations.
    """
    
    @abstractmethod
    async def save(self, piece_vector: PieceVector) -> PieceVector:
        """
        Save or update a piece vector.
        
        Args:
            piece_vector: The piece vector to save
            
        Returns:
            The saved piece vector with updated fields
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, vector_id: str) -> Optional[PieceVector]:
        """
        Get a piece vector by its ID.
        
        Args:
            vector_id: The unique identifier of the piece vector
            
        Returns:
            The piece vector if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_ids(self, faiss_ids: List[str]) -> List[Tuple[str, str, str]]:
        """
        Get multiple piece vectors by their FAISS IDs using a single query with JOIN to piece table.
        
        Args:
            faiss_ids: List of FAISS IDs to retrieve (from FAISS index results)
            
        Returns:
            List of tuples (faiss_id, piece_id, catalog_unique_id)
        """
        pass
    
    @abstractmethod
    def get_by_ids_sync(self, faiss_ids: List[str]) -> List[Tuple[str, str, str]]:
        """
        Get multiple piece vectors by their FAISS IDs (synchronous version).
        
        Args:
            faiss_ids: List of FAISS IDs to retrieve
            
        Returns:
            List of tuples (faiss_id, piece_id, catalog_unique_id)
        """
        pass
    
    @abstractmethod
    async def get_by_piece_id(self, piece_id: str) -> List[PieceVector]:
        """
        Get all vectors for a specific piece.
        
        Args:
            piece_id: The unique identifier of the piece
            
        Returns:
            List of piece vectors for the given piece
        """
        pass
    
    @abstractmethod
    async def get_all_vectors(self) -> List[PieceVector]:
        """
        Get all piece vectors from the database.
        
        Returns:
            List of all piece vectors
        """
        pass
    
    @abstractmethod
    async def delete_by_piece_id(self, piece_id: str) -> None:
        """
        Delete all vectors for a specific piece.
        
        Args:
            piece_id: The unique identifier of the piece
        """
        pass
    
    @abstractmethod
    async def update_faiss_id(self, vector_id: str, faiss_id: int) -> None:
        """
        Update the FAISS ID for a piece vector.
        
        Args:
            vector_id: The unique identifier of the piece vector
            faiss_id: The new FAISS index ID
        """
        pass
