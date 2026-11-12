"""
Domain repository interface for FAISS index operations.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple


class FaissIndexRepository(ABC):
    """
    Abstract repository for FAISS index operations.
    """
    
    @abstractmethod
    async def add_vector(self, piece_id: str, vector: List[float]) -> None:
        """
        Add a vector to the FAISS index.
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The vector to add to the index
        """
        pass
    
    @abstractmethod
    def add_vector_sync(self, piece_id: str, vector: List[float]) -> None:
        """
        Add a vector to the FAISS index (synchronous).
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The vector to add to the index
        """
        pass
    
    @abstractmethod
    def add_vector_with_id_sync(self, faiss_id: int, vector: List[float]) -> None:
        """
        Add a vector to the FAISS index with a specific FAISS ID (synchronous).
        Uses IndexIDMap to map the vector to the given ID.
        
        Args:
            faiss_id: The specific FAISS ID to assign to this vector
            vector: The vector to add to the index
        """
        pass
    
    @abstractmethod
    async def update_vector(self, piece_id: str, vector: List[float]) -> None:
        """
        Update a vector in the FAISS index.
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The new vector representation
        """
        pass
    
    @abstractmethod
    async def remove_vector(self, piece_id: str) -> None:
        """
        Remove a vector from the FAISS index.
        
        Args:
            piece_id: The unique identifier of the piece to remove
        """
        pass
    
    @abstractmethod
    async def search_similar(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_vector: The query vector to search for
            k: Number of similar vectors to return
            
        Returns:
            List of tuples (piece_id, similarity_score)
        """
        pass
