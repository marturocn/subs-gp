"""
Domain repository interface for FAISS index operations.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

from reactivex import Observable


class FaissIndexRepository(ABC):
    """
    Abstract repository for FAISS index operations using FAISS-GPU with IndexIDMap2.
    """
    
    @abstractmethod
    def add_vector(self, piece_id: str, vector: List[float]) -> Observable[None]:
        """
        Add a vector to the FAISS index.
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The vector to add to the index
        """
        pass
    
    @abstractmethod
    def add_vector_sync(self, piece_id: str, vector: List[float]) -> Observable[None]:
        """
        Add a vector to the FAISS index (synchronous).
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The vector to add to the index
        """
        pass
    
    @abstractmethod
    async def update_vector(self, piece_id: str, vector: List[float]) -> Observable[None]:
        """
        Update a vector in the FAISS index.
        
        Args:
            piece_id: The unique identifier of the piece
            vector: The new vector representation
        """
        pass
    
    @abstractmethod
    def remove_vector(self, piece_id: str) -> Observable[None]:
        """
        Remove a vector from the FAISS index.
        
        Args:
            piece_id: The unique identifier of the piece to remove
        """
        pass
    
    @abstractmethod
    def search_similar(self, query_vector: List[float], k: int = 5) -> Observable[List[Tuple[str, float]]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_vector: The query vector to search for
            k: Number of similar vectors to return
            
        Returns:
            List of tuples (piece_id, similarity_score)
        """
        pass
    
    @abstractmethod
    def search_similar_sync(self, query_vector: List[float], k: int = 5) -> Observable[List[Tuple[str, float]]]:
        """
        Search for similar vectors in the index (synchronous).
        
        Args:
            query_vector: The query vector to search for
            k: Number of similar vectors to return
            
        Returns:
            List of tuples (piece_id, similarity_score)
        """
        pass
    
    @abstractmethod
    def get_all_vectors(self) -> Observable[List[Tuple[str, List[float]]]]:
        """
        Get all vectors from the index.
        
        Returns:
            List of tuples (piece_id, vector)
        """
        pass
    
    @abstractmethod
    def rebuild_index_from_vectors(self, vectors: List[Tuple[int, List[float]]]) -> Observable[int]:
        """
        Rebuild the entire FAISS index from a list of vectors.
        Uses FAISS-GPU with IndexIDMap2 for efficient ID mapping.
        
        Args:
            vectors: List of tuples (faiss_id, vector) where faiss_id is the FAISS index ID
            
        Returns:
            Number of vectors successfully added to the index
        """
        pass
    
    @abstractmethod
    def get_index_status(self) -> Observable[Dict[str, Any]]:
        """
        Get the current status of the FAISS index.
        
        Returns:
            Dictionary with index status information including:
            - total_vectors: Number of vectors in the index
            - vector_dimension: Dimension of vectors
            - gpu_enabled: Whether GPU is being used
        """
        pass
    
    @abstractmethod
    def get_vector_by_id(self, faiss_id: str) -> Observable[List[float]]:
        """
        Get a specific vector by its FAISS ID using reconstruct.
        
        Args:
            faiss_id: The FAISS index ID of the vector
            
        Returns:
            The vector as a list of floats
        """
        pass
