"""
Add Vector to Index use case implementation.
"""
import reactivex
from reactivex import Observable

from application.faiss.use_cases.add_vector_to_index_impl.exceptions.faiss_index_exception import FaissIndexException
from src.domain.faiss.use_cases.add_vector_to_index_case import AddVectorToIndexUseCase, AddVectorToIndexCommand
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository


class AddVectorToIndexUseCaseImpl(AddVectorToIndexUseCase):
    """
    Implementation of AddVectorToIndexUseCase.
    """
    
    def __init__(self, repository: FaissIndexRepository):
        """
        Initialize the use case with a FAISS index repository.
        
        Args:
            repository: FAISS index repository for vector operations
        """
        self.repository = repository
    
    def execute(self, command: AddVectorToIndexCommand) -> Observable[None]:
        """
        Execute the add vector to index use case.
        
        Args:
            command: Command with piece ID and vector data
            
        Returns:
            Observable[None]: Observable that completes when vector is added
        """
        try:
            # Add vector synchronously to avoid event loop issues
            self.repository.add_vector_sync(command.piece_id, command.vector)
            return reactivex.of(None)
        except Exception as e:
            raise FaissIndexException(detail={"error": str(e)})
