"""
Use case for adding vectors to FAISS index.
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from .commands import AddVectorToIndexCommand
from src.common.core.use_case import UseCase


class AddVectorToIndexUseCase(UseCase[AddVectorToIndexCommand, None], ABC):
    """
    Abstract use case for adding a vector to the FAISS index.
    """
    
    @abstractmethod
    def execute(self, command: AddVectorToIndexCommand) -> Observable[None]:
        """
        Execute the use case to add a vector to the index.
        
        Args:
            command: Command with the piece ID and vector to add
            
        Returns:
            Observable that completes when the vector is added
        """
        pass
