"""
Load Vectors From Database Use Case - Domain Interface
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from .commands import LoadVectorsFromDbCommand
from .responses import LoadVectorsFromDbResponse
from src.common.core.use_case import UseCase


class LoadVectorsFromDbUseCase(UseCase[LoadVectorsFromDbCommand, LoadVectorsFromDbResponse], ABC):
    """
    Abstract use case for loading vectors from database to FAISS index.
    This rebuilds the entire FAISS index from vectors stored in the database.
    """
    
    @abstractmethod
    def execute(self, command: LoadVectorsFromDbCommand) -> Observable[LoadVectorsFromDbResponse]:
        """
        Execute the load vectors from database use case.
        
        Args:
            command: Command to load vectors from database
            
        Returns:
            Observable that emits the response when complete
        """
        pass
