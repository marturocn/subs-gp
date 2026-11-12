"""
Add Vector Use Case - Domain Interface
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from .commands import AddVectorCommand
from .responses import AddVectorResponse
from src.common.core.use_case import UseCase


class AddVectorUseCase(UseCase[AddVectorCommand, AddVectorResponse], ABC):
    """
    Abstract coordinator use case for adding vectors to the system.
    This orchestrates the encoding, storage, and indexing process.
    """
    
    @abstractmethod
    def execute(self, command: AddVectorCommand) -> Observable[AddVectorResponse]:
        """
        Execute the add vector use case.
        
        Args:
            command: Command with piece ID and image data
            
        Returns:
            Observable that emits the response when complete
        """
        pass
