"""
Search Vector Use Case - Domain Interface
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from .commands import SearchVectorCommand
from .responses import SearchVectorResponse
from src.common.core.use_case import UseCase


class SearchVectorUseCase(UseCase[SearchVectorCommand, SearchVectorResponse], ABC):
    """
    Abstract use case for searching similar vectors from images.
    This generates a vector from the image and searches for similar ones in FAISS.
    """
    
    @abstractmethod
    def execute(self, command: SearchVectorCommand) -> Observable[SearchVectorResponse]:
        """
        Execute the search vector use case.
        
        Args:
            command: Command with image data and search parameters
            
        Returns:
            Observable that emits the response when complete
        """
        pass
