"""
Generate Vector Use Case - Domain Interface
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from .commands import GenerateVectorCommand
from .responses import GenerateVectorResponse
from src.common.core.use_case import UseCase


class GenerateVectorUseCase(UseCase[GenerateVectorCommand, GenerateVectorResponse], ABC):
    """
    Abstract use case for generating vectors from images.
    This only generates the vector without storing it anywhere.
    """
    
    @abstractmethod
    def execute(self, command: GenerateVectorCommand) -> Observable[GenerateVectorResponse]:
        """
        Execute the generate vector use case.
        
        Args:
            command: Command with image data
            
        Returns:
            Observable that emits the response when complete
        """
        pass
