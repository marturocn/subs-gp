"""
Use case for encoding images with CLIP model.
"""
from abc import ABC, abstractmethod
from typing import List

from reactivex import Observable

from .commands import EncodeImageCommand
from src.common.core.use_case import UseCase


class EncodeImageUseCase(UseCase[EncodeImageCommand, List[float]], ABC):
    """
    Abstract use case for encoding an image to vector using CLIP model.
    """
    
    @abstractmethod
    def execute(self, command: EncodeImageCommand) -> Observable[List[float]]:
        """
        Execute the use case to encode an image to vector.
        
        Args:
            command: Command with the image data to encode
            
        Returns:
            Observable that emits the vector representation of the image
        """
        pass
