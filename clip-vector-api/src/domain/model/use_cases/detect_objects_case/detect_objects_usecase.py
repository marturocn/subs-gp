"""
Detect Objects Use Case - Domain Interface
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from .commands import DetectObjectsCommand
from .responses import DetectObjectsResponse
from src.common.core.use_case import UseCase


class DetectObjectsUseCase(UseCase[DetectObjectsCommand, DetectObjectsResponse], ABC):
    """
    Abstract use case for detecting objects in images using Grounding DINO.
    Provides Google Lens-like functionality.
    """
    
    @abstractmethod
    def execute(self, command: DetectObjectsCommand) -> Observable[DetectObjectsResponse]:
        """
        Execute the detect objects use case.
        
        Args:
            command: Command with image data and optional text prompt
            
        Returns:
            Observable that emits the response with detected objects
        """
        pass
