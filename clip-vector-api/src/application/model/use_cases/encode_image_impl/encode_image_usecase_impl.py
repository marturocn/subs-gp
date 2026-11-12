"""
Encode Image use case implementation.
"""
import reactivex
from reactivex import Observable
from typing import List

from application.model.use_cases.encode_image_impl.exceptions.load_model_exception import LoadModelException
from src.domain.model.use_cases.encode_image_case import EncodeImageUseCase, EncodeImageCommand
from src.domain.model.repositories.clip_model_repository import ClipModelRepository


class EncodeImageUseCaseImpl(EncodeImageUseCase):
    """
    Implementation of EncodeImageUseCase.
    """
    
    def __init__(self, repository: ClipModelRepository):
        """
        Initialize the use case with a CLIP model repository.
        
        Args:
            repository: CLIP model repository for image encoding
        """
        self.repository = repository
    
    def execute(self, command: EncodeImageCommand) -> Observable[List[float]]:
        """
        Execute the encode image use case.
        
        Args:
            command: Command with image data to encode
            
        Returns:
            Observable[List[float]]: Observable that emits the vector representation
        """
        try:
            # Encode image synchronously to avoid event loop issues
            vector = self.repository.encode_image_sync(command.image_data)
            return reactivex.of(vector)
        except Exception as e:
            raise LoadModelException(detail={"error": str(e)})
