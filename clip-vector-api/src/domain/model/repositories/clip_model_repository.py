"""
Domain repository interface for CLIP model operations.
"""
from abc import ABC, abstractmethod
from typing import List
from io import BytesIO


class ClipModelRepository(ABC):
    """
    Abstract repository for CLIP model operations.
    """
    
    @abstractmethod
    async def encode_image(self, image_data: BytesIO) -> List[float]:
        """
        Convert an image to a vector using CLIP model.
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            The vector representation of the image
        """
        pass
    
    @abstractmethod
    def encode_image_sync(self, image_data: BytesIO) -> List[float]:
        """
        Convert an image to a vector using CLIP model (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            The vector representation of the image
        """
        pass
