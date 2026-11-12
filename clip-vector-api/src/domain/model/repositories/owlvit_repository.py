"""
Domain repository interface for OWL-ViT v2 model operations.
"""
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional
from PIL import Image

from src.domain.model.models.detection_models import DetectionResult


class OwlvitRepository(ABC):
    """
    Abstract repository for OWL-ViT v2 model operations.
    OWL-ViT v2 is an open-vocabulary object detector that uses text prompts.
    """
    
    @abstractmethod
    async def detect_objects(
        self, 
        image_data: BytesIO, 
        text_prompt: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect objects in an image using OWL-ViT v2 model (async).
        
        Args:
            image_data: The image data as BytesIO
            text_prompt: Text prompt for detection (e.g., ["person", "car", "dog"])
                        If None, uses default prompt from config
            
        Returns:
            DetectionResult containing all detected objects
        """
        pass
    
    @abstractmethod
    def detect_objects_sync(
        self, 
        image_data: BytesIO, 
        text_prompt: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect objects in an image using OWL-ViT v2 model (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            text_prompt: Text prompt for detection (e.g., ["person", "car", "dog"])
                        If None, uses default prompt from config
            
        Returns:
            DetectionResult containing all detected objects
        """
        pass
    
    @abstractmethod
    def crop_object(
        self, 
        image: Image.Image, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int
    ) -> Image.Image:
        """
        Crop an object from an image using bounding box coordinates.
        
        Args:
            image: The source image
            x1: Top-left x coordinate
            y1: Top-left y coordinate
            x2: Bottom-right x coordinate
            y2: Bottom-right y coordinate
            
        Returns:
            Cropped image of the detected object
        """
        pass
