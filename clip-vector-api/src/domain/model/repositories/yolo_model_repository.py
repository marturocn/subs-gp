"""
Domain repository interface for YOLO model operations.
"""
from abc import ABC, abstractmethod
from io import BytesIO
from PIL import Image

from src.domain.model.models.detection_models import DetectionResult


class YoloModelRepository(ABC):
    """
    Abstract repository for YOLO model operations.
    """
    
    @abstractmethod
    async def detect_objects(self, image_data: BytesIO) -> DetectionResult:
        """
        Detect objects in an image using YOLO model (async).
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            DetectionResult containing all detected objects
        """
        pass
    
    @abstractmethod
    def detect_objects_sync(self, image_data: BytesIO) -> DetectionResult:
        """
        Detect objects in an image using YOLO model (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            DetectionResult containing all detected objects
        """
        pass
    
    @abstractmethod
    def crop_object(self, image: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
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
