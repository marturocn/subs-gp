"""
Exception for image loading errors.
"""
from typing import Optional
from .detect_objects_exception import DetectObjectsException


class ImageLoadException(DetectObjectsException):
    """
    Exception raised when there is an error loading the image.
    """
    
    def __init__(self, message: str = "Failed to load image", original_exception: Optional[Exception] = None):
        super().__init__(message, original_exception)
