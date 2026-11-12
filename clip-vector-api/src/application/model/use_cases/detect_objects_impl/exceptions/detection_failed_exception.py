"""
Exception for detection failures.
"""
from typing import Optional
from .detect_objects_exception import DetectObjectsException


class DetectionFailedException(DetectObjectsException):
    """
    Exception raised when object detection fails.
    """
    
    def __init__(self, message: str = "Object detection failed", original_exception: Optional[Exception] = None):
        super().__init__(message, original_exception)
