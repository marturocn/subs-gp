"""
Base exception for Detect Objects use case.
"""
from typing import Optional


class DetectObjectsException(Exception):
    """
    Base exception for errors during object detection process.
    """
    
    def __init__(
        self, 
        message: str = "Error during object detection",
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)
    
    def __str__(self):
        if self.original_exception:
            return f"{self.message}. Original error: {str(self.original_exception)}"
        return self.message
