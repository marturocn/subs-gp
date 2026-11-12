"""
Command for searching similar vectors.
"""
from dataclasses import dataclass
from io import BytesIO
from typing import Optional


@dataclass
class SearchVectorCommand:
    """
    Command for searching similar vectors from an image.
    """
    image_data: BytesIO
    k: Optional[int] = 5  # Number of similar results to return
    threshold: Optional[float] = None  # Minimum similarity threshold to include results
    max_results: Optional[int] = None  # Maximum number of results to return
    enable_object_detector: Optional[bool] = True  # Enable/disable object detection
