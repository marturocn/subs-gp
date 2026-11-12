"""
Command for searching vectors in the system.
"""
from dataclasses import dataclass
from typing import List, Optional
from io import BytesIO


@dataclass
class SearchVectorCommand:
    """
    Command for searching similar vectors in the system.
    This coordinates encoding the query image and finding similar vectors.
    """
    image_data: BytesIO
    k: Optional[int] = 5
    threshold: Optional[float] = 0.5
    max_results: Optional[int] = None
    enable_object_detector: Optional[bool] = True


@dataclass
class SearchVectorResponse:
    """
    Response after successfully searching vectors.
    """
    results: List[dict]
