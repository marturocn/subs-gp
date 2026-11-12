"""
Command for generating a vector from an image without saving.
"""
from dataclasses import dataclass
from io import BytesIO


@dataclass
class GenerateVectorCommand:
    """
    Command for generating a vector from an image.
    This only generates the vector without storing it.
    """
    image_data: BytesIO
    enable_object_detector: bool = True
