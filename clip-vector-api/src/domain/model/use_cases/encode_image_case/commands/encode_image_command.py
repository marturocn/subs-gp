"""
Command for encoding an image to vector.
"""
from dataclasses import dataclass
from io import BytesIO


@dataclass
class EncodeImageCommand:
    """
    Command for encoding an image to vector.
    """
    image_data: BytesIO
