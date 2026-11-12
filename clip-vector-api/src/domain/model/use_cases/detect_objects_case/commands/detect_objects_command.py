"""
Command for detecting objects in an image.
"""
from dataclasses import dataclass
from io import BytesIO
from typing import Optional


@dataclass
class DetectObjectsCommand:
    """
    Command for detecting objects in an image using Grounding DINO.
    
    Args:
        image_data: The image data as BytesIO
        text_prompt: Optional text prompt for detection (e.g., "person . car . dog").
                    If None, uses default prompt from environment config.
    """
    image_data: BytesIO
    text_prompt: Optional[str] = None
