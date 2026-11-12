"""
DTO for object detection request
"""
from typing import Optional
from pydantic import BaseModel


class DetectObjectsDto(BaseModel):
    """
    DTO for object detection request.
    
    Attributes:
        text_prompt: Optional text prompt for Grounding DINO detection.
                    If not provided, uses default from environment config.
                    Example: "person . car . dog . cat"
    """
    text_prompt: Optional[str] = None
