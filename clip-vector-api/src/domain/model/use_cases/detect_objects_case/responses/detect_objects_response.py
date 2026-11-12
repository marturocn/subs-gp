"""
Response for object detection.
"""
from typing import List, Dict, Any
from common import ServerResponseModel


class DetectObjectsResponse(ServerResponseModel[dict, dict]):
    """
    Response after successfully detecting objects in an image.
    Contains a list of detected objects with their bounding box coordinates.
    """
    def __init__(
        self, 
        objects: List[Dict[str, Any]], 
        image_width: int, 
        image_height: int,
        text_prompt: str,
        detection_count: int
    ):
        """
        Args:
            objects: List of detected objects with label, confidence, and coordinates
            image_width: Original image width in pixels
            image_height: Original image height in pixels
            text_prompt: Text prompt used for detection
            detection_count: Number of objects detected
        """
        data = {
            "detections": {
                "objects": objects,
                "count": detection_count,
                "text_prompt": text_prompt,
                "image_dimensions": {
                    "width": image_width,
                    "height": image_height
                }
            }
        }
        super().__init__(data=data)
