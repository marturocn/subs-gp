"""
Domain models for object detection.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BoundingBox:
    """
    Represents a bounding box for a detected object.
    
    Attributes:
        x1: Top-left x coordinate
        y1: Top-left y coordinate
        x2: Bottom-right x coordinate
        y2: Bottom-right y coordinate
        confidence: Detection confidence (0.0 - 1.0)
        class_id: Class ID of the detected object
        class_name: Class name of the detected object
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def width(self) -> int:
        """Get bounding box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get bounding box height."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Get bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center coordinates."""
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": self.center,
            "area": self.area,
        }


@dataclass
class DetectionResult:
    """
    Result of object detection on an image.
    
    Attributes:
        bounding_boxes: List of detected bounding boxes
        image_width: Original image width
        image_height: Original image height
        inference_time: Time taken for inference (ms)
    """
    bounding_boxes: List[BoundingBox]
    image_width: int
    image_height: int
    inference_time: float
    
    @property
    def num_detections(self) -> int:
        """Get number of detections."""
        return len(self.bounding_boxes)
    
    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence threshold."""
        filtered_boxes = [
            box for box in self.bounding_boxes
            if box.confidence >= threshold
        ]
        return DetectionResult(
            bounding_boxes=filtered_boxes,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time=self.inference_time,
        )
    
    def filter_by_class(self, class_names: List[str]) -> "DetectionResult":
        """Filter detections by class names."""
        filtered_boxes = [
            box for box in self.bounding_boxes
            if box.class_name in class_names
        ]
        return DetectionResult(
            bounding_boxes=filtered_boxes,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time=self.inference_time,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_detections": self.num_detections,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time_ms": self.inference_time,
            "detections": [box.to_dict() for box in self.bounding_boxes],
        }
