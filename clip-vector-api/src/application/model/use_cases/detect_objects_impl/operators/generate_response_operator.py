"""
Operator to generate the final response with detected objects.
"""
from typing import Callable, Tuple, Optional
from logging import Logger
from reactivex import Observable
from PIL import Image

from src.domain.model.models.detection_models import DetectionResult
from src.domain.model.use_cases.detect_objects_case import DetectObjectsResponse


def generate_response_operator(
    text_prompt: Optional[str],
    logger: Logger
) -> Callable[[Observable], Observable]:
    """
    RxPY operator to generate the final response with detected objects.
    
    Args:
        text_prompt: Text prompt used for detection (for response metadata)
        logger: Logger instance for logging
        
    Returns:
        Operator function that transforms Observable[Tuple[DetectionResult, Image.Image]] to Observable[DetectObjectsResponse]
    """
    def _generate_response(source: Observable) -> Observable:
        def subscribe(observer, scheduler=None):
            def on_next(data: Tuple[DetectionResult, Image.Image]):
                try:
                    detection_result, image = data
                    
                    logger.info("Generating response with detected objects...")
                    
                    # Get image dimensions
                    image_width, image_height = image.size
                    
                    # Format detected objects
                    objects = []
                    for bbox in detection_result.bounding_boxes:
                        obj = {
                            "label": bbox.class_name,
                            "confidence": round(bbox.confidence, 4),
                            "bounding_box": {
                                "x1": bbox.x1,
                                "y1": bbox.y1,
                                "x2": bbox.x2,
                                "y2": bbox.y2,
                                "width": bbox.width,
                                "height": bbox.height
                            },
                            "center": {
                                "x": bbox.center[0],
                                "y": bbox.center[1]
                            },
                            "area": bbox.area
                        }
                        objects.append(obj)
                    
                    # Use the provided text prompt
                    used_prompt = text_prompt or "default"
                    
                    # Create response
                    response = DetectObjectsResponse(
                        objects=objects,
                        image_width=image_width,
                        image_height=image_height,
                        text_prompt=used_prompt,
                        detection_count=len(objects)
                    )
                    
                    logger.info(f"✅ Response generated with {len(objects)} detected objects")
                    
                    observer.on_next(response)
                    observer.on_completed()
                    
                except Exception as e:
                    logger.error(f"Failed to generate response: {str(e)}")
                    observer.on_error(e)
            
            def on_error(e):
                observer.on_error(e)
            
            def on_completed():
                observer.on_completed()
            
            return source.subscribe(
                on_next=on_next,
                on_error=on_error,
                on_completed=on_completed,
                scheduler=scheduler
            )
        
        return Observable(subscribe)
    
    return _generate_response
