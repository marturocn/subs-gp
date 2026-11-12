"""
Operator to detect objects using Grounding DINO.
"""
from typing import Callable, Optional, Tuple, Union
from logging import Logger
from reactivex import Observable
from PIL import Image
from io import BytesIO

from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.models.detection_models import DetectionResult
from src.application.model.use_cases.detect_objects_impl.exceptions.detection_failed_exception import DetectionFailedException


def detect_objects_operator(
    grounding_dino_repository: GroundingDinoRepository,
    text_prompt: Optional[str],
    logger: Logger
) -> Callable[[Observable], Observable]:
    """
    RxPY operator to detect objects in an image using Grounding DINO.
    
    Args:
        grounding_dino_repository: Repository for Grounding DINO model
        text_prompt: Optional text prompt for detection
        logger: Logger instance for logging
        
    Returns:
        Operator function that transforms Observable[Image.Image] to Observable[Tuple[DetectionResult, Image.Image]]
    """
    def _detect_objects(source: Observable) -> Observable:
        def subscribe(observer, scheduler=None):
            def on_next(image: Image.Image):
                try:
                    logger.info("Detecting objects in image using Grounding DINO...")
                    if text_prompt:
                        logger.info(f"Using text prompt: '{text_prompt}'")
                    else:
                        logger.info("Using default text queries from config")
                    
                    # Convert image to BytesIO for repository
                    image_buffer = BytesIO()
                    image.save(image_buffer, format='PNG')
                    image_buffer.seek(0)
                    
                    # Detect objects using Grounding DINO (synchronous)
                    detection_result: DetectionResult = grounding_dino_repository.detect_objects_sync(
                        image_data=image_buffer,
                        text_prompt=text_prompt
                    )
                    
                    logger.info(f"✅ Detected {len(detection_result.bounding_boxes)} objects")
                    
                    # Emit tuple with detection result and original image
                    observer.on_next((detection_result, image))
                    observer.on_completed()
                    
                except Exception as e:
                    logger.error(f"Failed to detect objects: {str(e)}")
                    error = DetectionFailedException(
                        message=f"Failed to detect objects: {str(e)}",
                        original_exception=e
                    )
                    observer.on_error(error)
            
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
    
    return _detect_objects
