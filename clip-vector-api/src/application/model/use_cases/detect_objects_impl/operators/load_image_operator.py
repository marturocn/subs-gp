"""
Operator to load image from BytesIO.
"""
from typing import Callable
from logging import Logger
from reactivex import Observable
from PIL import Image

from src.domain.model.use_cases.detect_objects_case import DetectObjectsCommand
from src.application.model.use_cases.detect_objects_impl.exceptions.image_load_exception import ImageLoadException


def load_image_operator(logger: Logger) -> Callable[[Observable], Observable]:
    """
    RxPY operator to load a PIL Image from BytesIO.
    
    Args:
        logger: Logger instance for logging
        
    Returns:
        Operator function that transforms Observable[DetectObjectsCommand] to Observable[Image.Image]
    """
    def _load_image(source: Observable) -> Observable:
        def subscribe(observer, scheduler=None):
            def on_next(command: DetectObjectsCommand):
                try:
                    logger.info("Loading image from BytesIO...")
                    
                    # Reset stream position
                    command.image_data.seek(0)
                    
                    # Load image using PIL
                    image = Image.open(command.image_data)
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        logger.info(f"Converting image from {image.mode} to RGB")
                        image = image.convert('RGB')
                    
                    logger.info(f"✅ Image loaded successfully: {image.size[0]}x{image.size[1]} pixels")
                    observer.on_next(image)
                    observer.on_completed()
                    
                except Exception as e:
                    logger.error(f"Failed to load image: {str(e)}")
                    error = ImageLoadException(
                        message=f"Failed to load image: {str(e)}",
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
    
    return _load_image
