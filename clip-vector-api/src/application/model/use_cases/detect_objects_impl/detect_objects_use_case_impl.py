"""
Detect Objects Use Case Implementation - Application Layer
"""
from reactivex import Observable, of
from reactivex import operators as ops
from PIL import Image
from typing import Tuple

from src.domain.model.use_cases.detect_objects_case import (
    DetectObjectsUseCase, 
    DetectObjectsCommand, 
    DetectObjectsResponse
)
from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.application.model.use_cases.detect_objects_impl.operators.load_image_operator import load_image_operator
from src.application.model.use_cases.detect_objects_impl.operators.detect_objects_operator import detect_objects_operator
from src.application.model.use_cases.detect_objects_impl.operators.generate_response_operator import generate_response_operator
from src.application.model.use_cases.detect_objects_impl.exceptions.detect_objects_exception import DetectObjectsException
from src.infraestructure import logger


class DetectObjectsUseCaseImpl(DetectObjectsUseCase):
    """
    Implementation of the Detect Objects use case using RxPY observables pipeline.
    Detects objects in images using Grounding DINO model.
    Provides Google Lens-like functionality with bounding box coordinates.
    """
    
    def __init__(
        self,
        grounding_dino_repository: GroundingDinoRepository,
    ):
        self._grounding_dino_repository = grounding_dino_repository

    def execute(self, command: DetectObjectsCommand) -> Observable[DetectObjectsResponse]:
        """
        Execute the detect objects use case using RxPY pipeline with custom operators.
        Pipeline:
        1. Load image from BytesIO
        2. Detect objects using Grounding DINO
        3. Create response with detected objects and their coordinates
        
        Args:
            command: Command with image data and optional text prompt
            
        Returns:
            Observable[DetectObjectsResponse]: Observable that emits the response
        """
        return (
            of(command)
            .pipe(
                # Step 1: Load image from BytesIO
                load_image_operator(logger),
                # Step 2: Detect objects using Grounding DINO
                detect_objects_operator(
                    self._grounding_dino_repository,
                    command.text_prompt,
                    logger
                ),
                # Step 3: Create response
                generate_response_operator(command.text_prompt, logger),
                # Error handling
                ops.catch(self._handle_error)
            )
        )
    
    def _handle_error(self, error: Exception, source: Observable) -> Observable:
        """
        Handle errors that occur during object detection.
        
        Args:
            error: The exception that occurred
            source: The source observable
            
        Returns:
            Observable that emits the error
        """
        logger.error(f"Error in DetectObjectsUseCaseImpl: {str(error)}")
        
        if isinstance(error, DetectObjectsException):
            raise error
        
        # Wrap unknown errors
        wrapped_error = DetectObjectsException(
            message=f"Unexpected error during object detection: {str(error)}",
            original_exception=error
        )
        raise wrapped_error
