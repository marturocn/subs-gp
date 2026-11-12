"""
Generate Vector Use Case Implementation - Application Layer
"""
from reactivex import Observable, of
from reactivex import operators as ops
from typing import Optional, Tuple
from PIL import Image

from src.application.vector.use_cases.generate_vector_impl.operators.encode_full_image_only_rxpy import encode_full_image_only_operator
from src.application.vector.use_cases.generate_vector_impl.operators.load_image_operator_rxpy import load_image_operator
from src.application.vector.use_cases.generate_vector_impl.operators.detect_objects_operator_rxpy import detect_objects_operator
from src.application.vector.use_cases.generate_vector_impl.operators.encode_full_image_and_objects_operator_rxpy import encode_full_image_and_objects_operator
from domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.vector.use_cases.generate_vector_case import GenerateVectorUseCase, GenerateVectorCommand, GenerateVectorResponse
from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.application.vector.use_cases.generate_vector_impl.operators.generate_response_operator import generate_response_operator
from src.application.vector.use_cases.generate_vector_impl.exceptions.generate_vector_exception import GenerateVectorException
from src.application.vector.use_cases.generate_vector_impl.exceptions.vector_generation_encode_exception import VectorGenerationEncodeException
from src.domain.model.models.detection_models import DetectionResult
from src.infraestructure import logger


class GenerateVectorUseCaseImpl(GenerateVectorUseCase):
    """
    Implementation of the Generate Vector use case using RxPY observables pipeline.
    Detects objects in the image and generates vectors for the full image and each detected object.
    Uses DINOv3 for encoding images and objects.
    """
    
    def __init__(
        self,
        dinov3_model_repository: DINOv3ModelRepository,
        dino_repository: GroundingDinoRepository,
    ):
        self._dinov3_model_repository = dinov3_model_repository
        self._dino_repository = dino_repository

    def execute(self, command: GenerateVectorCommand) -> Observable[GenerateVectorResponse]:
        """
        Execute the generate vector use case using RxPY pipeline with custom operators.
        Pipeline:
        1. Load image from BytesIO
        2. If enable_object_detector is True: Detect objects and encode both image and objects
        3. If enable_object_detector is False: Only encode the full image
        4. Create response with piece vector and object vectors
        
        Args:
            command: Command with vector generation data
            
        Returns:
            Observable[GenerateVectorResponse]: Observable that emits the response
        """
        # Clean RxPY pipeline using custom operators
        return (
            of(command)
            .pipe(
                # Step 1: Load image from BytesIO
                load_image_operator(logger),
                # Step 2: Choose pipeline based on enable_object_detector
                ops.flat_map(lambda image: self._choose_pipeline(image, command.enable_object_detector)),
                # Step 3: Create response
                generate_response_operator(),
                # Error handling
                ops.catch(self._handle_error)
            )
        )
    
    def _choose_pipeline(self, image: Image.Image, enable_object_detector: bool) -> Observable[dict]:
        """
        Choose which pipeline to use based on enable_object_detector flag.
        
        Args:
            image: PIL Image to process
            enable_object_detector: Whether to detect objects or not
            
        Returns:
            Observable with encoded data
        """
        if enable_object_detector:
            # Use existing pipeline with object detection
            return (
                of((image, image))
                .pipe(
                    ops.flat_map(lambda data: self._detect_and_combine(data[0], data[1])),
                    encode_full_image_and_objects_operator(
                        self._dinov3_model_repository,
                        self._dino_repository,
                        logger
                    )
                )
            )
        else:
            # Use new pipeline without object detection
            return (
                of(image)
                .pipe(
                    encode_full_image_only_operator(
                        self._dinov3_model_repository,
                        logger
                    )
                )
            )

    
    def _detect_and_combine(self, image: Image.Image, image_for_detection: Image.Image) -> Observable[Tuple[Image.Image, DetectionResult]]:
        """
        Detect objects and combine with original image.
        
        Args:
            image: Original PIL Image to preserve
            image_for_detection: PIL Image to use for detection
            
        Returns:
            Observable with tuple (Image, DetectionResult)
        """
        return (
            of(image_for_detection)
            .pipe(
                # Detect objects
                detect_objects_operator(
                    self._dino_repository,
                    text_prompt=None,  # No text prompt for generic detection
                    logger=logger
                ),
                # Combine with original image
                ops.map(lambda detection: (image, detection))
            )
        )
    
    def _handle_error(self, error: Exception, source: Observable) -> Observable[GenerateVectorResponse]:
        """Handle and transform errors appropriately."""
        if isinstance(error, VectorGenerationEncodeException):
            raise error
        
        # Transform encode errors from add_vector_impl to generate_vector_impl specific errors
        if hasattr(error, 'detail'):
            logger.error(f"Error in GenerateVectorUseCase: {error}")
            raise VectorGenerationEncodeException(
                message="Error encoding image for vector generation",
                detail=getattr(error, 'detail', {"error": str(error)})
            )
        
        logger.error(f"Unexpected error in GenerateVectorUseCase: {error}")
        raise GenerateVectorException(
            message=f"Unexpected error in GenerateVectorUseCase: {str(error)}",
            detail={"error": str(error)}
        )
