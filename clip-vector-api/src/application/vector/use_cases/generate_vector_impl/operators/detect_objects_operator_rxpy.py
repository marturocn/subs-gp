"""
RxPY Operator for detecting objects with Grounding DINO.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
from typing import Callable, Optional
from PIL import Image
from io import BytesIO
from reactivex import Observable
from reactivex.operators import flat_map
import reactivex as rx

from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.models.detection_models import DetectionResult
from src.application.vector.use_cases.generate_vector_impl.exceptions.generate_vector_exception import GenerateVectorException


class DetectObjectsOperator:
    """
    Operador para detectar objetos en una imagen usando GroundingDINO.
    """

    @staticmethod
    def execute(
        dino_repository: GroundingDinoRepository,
        image: Image.Image,
        text_prompt: Optional[str],
        logger: logging.Logger
    ) -> Observable[DetectionResult]:
        """
        Detecta objetos en la imagen usando GroundingDINO.

        :param dino_repository: Repositorio de GroundingDINO
        :param image: PIL Image a procesar
        :param text_prompt: Prompt de texto para detección
        :param logger: Logger instance
        :return: Observable[DetectionResult] con los objetos detectados
        """
        try:
            logger.info("🔍 Detectando objetos con GroundingDINO...")
            if text_prompt:
                logger.info(f"🎯 Usando text prompt: '{text_prompt}'")
            
            # Convertir PIL Image a BytesIO
            image_bytes = BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            
            # Detectar objetos (sync call wrapped in Observable)
            detection_result = dino_repository.detect_objects_sync(image_bytes, text_prompt)
            
            num_detections = detection_result.num_detections
            logger.info(f"✅ GroundingDINO detectó {num_detections} objetos")
            
            # Log detected objects
            for idx, bbox in enumerate(detection_result.bounding_boxes, 1):
                logger.info(f"   {idx}. {bbox.class_name} (confianza: {bbox.confidence:.2f})")
            
            return rx.of(detection_result)
            
        except Exception as e:
            logger.error(f"❌ Error detectando objetos: {e}")
            return rx.throw(GenerateVectorException(
                message="Error detecting objects with GroundingDINO",
                detail={"error": str(e)}
            ))


def detect_objects_operator(
    dino_repository: GroundingDinoRepository,
    text_prompt: Optional[str],
    logger: logging.Logger
) -> Callable[[Observable], Observable[DetectionResult]]:
    """
    Operator para detectar objetos con GroundingDINO.
    
    :param dino_repository: Repositorio de GroundingDINO
    :param text_prompt: Prompt de texto para detección
    :param logger: Logger instance
    :return: Observable con DetectionResult
    """
    def operator(source: Observable) -> Observable[DetectionResult]:
        return source.pipe(
            flat_map(lambda image: DetectObjectsOperator.execute(
                dino_repository, 
                image,
                text_prompt,
                logger
            ))
        )
    return operator
