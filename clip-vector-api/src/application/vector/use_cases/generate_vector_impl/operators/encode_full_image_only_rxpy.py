"""
RxPY Operator for encoding only the full image without object detection.
"""
import logging
import numpy as np
from typing import Callable
from io import BytesIO
from PIL import Image
from reactivex import Observable
from reactivex.operators import map, flat_map
import reactivex as rx

from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.application.vector.use_cases.generate_vector_impl.exceptions.vector_generation_encode_exception import VectorGenerationEncodeException


class EncodeFullImageOnlyOperator:
    """
    Operador para codificar solo la imagen completa sin detección de objetos.
    """

    @staticmethod
    def execute(
        dinov3_repository: DINOv3ModelRepository,
        image: Image.Image,
        logger: logging.Logger
    ) -> Observable[dict]:
        """
        Codifica solo la imagen completa con DINOv3.

        :param dinov3_repository: Repositorio de DINOv3
        :param image: PIL Image a procesar
        :param logger: Logger instance
        :return: Observable[dict] con vector de pieza completa
        """
        try:
            logger.info("🎨 Codificando solo imagen completa con DINOv3...")
            
            # Convertir PIL Image a BytesIO
            image_bytes = BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            
            # Codificar imagen completa
            piece_vector = dinov3_repository.encode_image_sync(image_bytes)
            
            if isinstance(piece_vector, np.ndarray):
                piece_vector = piece_vector.tolist()
            
            logger.info(f"✅ Vector de pieza generado: dimensión {len(piece_vector)}")
            
            return rx.of({
                "piece_vector": piece_vector,
                "objects": []
            })
            
        except Exception as e:
            logger.error(f"❌ Error codificando imagen: {e}")
            return rx.throw(VectorGenerationEncodeException(
                message="Failed to encode image with DINOv3",
                detail={"error": str(e)}
            ))


def encode_full_image_only_operator(
    dinov3_repository: DINOv3ModelRepository,
    logger: logging.Logger
) -> Callable[[Observable], Observable[dict]]:
    """
    Operator para codificar solo imagen completa con DINOv3.
    
    :param dinov3_repository: Repositorio de DINOv3
    :param logger: Logger instance
    :return: Observable con vector codificado
    """
    def operator(source: Observable) -> Observable[dict]:
        return source.pipe(
            flat_map(lambda image: EncodeFullImageOnlyOperator.execute(
                dinov3_repository,
                image,
                logger
            ))
        )
    return operator
