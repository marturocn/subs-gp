"""
RxPY Operator for encoding full image and detected objects with DINOv3.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
import numpy as np
from typing import Callable, List, Dict, Any, Tuple
from io import BytesIO
from PIL import Image
from reactivex import Observable
from reactivex.operators import flat_map
import reactivex as rx

from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.models.detection_models import DetectionResult
from src.application.vector.use_cases.generate_vector_impl.exceptions.vector_generation_encode_exception import VectorGenerationEncodeException


class EncodeFullImageAndObjectsOperator:
    """
    Operador para codificar la imagen completa y cada objeto detectado con DINOv3.
    """

    @staticmethod
    def _encode_full_image(image: Image.Image, dinov3_repository: DINOv3ModelRepository, logger: logging.Logger) -> list:
        """Encode the full image (piece) with DINOv3."""
        logger.info("🎨 Codificando imagen completa (pieza) con DINOv3...")
        full_image_bytes = BytesIO()
        image.save(full_image_bytes, format='PNG')
        full_image_bytes.seek(0)
        piece_vector = dinov3_repository.encode_image_sync(full_image_bytes)
        
        if isinstance(piece_vector, np.ndarray):
            piece_vector = piece_vector.tolist()
        
        logger.info(f"✅ Vector de pieza generado: dimensión {len(piece_vector)}")
        return piece_vector

    @staticmethod
    def _prepare_cropped_images(detection_result: DetectionResult, image: Image.Image, dino_repository: GroundingDinoRepository, logger: logging.Logger) -> Tuple[list, list]:
        """Prepare all cropped images and metadata."""
        cropped_images = []
        bbox_info = []
        
        for idx, bbox in enumerate(detection_result.bounding_boxes, 1):
            try:
                cropped_image = dino_repository.crop_object(image, bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                cropped_images.append(cropped_image)
                bbox_info.append({
                    "idx": idx,
                    "class_name": bbox.class_name,
                    "confidence": float(bbox.confidence),
                    "box": [int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)]
                })
            except Exception as e:
                logger.error(f"      ❌ Error cropping objeto {idx}: {e}")
                continue
        
        return cropped_images, bbox_info

    @staticmethod
    def _process_batch_encoding(cropped_images: list, bbox_info: list, dinov3_repository: DINOv3ModelRepository, logger: logging.Logger) -> list:
        """Process all cropped images in batch."""
        logger.info(f"   📦 Procesando {len(cropped_images)} crops en batch...")
        object_vectors = dinov3_repository.encode_images_batch(cropped_images)
        
        encoded_objects = []
        for info, vector in zip(bbox_info, object_vectors):
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            encoded_objects.append({
                "type": info["class_name"],
                "vector": vector,
                "confidence": info["confidence"],
                "crop_index": info["idx"],
                "box": info["box"]
            })
            
            logger.info(f"      ✅ Objeto {info['idx']} ({info['class_name']}) codificado: dimensión {len(vector)}")
        
        return encoded_objects

    @staticmethod
    def _process_fallback_encoding(cropped_images: list, bbox_info: list, dinov3_repository: DINOv3ModelRepository, logger: logging.Logger) -> list:
        """Fallback: process images one by one when batch fails."""
        logger.warning("      ⚠️  Fallback a procesamiento individual...")
        encoded_objects = []
        
        for idx, (cropped_image, info) in enumerate(zip(cropped_images, bbox_info), 1):
            try:
                object_vector = dinov3_repository.encode_cropped_image_sync(cropped_image)
                
                if object_vector is None or len(object_vector) == 0:
                    logger.warning(f"      ⚠️  Error codificando objeto {idx}, saltando...")
                    continue
                
                if isinstance(object_vector, np.ndarray):
                    object_vector = object_vector.tolist()
                
                encoded_objects.append({
                    "type": info["class_name"],
                    "vector": object_vector,
                    "confidence": info["confidence"],
                    "crop_index": info["idx"],
                    "box": info["box"]
                })
                
                logger.info(f"      ✅ Objeto {idx} codificado: dimensión {len(object_vector)}")
                
            except Exception as e2:
                logger.error(f"      ❌ Error procesando objeto {idx}: {e2}")
                continue
        
        return encoded_objects

    @staticmethod
    def _encode_all_objects(cropped_images: list, bbox_info: list, dinov3_repository: DINOv3ModelRepository, logger: logging.Logger) -> list:
        """Encode all objects with batch processing and fallback."""
        if len(cropped_images) == 0:
            return []
        
        try:
            return EncodeFullImageAndObjectsOperator._process_batch_encoding(cropped_images, bbox_info, dinov3_repository, logger)
        except Exception as e:
            logger.error(f"      ❌ Error en batch processing: {e}")
            return EncodeFullImageAndObjectsOperator._process_fallback_encoding(cropped_images, bbox_info, dinov3_repository, logger)

    @staticmethod
    def execute(
        dinov3_repository: DINOv3ModelRepository,
        dino_repository: GroundingDinoRepository,
        image_and_detection: Tuple[Image.Image, DetectionResult],
        logger: logging.Logger
    ) -> Observable[Dict[str, Any]]:
        """
        Codifica la imagen completa y cada objeto detectado con DINOv3.

        :param dinov3_repository: Repositorio de DINOv3
        :param dino_repository: Repositorio de GroundingDINO (para cropping)
        :param image_and_detection: Tupla con (PIL Image, DetectionResult)
        :param logger: Logger instance
        :return: Observable[Dict] con vector de pieza completa y objetos
        """
        try:
            image, detection_result = image_and_detection
            
            # PASO 1: Codificar imagen completa (pieza)
            piece_vector = EncodeFullImageAndObjectsOperator._encode_full_image(image, dinov3_repository, logger)
            
            # Si no hay detecciones, retornar solo el vector de la pieza
            if detection_result.num_detections == 0:
                logger.warning("⚠️  No se detectaron objetos")
                return rx.of({"piece_vector": piece_vector, "objects": []})
            
            # PASO 2: Codificar cada objeto detectado
            logger.info(f"🎯 Codificando {detection_result.num_detections} objetos detectados con DINOv3...")
            
            cropped_images, bbox_info = EncodeFullImageAndObjectsOperator._prepare_cropped_images(detection_result, image, dino_repository, logger)
            encoded_objects = EncodeFullImageAndObjectsOperator._encode_all_objects(cropped_images, bbox_info, dinov3_repository, logger)
            
            logger.info(f"🎉 Exitosamente codificados {len(encoded_objects)}/{detection_result.num_detections} objetos")
            
            return rx.of({"piece_vector": piece_vector, "objects": encoded_objects})
            
        except Exception as e:
            logger.error(f"❌ Error en pipeline de codificación: {e}")
            return rx.throw(VectorGenerationEncodeException(
                message="Failed to encode image and objects with DINOv3",
                detail={"error": str(e)}
            ))


def encode_full_image_and_objects_operator(
    dinov3_repository: DINOv3ModelRepository,
    dino_repository: GroundingDinoRepository,
    logger: logging.Logger
) -> Callable[[Observable], Observable[Dict[str, Any]]]:
    """
    Operator para codificar imagen completa y objetos detectados con DINOv3.
    
    :param dinov3_repository: Repositorio de DINOv3
    :param dino_repository: Repositorio de GroundingDINO
    :param logger: Logger instance
    :return: Observable con vectores codificados
    """
    def operator(source: Observable) -> Observable[Dict[str, Any]]:
        return source.pipe(
            flat_map(lambda data: EncodeFullImageAndObjectsOperator.execute(
                dinov3_repository,
                dino_repository,
                data,
                logger
            ))
        )
    return operator
