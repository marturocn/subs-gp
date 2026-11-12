"""
RxPY Operator for encoding objects with DINOv3.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
import numpy as np
from typing import Callable, List, Dict, Any
from io import BytesIO
from PIL import Image
from reactivex import Observable
from reactivex.operators import map, flat_map, catch
import reactivex as rx

from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.models.detection_models import DetectionResult
from src.application.vector.use_cases.add_vector_impl.exceptions.encode_image_exception import EncodeImageException


class EncodeObjectsOperator:
    """
    Operador para codificar objetos detectados usando DINOv3.
    Genera embeddings para cada objeto y los combina en uno solo.
    """

    @staticmethod
    def execute(
        dinov3_repository: DINOv3ModelRepository,
        dino_repository: GroundingDinoRepository,
        detection_result: DetectionResult,
        original_image: Image.Image,
        logger: logging.Logger
    ) -> Observable[Dict[str, Any]]:
        """
        Codifica cada objeto detectado con DINOv3 y combina los vectores.
        TAMBIÉN codifica la imagen original completa sin recortes.

        :param dinov3_repository: Repositorio de DINOv3
        :param dino_repository: Repositorio de GroundingDINO (para cropping)
        :param detection_result: Resultado de detección de objetos
        :param original_image: Imagen original PIL
        :param logger: Logger instance
        :return: Observable[Dict] con vector principal, vector original y objetos
        """
        try:
            num_detections = detection_result.num_detections
            
            # PASO 0: Codificar imagen ORIGINAL completa (sin recortes)
            logger.info("🎨 Codificando imagen original completa (sin recortes)...")
            original_image_bytes = BytesIO()
            original_image.save(original_image_bytes, format='PNG')
            original_image_bytes.seek(0)
            original_full_embedding = dinov3_repository.encode_image_sync(original_image_bytes)
            
            if isinstance(original_full_embedding, np.ndarray):
                original_full_embedding = original_full_embedding.tolist()
            
            logger.info(f"✅ Vector original generado: dimensión {len(original_full_embedding)}")
            
            # Si no hay detecciones, usar el vector original como principal también
            if num_detections == 0:
                logger.warning("⚠️  No se detectaron objetos")
                logger.info("   Usando vector original como vector principal")
                
                return rx.of({
                    "original_embedding": original_full_embedding,
                    "main_embedding": original_full_embedding,
                    "objects": [],
                    "num_objects": 0
                })
            
            # Procesar cada objeto detectado
            logger.info(f"🎯 Procesando {num_detections} objetos detectados...")
            
            object_embeddings = []
            object_metadata = []
            
            for idx, bbox in enumerate(detection_result.bounding_boxes, 1):
                try:
                    logger.info(f"   🖼️  Procesando objeto {idx}/{num_detections}: {bbox.class_name} (confianza: {bbox.confidence:.2f})")
                    
                    # Crop object from image
                    cropped_image = dino_repository.crop_object(
                        original_image,
                        bbox.x1,
                        bbox.y1,
                        bbox.x2,
                        bbox.y2
                    )
                    
                    # Encode con DINOv3 (sync method)
                    obj_embedding = dinov3_repository.encode_cropped_image_sync(cropped_image)
                    
                    if obj_embedding is None or len(obj_embedding) == 0:
                        logger.warning(f"      ⚠️  Error codificando objeto {idx}, saltando...")
                        continue
                    
                    # Convertir a lista si es numpy array
                    if isinstance(obj_embedding, np.ndarray):
                        obj_embedding = obj_embedding.tolist()
                    
                    object_embeddings.append(obj_embedding)
                    object_metadata.append({
                        "index": idx - 1,
                        "label": bbox.class_name,
                        "confidence": float(bbox.confidence),
                        "box": [int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)],
                        "embedding": obj_embedding
                    })
                    
                    logger.info(f"      ✅ Objeto {idx} codificado: dimensión {len(obj_embedding)}")
                    
                except Exception as e:
                    logger.error(f"      ❌ Error procesando objeto {idx}: {e}")
                    continue
            
            # Validar resultados
            if len(object_embeddings) == 0:
                logger.error("❌ No se pudo codificar ningún objeto detectado")
                raise EncodeImageException(
                    message="Failed to encode any detected objects",
                    detail={"num_detected": num_detections, "num_encoded": 0}
                )
            
            logger.info(f"🎉 Exitosamente codificados {len(object_embeddings)}/{num_detections} objetos")
            
            # Combinar vectores en uno solo (weighted average)
            confidences = [obj["confidence"] for obj in object_metadata]
            main_embedding = _combine_vectors_weighted(object_embeddings, confidences, logger)
            
            logger.info(f"✅ Vector principal combinado: dimensión {len(main_embedding)}")
            
            return rx.of({
                "original_embedding": original_full_embedding,  # Imagen original SIN recortes
                "main_embedding": main_embedding,  # Vector combinado de objetos
                "objects": object_metadata,
                "num_objects": len(object_embeddings)
            })
            
        except EncodeImageException:
            raise
        except Exception as e:
            logger.error(f"❌ Error en pipeline de codificación: {e}")
            return rx.throw(EncodeImageException(
                message="Failed to encode objects with DINOv3",
                detail={"error": str(e)}
            ))


def _combine_vectors_weighted(vectors: List[List[float]], confidences: List[float], logger: logging.Logger) -> List[float]:
    """
    Combina múltiples vectores en uno usando promedio ponderado basado en confianza.
    
    Formula: combined_vector = sum(vector_i * confidence_i) / sum(confidence_i)
    
    :param vectors: Lista de vectores a combinar
    :param confidences: Lista de scores de confianza
    :param logger: Logger instance
    :return: Vector combinado con normalización L2
    """
    if len(vectors) == 0:
        raise ValueError("Cannot combine empty vector list")
    
    if len(vectors) != len(confidences):
        raise ValueError(f"Vectors and confidences length mismatch: {len(vectors)} vs {len(confidences)}")
    
    # Caso especial: un solo vector
    if len(vectors) == 1:
        logger.info("📊 Un solo objeto detectado, usando su vector directamente")
        return vectors[0]
    
    # Convertir a numpy
    vectors_array = np.array(vectors, dtype=np.float32)  # Shape: (N, dim)
    confidences_array = np.array(confidences, dtype=np.float32)  # Shape: (N,)
    
    # Log estadísticas
    logger.info(f"📊 Combinando {len(vectors)} vectores:")
    logger.info(f"   - Rango de confianza: [{confidences_array.min():.3f}, {confidences_array.max():.3f}]")
    logger.info(f"   - Confianza promedio: {confidences_array.mean():.3f}")
    
    # Promedio ponderado: sum(v_i * w_i) / sum(w_i)
    confidence_sum = confidences_array.sum()
    
    if confidence_sum == 0:
        logger.warning("⚠️  Todas las confianzas son cero, usando promedio simple")
        combined = vectors_array.mean(axis=0)
    else:
        # Reshape para broadcasting: (N,) -> (N, 1)
        weights = (confidences_array / confidence_sum).reshape(-1, 1)
        combined = (vectors_array * weights).sum(axis=0)
    
    # Normalización L2
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
        logger.info(f"   ✅ Normalización L2 aplicada (norma: {norm:.6f})")
    else:
        logger.warning("   ⚠️  Vector combinado tiene norma cero, saltando normalización")
    
    # Convertir a lista
    combined_list = combined.tolist()
    
    logger.info(f"✅ Combinación de vectores completa: {len(vectors)} vectores → 1 vector combinado")
    
    return combined_list


def encode_objects_with_dinov3_operator(
    dinov3_repository: DINOv3ModelRepository,
    dino_repository: GroundingDinoRepository,
    logger: logging.Logger
) -> Callable[[Observable], Observable[Dict[str, Any]]]:
    """
    Operator para codificar objetos detectados con DINOv3.

    :param dinov3_repository: Repositorio de DINOv3
    :param dino_repository: Repositorio de GroundingDINO
    :param logger: Logger instance
    :return: Observable con vectores codificados
    """
    def operator(source: Observable) -> Observable[Dict[str, Any]]:
        return source.pipe(
            flat_map(lambda data: EncodeObjectsOperator.execute(
                dinov3_repository,
                dino_repository,
                data["detection_result"],
                data["original_image"],
                logger
            ))
        )
    return operator
