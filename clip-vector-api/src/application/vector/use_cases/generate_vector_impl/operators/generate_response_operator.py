"""
RxPY Operator for generating vector response.
Logs (at INFO level) show a reduced preview of vectors with status and meta info.
"""
import logging
from typing import Dict, Any, Callable
from reactivex import Observable
from reactivex import operators as ops

from src.domain.vector.use_cases.generate_vector_case import GenerateVectorResponse
from src.application.vector.use_cases.generate_vector_impl.exceptions.generate_vector_exception import GenerateVectorException

logger = logging.getLogger(__name__)


def generate_response_operator() -> Callable[[Observable], Observable]:
    """
    RxPY operator that creates GenerateVectorResponse from encoded data.
    Logs include a reduced version of the vectors with extra metadata.
    """

    def _shorten_vector(vector, max_len: int = 5):
        """Return a short preview of a vector for logging purposes."""
        if not isinstance(vector, list):
            return []
        if len(vector) > max_len:
            return vector[:max_len] + ["..."]
        return vector

    def create_response_step(encoded_data: Dict[str, Any]) -> GenerateVectorResponse:
        """Create response with the generated vectors for piece and objects."""
        try:
            logger.info("🚀 Iniciando creación de GenerateVectorResponse...")
            logger.info(f"📥 Keys recibidas: {list(encoded_data.keys())}")

            piece_vector = encoded_data.get("piece_vector", [])
            objects = encoded_data.get("objects", [])

            logger.info(f"🔹 Dimensión del vector principal: {len(piece_vector)}")
            logger.info(f"🔹 Total de objetos detectados: {len(objects)}")

            # Mostrar solo una parte del vector en logs
            logger.info(f"🔸 Vista previa del vector principal: {_shorten_vector(piece_vector)}")

            # Remover información extra de los objetos para la respuesta
            formatted_objects = []
            for i, obj in enumerate(objects):
                formatted_obj = {
                    "type": obj["type"],
                    "vector": obj["vector"],
                    "confidence": obj["confidence"],
                    "crop_index": obj["crop_index"]
                }
                formatted_objects.append(formatted_obj)

                logger.info(
                    f"🧩 Objeto {i}: type={obj['type']}, "
                    f"vector_preview={_shorten_vector(obj['vector'])}, "
                    f"confidence={obj['confidence']}, crop_index={obj['crop_index']}"
                )

            logger.info(
                f"✅ Respuesta generada correctamente → "
                f"vector_pieza_dim={len(piece_vector)}, total_objetos={len(formatted_objects)}"
            )

            # 🧠 Crear vista reducida para logging
            reduced_response_preview = {
                "status": 200,
                "meta": {},
                "piece_vector": _shorten_vector(piece_vector),
                "objects": [
                    {
                        "type": o["type"],
                        "vector": _shorten_vector(o["vector"]),
                        "confidence": o["confidence"],
                        "crop_index": o["crop_index"]
                    }
                    for o in formatted_objects
                ]
            }
            import json
            # Log del preview con formato JSON legible
            logger.info("📦 Vista reducida de la respuesta generada (solo para log):")
            logger.info(json.dumps(reduced_response_preview, indent=2))

            # 🔸 La response se devuelve sin modificaciones
            return GenerateVectorResponse(
                piece_vector=piece_vector,
                objects=formatted_objects
            )

        except Exception as e:
            logger.exception("💥 Error al crear la respuesta de vectores.")
            raise GenerateVectorException(
                message="Error creating generate vector response",
                detail={"error": str(e), "input_keys": list(encoded_data.keys())}
            ) from e

    return ops.map(create_response_step)
