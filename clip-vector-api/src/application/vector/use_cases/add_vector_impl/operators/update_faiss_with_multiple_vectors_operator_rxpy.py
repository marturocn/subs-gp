"""
RxPY Operator for updating FAISS index with multiple vectors.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import base64
import json
import logging
import os
from typing import Callable, Dict, Any, Optional, Tuple
import httpcore
import httpx
from reactivex import Observable
from reactivex.operators import map, flat_map, catch
import reactivex as rx

from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.application.vector.use_cases.add_vector_impl.exceptions.update_faiss_index_exception import UpdateFaissIndexException


class UpdateFaissWithMultipleVectorsOperator:
    """
    Operador para actualizar el índice FAISS con múltiples vectores.
    """

    @staticmethod
    def execute(
        save_result: Dict[str, Any],
        logger: logging.Logger
    ) -> Observable[Dict[str, Any]]:
        """
        Actualiza el índice FAISS con el vector principal y los vectores de objetos.

        :param faiss_repository: Repositorio FAISS
        :param save_result: Resultado del guardado en BD con faiss_ids
        :param logger: Logger instance
        :return: Observable[Dict] con resultado de actualización
        """
        try:
            piece_id = save_result["piece_id"]
            saved_piece_vector = save_result.get("saved_piece_vector", {})
            saved_objects = save_result.get("saved_objects", [])
            num_objects_saved = save_result.get("num_objects_saved", 0)
            shared_faiss_id = save_result.get("shared_faiss_id")
            
            if save_result.get("is_duplicate"):
                logger.info(f"   ℹ️ El vector ya existe en BD e índice para piece_id={piece_id}. Se omite actualización redundante de FAISS.")
                return rx.of({
                    "piece_id": piece_id,
                    "num_vectors_added": 0,
                    "shared_faiss_id": shared_faiss_id,
                    "success": True,
                    "saved_piece_vector": saved_piece_vector,
                    "saved_objects": [],
                    "num_objects_saved": 0
                })

            # Total de vectores = 1 (imagen completa) + objetos detectados
            total_vectors = 1 + num_objects_saved
            
            logger.info(f"🔍 Actualizando índice FAISS con {total_vectors} vectores...")
            logger.info(f"   - 1 vector de imagen completa (faiss_id={shared_faiss_id})")
            logger.info(f"   - {num_objects_saved} vectores de objetos detectados")
            
            UpdateFaissWithMultipleVectorsOperator.update_step(
                logger,
                piece_id
            )

            # Log del vector de imagen completa
            if saved_piece_vector:
                logger.info(f"   ✅ Vector de imagen agregado a FAISS: faiss_id={saved_piece_vector.get('faiss_id')}")
            
            # Log de objetos
            for obj in saved_objects:
                faiss_id = obj.get("faiss_id")
                object_name = obj.get("object_name")
                confidence = obj.get("confidence", 0)
                logger.info(f"   ✅ Objeto agregado a FAISS: faiss_id={faiss_id}, name={object_name}, conf={confidence:.2%}")
            

            logger.info(f"✅ Índice FAISS actualizado exitosamente con {total_vectors} vectores")
            
            # Re-emmitir información importante del save_result para que los operadores
            # siguientes (p. ej. to_add_vector_response_operator) puedan acceder a
            # `saved_piece_vector` y construir la respuesta correctamente.
            return rx.of({
                "piece_id": piece_id,
                "num_vectors_added": total_vectors,
                "shared_faiss_id": shared_faiss_id,
                "success": True,
                "saved_piece_vector": saved_piece_vector,
                "saved_objects": saved_objects,
                "num_objects_saved": num_objects_saved
            })
            
        except Exception as e:
            logger.error(f"❌ Error actualizando FAISS: {e}")
            return rx.throw(UpdateFaissIndexException(
                message="Failed to update FAISS index",
                detail={
                    "error": str(e),
                    "piece_id": save_result.get("piece_id", "unknown")
                }
            ))
    
    @staticmethod
    def _get_faiss_api_credentials(
        logger: logging.Logger
    ) -> Optional[Tuple[str, str]]:

        try:
            credentials_env = os.getenv('COSINE_API_CREDENTIALS')
            if not credentials_env:
                logger.warning("⚠️ [UPDATE_FAISS] COSINE_API_CREDENTIALS no encontradas en entorno")
                return None
            credentials_list = json.loads(credentials_env)
            if not credentials_list or len(credentials_list) == 0:
                logger.warning("⚠️ [UPDATE_FAISS] No hay credenciales en COSINE_API_CREDENTIALS")
                return None
            cred = credentials_list[0]
            username = cred.get('username')
            password = cred.get('password')
            if not username or not password:
                logger.error("❌ [UPDATE_FAISS] Formato de credencial inválido en COSINE_API_CREDENTIALS")
                return None
            logger.info(f"✅ [UPDATE_FAISS] Credenciales cargadas para usuario: {username}")
            return (username, password)
        except json.JSONDecodeError as e:
            logger.error(f"❌ [UPDATE_FAISS] Error al parsear COSINE_API_CREDENTIALS: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ [UPDATE_FAISS] Error al obtener credenciales: {e}")
            return None
    
    @staticmethod
    def update_step(
        logger: logging.Logger,
        piece_id: str
        ):
        try:
            import httpx
        except Exception as e:
            logger.error(f"❌ [UPDATE_FAISS] httpx no disponible: {e}")
            raise UpdateFaissIndexException(
                message="httpx library is required but not available",
                detail={"reason": str(e)}
            )

        credentials = UpdateFaissWithMultipleVectorsOperator._get_faiss_api_credentials(
            logger
        )
        if not credentials:
            logger.error("❌ [UPDATE_FAISS] No hay credenciales válidas para FAISS API")
            raise UpdateFaissIndexException(
                message="Authentication credentials not configured",
                detail={"reason": "COSINE_API_CREDENTIALS not found or invalid"}
            )
        username, password = credentials

        cosine_api_url = os.getenv("COSINE_API_URL", "")
        load_faiss_endpoint = os.getenv("LOAD_FAISS_FROM_DB", "")
        url = f"{cosine_api_url.rstrip('/')}{load_faiss_endpoint}"

        # Auth header tipo Basic
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        auth_header = f"Basic {auth_b64}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_header,
        }

        logger.info(f"🔗 [UPDATE_FAISS] Realizando petición a: {url} para piece_id={piece_id}")
        try:
            response = httpx.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            logger.info(f"✅ [UPDATE_FAISS] FAISS actualizado correctamente para piece {piece_id} (status={response.status_code})")
            return True
        except httpx.HTTPError as e:
            logger.error(f"❌ [UPDATE_FAISS] Error HTTP: {e}")
            raise UpdateFaissIndexException(
                message="Failed to connect to FAISS update service",
                detail={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"❌ [UPDATE_FAISS] Error general: {e}")
            raise UpdateFaissIndexException(
                message="Failed to update FAISS index via remote HTTP call",
                detail={"reason": str(e)}
            )



def update_faiss_with_multiple_vectors_operator(
    logger: logging.Logger
) -> Callable[[Observable], Observable[Dict[str, Any]]]:
    """
    Operator para actualizar FAISS con múltiples vectores.

    :param faiss_repository: Repositorio FAISS
    :param logger: Logger instance
    :return: Observable con resultado de actualización
    """
    def operator(source: Observable) -> Observable[Dict[str, Any]]:
        return source.pipe(
            flat_map(lambda save_result: UpdateFaissWithMultipleVectorsOperator.execute(
                save_result,
                logger
            ))
        )
    return operator
