"""
RxPY Operator for saving vectors to database.
Saves the original full-image vector in piece_vector (one per image/angle).
Detected objects are saved in piece_object table with their vectors.
Each piece can have multiple images from different angles, and each image has its detected objects.
"""
import logging
import json
from typing import Callable, Dict, Any
import uuid
from reactivex import Observable
from reactivex.operators import map, flat_map, catch
import reactivex as rx

from src.infraestructure.persistance.piece.models.piece_vector_model import PieceVectorModel
from src.infraestructure.persistance.piece.models.piece_model import PieceModel
from src.infraestructure.persistance.piece.models.piece_object_model import PieceObjectModel
from src.domain.piece.repositories.piece_vector_repository import PieceVectorRepository
from src.domain.piece.repositories.piece_object_repository import PieceObjectRepository
from src.application.vector.use_cases.add_vector_impl.exceptions.save_to_database_exception import SaveToDatabaseException


def _generate_faiss_id_from_piece_uuid(piece_uuid: str) -> int:
    """
    Genera un FAISS ID base a partir del UUID de la pieza.
    Este ID base se usará para el primer vector, y se incrementará +1 por cada vector adicional.
    
    :param piece_uuid: UUID de la pieza (string)
    :return: FAISS ID base (entero positivo)
    """
    # Convertir UUID a entero usando los primeros 8 bytes
    import uuid as uuid_module
    piece_uuid_obj = uuid_module.UUID(piece_uuid)
    faiss_id = abs(int.from_bytes(piece_uuid_obj.bytes[:8], byteorder='big')) % (2**31 - 1)
    return faiss_id


class SaveVectorsToDbOperator:
    """
    Operador para guardar el vector de la imagen original en piece_vector
    y los objetos detectados en piece_object.
    
    Guarda:
    1. Un registro en piece_vector con el vector de la imagen original completa
    2. Múltiples registros en piece_object, uno por cada objeto detectado
    
    Nota: Cada imagen de la misma pieza (diferentes ángulos) genera un nuevo registro
    en piece_vector y sus propios objetos en piece_object.
    
    FAISS IDs:
    - Vector de imagen original: faiss_id = base_id
    - Objetos detectados: faiss_id = base_id + 1, base_id + 2, etc.
    """

    @staticmethod
    def execute(
        piece_repository: PieceVectorRepository,
        piece_object_repository: PieceObjectRepository,
        catalog_unique_id: str,
        encoding_result: Dict[str, Any],
        logger: logging.Logger
    ) -> Observable[Dict[str, Any]]:
        """
        Guarda el vector original, los vectores de objetos en piece_vector
        y la información de objetos detectados en piece_object.

        :param piece_repository: Repositorio para operaciones de piece_vector
        :param piece_object_repository: Repositorio para operaciones de piece_object
        :param catalog_unique_id: catalog_unique_id de la pieza (se usa para buscar el UUID)
        :param encoding_result: Resultado de codificación con original_embedding, main_embedding y objects
        :param logger: Logger instance
        :return: Observable[Dict] con información de guardado
        """
        try:
            original_embedding = encoding_result["original_embedding"]  # Vector de imagen completa SIN recortes
            _ = encoding_result["main_embedding"]  # Vector combinado de objetos (o igual a original si no hay objetos)
            objects = encoding_result["objects"]
            num_objects = encoding_result["num_objects"]
            
            logger.info(f"💾 Guardando vectores en base de datos para catalog_unique_id={catalog_unique_id}")
            logger.info("   - 1 vector original de imagen completa (piece_vector)")
            logger.info(f"   - {num_objects} objetos detectados (piece_object)")
            from sqlalchemy.orm import Query
            # Ejecutar todo en una única transacción
            def save_vectors(query: Query[PieceVectorModel]):
                session = query.session
                saved_piece_vector = None
                saved_objects = []
                try:
                    # 1. Obtener la pieza por catalog_unique_id para obtener su UUID (o auto-crearla si no existe)
                    piece = session.query(PieceModel).filter(PieceModel.catalog_unique_id == catalog_unique_id).first()
                    if not piece:
                        new_piece_id = str(uuid.uuid4())
                        piece = PieceModel(
                            id=new_piece_id,
                            catalog_unique_id=str(catalog_unique_id),
                            title=f"Articulo {catalog_unique_id}"
                        )
                        session.add(piece)
                        session.flush()
                        logger.info(f"   ✨ Pieza no existía en BD. Auto-creada: UUID={new_piece_id}, catalog_unique_id={catalog_unique_id}")
                    
                    piece_uuid = piece.id
                    logger.info(f"   ✅ Pieza lista: UUID={piece_uuid}")
                    
                    # 2. Comprobar si ya existe un vector idéntico para esta pieza (idempotencia)
                    existing_vectors = session.query(PieceVectorModel).filter(PieceVectorModel.piece_id == piece_uuid).all()
                    import numpy as np
                    orig_np = np.array(original_embedding, dtype=np.float32)
                    orig_norm = float(np.linalg.norm(orig_np))
                    
                    duplicate_found = None
                    if orig_norm > 0 and existing_vectors:
                        for ev in existing_vectors:
                            if ev.vector:
                                try:
                                    raw_v = ev.vector
                                    while isinstance(raw_v, str):
                                        raw_v = json.loads(raw_v)
                                    ev_np = np.array(raw_v, dtype=np.float32)
                                    ev_norm = float(np.linalg.norm(ev_np))
                                    if ev_norm > 0:
                                        sim = float(np.dot(orig_np, ev_np) / (orig_norm * ev_norm))
                                        if sim >= 0.995:
                                            duplicate_found = ev
                                            break
                                except Exception as ex_check:
                                    logger.warning(f"Error verificando similitud con vector existente: {ex_check}")
                    
                    if duplicate_found:
                        logger.info(f"   ℹ️ Vector idéntico ya existente en BD para catalog_unique_id={catalog_unique_id} (faiss_id={duplicate_found.faiss_id}). Se omite duplicado.")
                        saved_piece_vector = {
                            "id": duplicate_found.id,
                            "faiss_id": duplicate_found.faiss_id,
                            "num_objects": duplicate_found.piece_metadata.get("num_objects_detected", 0) if isinstance(duplicate_found.piece_metadata, dict) else 0,
                            "is_duplicate": True
                        }
                        return {
                            "piece_id": catalog_unique_id,
                            "shared_faiss_id": duplicate_found.faiss_id,
                            "saved_piece_vector": saved_piece_vector,
                            "saved_objects": [],
                            "num_objects_saved": 0,
                            "is_duplicate": True
                        }

                    # 3. Generar FAISS ID base a partir del UUID de la pieza + timestamp para unicidad
                    import time
                    timestamp_component = int(time.time() * 1000) % 1000000  # últimos 6 dígitos del timestamp
                    base_piece_faiss_id = _generate_faiss_id_from_piece_uuid(piece_uuid)
                    shared_faiss_id = base_piece_faiss_id + timestamp_component
                    logger.info(f"   - FAISS ID base: {shared_faiss_id} (para esta imagen)")
                    
                    # 4. NO borrar vectores existentes si son imágenes/ángulos distintos
                    # Cada imagen nueva agrega un nuevo piece_vector y sus objetos

                    # 4. Contador para incrementar faiss_id por cada objeto
                    current_faiss_id = shared_faiss_id

                    # 5. Guardar vector ORIGINAL de la imagen completa en piece_vector
                    original_vector_model = PieceVectorModel(
                        piece_id=piece_uuid,
                        faiss_id=current_faiss_id,
                        vector=json.dumps(original_embedding),
                        thumbnail=None,  # Thumbnail generation is handled by a different service
                        piece_metadata={
                            "type": "original-full-image",
                            "description": "Vector de imagen completa sin recortes",
                            "num_objects_detected": num_objects
                        }
                    )
                    session.add(original_vector_model)
                    session.flush()  # Para obtener el ID generado
                    
                    saved_piece_vector = {
                        "id": original_vector_model.id,
                        "faiss_id": current_faiss_id,
                        "num_objects": num_objects
                    }
                    logger.info(f"   ✅ Vector de imagen original guardado: faiss_id={current_faiss_id}")
                    
                    # Incrementar faiss_id para los objetos
                    current_faiss_id += 1

                    # 6. Guardar objetos detectados SOLO en piece_object (no en piece_vector)
                    for obj in objects:
                        obj_idx = obj["index"]
                        obj_label = obj["label"]
                        obj_confidence = obj["confidence"]
                        obj_embedding = obj["embedding"]
                        obj_box = obj["box"]  # [x1, y1, x2, y2]

                        # Guardar información del objeto en piece_object con su vector
                        piece_object_model = PieceObjectModel(
                            piece_vector_id=original_vector_model.id,  # Vinculado al vector de la imagen original
                            faiss_id=current_faiss_id,
                            object_name=obj_label,
                            vector=obj_embedding,  # Vector del objeto en columna separada
                            object_metadata={
                                "confidence": obj_confidence,
                                "index": obj_idx,
                                "description": f"Detected {obj_label} with {obj_confidence:.2%} confidence"
                            },
                            bounding_box={
                                "x1": obj_box[0],
                                "y1": obj_box[1],
                                "x2": obj_box[2],
                                "y2": obj_box[3],
                                "width": obj_box[2] - obj_box[0],
                                "height": obj_box[3] - obj_box[1]
                            }
                        )
                        session.add(piece_object_model)
                        
                        saved_objects.append({
                            "object_name": obj_label,
                            "faiss_id": current_faiss_id,
                            "confidence": obj_confidence
                        })
                        logger.info(f"   ✅ Objeto guardado: name={obj_label}, faiss_id={current_faiss_id}, conf={obj_confidence:.2%}")
                        
                        # Incrementar faiss_id para el siguiente objeto
                        current_faiss_id += 1

                    # Commit único para toda la transacción
                    session.commit()

                    logger.info("✅ Vector de imagen guardado en piece_vector")
                    logger.info(f"✅ Total de {len(saved_objects)} objetos guardados en piece_object")
                    logger.info(f"   FAISS IDs asignados: {shared_faiss_id} (imagen) y {shared_faiss_id + 1} a {current_faiss_id - 1} (objetos)")

                    return {
                        "piece_id": catalog_unique_id,
                        "shared_faiss_id": shared_faiss_id,
                        "saved_piece_vector": saved_piece_vector,
                        "saved_objects": saved_objects,
                        "num_objects_saved": len(saved_objects)
                    }
                except Exception:
                    # Asegurar rollback si ocurre cualquier error
                    try:
                        session.rollback()
                    except Exception as rb_err:
                        logger.error(f"Error during rollback: {rb_err}")
                    raise

            return piece_repository.create_query_builder().pipe(
                map(save_vectors),
                catch(lambda err, src: rx.throw(SaveToDatabaseException(
                    message="Error saving vectors to database",
                    detail={
                        "error": str(err),
                        "catalog_unique_id": catalog_unique_id,
                        "num_objects": num_objects
                    }
                )))
            )
            
        except Exception as e:
            logger.error(f"❌ Error guardando vectores: {e}")
            return rx.throw(SaveToDatabaseException(
                message="Failed to save vectors to database",
                detail={"error": str(e), "catalog_unique_id": catalog_unique_id}
            ))


def save_vectors_to_db_operator(
    piece_repository: PieceVectorRepository,
    piece_object_repository: PieceObjectRepository,
    logger: logging.Logger
) -> Callable[[Observable], Observable[Dict[str, Any]]]:
    """
    Operator para guardar el vector de imagen completa en piece_vector
    y los objetos detectados en piece_object.
    
    Estructura:
    - piece_vector: Un registro por imagen/ángulo de la pieza (vector de imagen completa)
    - piece_object: Múltiples registros por imagen, uno por cada objeto detectado
    
    Permite múltiples imágenes de la misma pieza desde diferentes ángulos.

    :param piece_repository: Repositorio para operaciones de piece_vector
    :param piece_object_repository: Repositorio para operaciones de piece_object
    :param logger: Logger instance
    :return: Observable con información de guardado
    """
    def operator(source: Observable) -> Observable[Dict[str, Any]]:
        return source.pipe(
            flat_map(lambda data: SaveVectorsToDbOperator.execute(
                piece_repository,
                piece_object_repository,
                data["command"].piece_id,
                data["encoding_result"],
                logger
            ))
        )
    return operator

