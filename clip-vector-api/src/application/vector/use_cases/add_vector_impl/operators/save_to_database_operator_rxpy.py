"""
RxPY Operator for saving vector to database.
"""
import logging
from typing import Callable
import reactivex as rx
from reactivex import Observable
from reactivex.operators import map, flat_map, catch

from src.application.vector.use_cases.add_vector_impl.exceptions.add_vector_exception import AddVectorException
from src.infraestructure.persistance.piece.models.piece_vector_model import PieceVectorModel
from src.domain.piece.repositories.piece_vector_repository import PieceVectorRepository


class FindPiecesByIdOperator:
    """
    Operador para guardar vectores en la tabla piece_vector.
    """

    @staticmethod
    def execute(piece_repository: PieceVectorRepository, piece: PieceVectorModel, logger: logging.Logger) -> Observable[int]:
        """
        Inserta un nuevo vector en la tabla piece_vector para el piece_id dado,
        devolviendo un Observable con el número de filas afectadas (int).

        :param piece_repository: Repositorio para operaciones de base de datos
        :param piece_id: ID de la pieza asociada
        :param faiss_id: ID único para FAISS
        :param vector: Vector a guardar (en formato JSON)
        :param vector_type: Tipo de vector (e.g., "full-image", "object-{class_name}")
        :param description: Descripción opcional del vector
        :return: Observable[int] con el número de filas afectadas
        """
            # Crear un nuevo registro para piece_vector
        piece_vector = PieceVectorModel(
                piece_id=piece.piece_id,
                faiss_id=piece.faiss_id,
                vector=piece.vector,
                vector_type=piece.vector_type,
                description= piece.description,
            )
        logger.info(f"Guardando vector en base de datos: piece_id={piece.piece_id}, faiss_id={piece.faiss_id}, vector_type={piece.vector_type}")
            # Usar el repositorio para insertar el registro
            # Usar el repositorio para insertar el registro
        return piece_repository.create_query_builder().pipe(
                map(lambda query: (query.session.add(piece_vector), query.session.commit(), 1)[2]),
                catch(lambda err, src: rx.throw(AddVectorException(
                    message="Error saving vector to database",
                    detail={"error": str(err), "piece_id": piece.piece_id, "faiss_id": piece.faiss_id, "vector_type": piece.vector_type}
                )))
            )
  

def save_to_database_operator(
    piece: PieceVectorModel,
    logger: logging.Logger,
    piece_repository: PieceVectorRepository,
) -> Callable[[Observable], Observable[int]]:
    """
    Operator to save a vector to the database for a given piece.
    Utilizes the FindPiecesByIdOperator static class to execute the logic.

    :param piece_id: ID de la pieza asociada
    :param faiss_id: ID único para FAISS
    :param vector: Vector a guardar
    :param vector_type: Tipo de vector
    :param logger: Logger instance for tracking the save process
    :param piece_repository: The PieceVectorRepository for database operations
    :param description: Optional description of the vector
    :returns Observable containing the number of rows affected (int)
    """
    def operator(source: Observable) -> Observable[int]:
        logger.info(f"Saving vector for piece ID: {piece.piece_id}, FAISS ID: {piece.faiss_id}, Vector Type: {piece.vector_type}")
        return source.pipe(
            flat_map(lambda _: FindPiecesByIdOperator.execute(
                piece_repository, piece, logger
            ))
        )
    return operator