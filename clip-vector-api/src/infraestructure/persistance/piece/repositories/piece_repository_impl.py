"""
Piece Repository Implementation - Infrastructure Layer (ORM)
"""
from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from torch import cat

from src.domain.piece.repositories.piece_repository import PieceRepository
from src.domain.piece.models.entities.piece import Piece
from src.infraestructure.db.db_connection import get_database_manager
from src.infraestructure.persistance.piece.models.piece_model import PieceModel
from src.infraestructure.framework.providers.logger.logger import logger
from reactivex import of
from sqlalchemy import select

class PieceRepositoryImpl(PieceRepository):
    """
    Implementación del repositorio Piece usando SQLAlchemy ORM (sin SQL crudo).
    """

    def __init__(self):
        self.db_manager = get_database_manager()

    def create_query_builder(self):
        qb = self.db_manager.get_session().query(PieceModel)
        return of(qb)

    def save_vector_sync(self, piece_id: str, vector: List[float]) -> None:
        """
        Actualiza el vector usando ORM.
        """
        session = self.db_manager.get_session()
        try:
            result = session.query(PieceModel).filter(PieceModel.catalog_unique_id == piece_id).update({PieceModel.vector: vector})
            if result:
                session.commit()
                logger.info(f"Vector updated for piece {piece_id}")
            else:
                logger.warning(f"No piece found with ID {piece_id} to update vector")
                raise ValueError(f"No piece found with ID {piece_id} to update vector")
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Integrity error updating vector for piece {piece_id}: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating vector for piece {piece_id}: {e}")
            raise
        finally:
            session.close()

    async def save_vector(self, piece_id: str, vector: List[float]) -> None:
        """
        Envoltorio asíncrono de save_vector_sync.
        """
        self.save_vector_sync(piece_id, vector)

    async def get_by_id(self, piece_id: str) -> Optional[Piece]:
        """
        Obtiene una pieza por su ID usando ORM.
        """
        session = self.db_manager.get_session()
        try:
            piece_model: Optional[PieceModel] = session.get(PieceModel, piece_id)
            if not piece_model:
                return None
            return piece_model.to_domain()
        except Exception as e:
            logger.error(f"Error retrieving piece {piece_id}: {e}")
            raise
        finally:
            session.close()

    async def exists(self, piece_id: str) -> bool:
        """
        Verifica si existe la pieza por ID usando ORM.
        """
        session = self.db_manager.get_session()
        try:
            return session.get(PieceModel, piece_id) is not None
        except Exception as e:
            logger.error(f"Error checking if piece {piece_id} exists: {e}")
            raise
        finally:
            session.close()
