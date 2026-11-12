"""
PieceObject Repository Implementation - Infrastructure Layer (ORM)
"""
from reactivex import of

from src.domain.piece.repositories.piece_object_repository import PieceObjectRepository
from src.infraestructure.db.db_connection import get_database_manager
from src.infraestructure.persistance.piece.models.piece_object_model import PieceObjectModel


class PieceObjectRepositoryImpl(PieceObjectRepository):
    """
    Implementation of PieceObject repository using SQLAlchemy ORM.
    """

    def __init__(self):
        self.db_manager = get_database_manager()

    def create_query_builder(self):
        qb = self.db_manager.get_session().query(PieceObjectModel)
        return of(qb)
