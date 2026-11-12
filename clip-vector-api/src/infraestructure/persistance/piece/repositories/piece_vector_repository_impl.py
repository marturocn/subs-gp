"""
PieceVector Repository Implementation - Infrastructure Layer (ORM)
"""
import logging
import uuid
from typing import List, Optional

from reactivex import of
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from src.domain.piece.repositories.piece_vector_repository import PieceVectorRepository
from src.domain.piece.models.entities.piece_vector import PieceVector
from src.infraestructure.db.db_connection import get_database_manager
from src.infraestructure.persistance.piece.models.piece_vector_model import PieceVectorModel


from src.infraestructure.framework.providers.logger.logger import logger

class PieceVectorRepositoryImpl(PieceVectorRepository):
    """
    Implementation of PieceVector repository using SQLAlchemy ORM.
    """

    def __init__(self):
        self.db_manager = get_database_manager()

    def create_query_builder(self):
        qb = self.db_manager.get_session().query(PieceVectorModel)
        return of(qb)
