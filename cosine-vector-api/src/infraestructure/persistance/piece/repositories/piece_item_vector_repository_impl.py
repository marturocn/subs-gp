"""
Piece Item Vector Repository Implementation - Infrastructure Layer (ORM)
"""
import logging
from typing import Dict, List

from src.infraestructure.db.db_connection import get_database_manager
from src.infraestructure.persistance.piece.models.piece_object_model import PieceObjectModel


logger = logging.getLogger(__name__)


class PieceItemVectorRepositoryImpl:
    """
    Implementation for retrieving piece item vectors (objects) by piece vector IDs.
    """

    def __init__(self):
        self.db_manager = get_database_manager()

    def get_by_piece_vector_ids_sync(self, ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Retrieve objects for given piece vector IDs.
        Returns: {piece_vector_id: [{'object_name': str, 'vector': List[float], 'bounding_box': Dict}, ...]}
        """
        session = self.db_manager.get_session()
        try:
            objects = session.query(PieceObjectModel).filter(PieceObjectModel.piece_vector_id.in_(ids)).order_by(PieceObjectModel.piece_vector_id, PieceObjectModel.object_name).all()

            objects_by_id = {}
            for obj in objects:
                piece_vector_id = obj.piece_vector_id
                item = {
                    'object_name': obj.object_name,
                    'vector': obj.vector,
                    'bounding_box': obj.bounding_box
                }
                if piece_vector_id not in objects_by_id:
                    objects_by_id[piece_vector_id] = []
                objects_by_id[piece_vector_id].append(item)

            logger.debug(f"Retrieved objects for {len(ids)} piece vector IDs")
            return objects_by_id

        except Exception as e:
            logger.error(f"Error retrieving piece item vectors: {e}")
            return {}
        finally:
            session.close()