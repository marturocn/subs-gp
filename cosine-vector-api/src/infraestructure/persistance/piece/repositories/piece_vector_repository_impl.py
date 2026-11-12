"""
Piece Vector Repository Implementation - Infrastructure Layer (ORM)
"""
import logging
from typing import List, Optional

from sqlalchemy.exc import IntegrityError

from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.domain.piece_vector.models.entities.piece_vector import PieceVector
from src.infraestructure.db.db_connection import get_database_manager
from src.infraestructure.persistance.piece.models.piece_vector_model import PieceVectorModel


logger = logging.getLogger(__name__)


class PieceVectorRepositoryImpl(PieceVectorRepository):
    """
    Implementation of PieceVectorRepository using SQLAlchemy ORM.
    """

    def __init__(self):
        self.db_manager = get_database_manager()

    async def save(self, piece_vector: PieceVector) -> PieceVector:
        """Save or update a piece vector using ORM."""
        session = self.db_manager.get_session()
        try:
            # Check if it exists
            existing = session.get(PieceVectorModel, piece_vector.id)
            
            if existing:
                # Update existing
                existing.piece_id = piece_vector.piece_id
                existing.faiss_id = piece_vector.faiss_id
                existing.vector = piece_vector.vector
                session.commit()
                session.refresh(existing)
                logger.info(f"✅ Updated piece vector {piece_vector.id}")
                return existing.to_domain()
            else:
                # Create new
                model = PieceVectorModel.from_domain(piece_vector)
                session.add(model)
                session.commit()
                session.refresh(model)
                logger.info(f"✅ Created piece vector {model.id}")
                return model.to_domain()
                
        except IntegrityError as e:
            session.rollback()
            logger.error(f"❌ Integrity error saving piece vector: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving piece vector: {e}")
            raise
        finally:
            session.close()

    async def get_by_id(self, vector_id: str) -> Optional[PieceVector]:
        """Get a piece vector by its ID using ORM."""
        session = self.db_manager.get_session()
        try:
            model: Optional[PieceVectorModel] = session.get(PieceVectorModel, vector_id)
            if not model:
                return None
            return model.to_domain()
        except Exception as e:
            logger.error(f"❌ Error retrieving piece vector {vector_id}: {e}")
            raise
        finally:
            session.close()
    
    async def get_by_ids(self, faiss_ids: List[str]) -> List[tuple[str, str, str]]:
        """Get multiple piece vectors by their FAISS IDs with JOIN to get catalog_unique_id in one query."""
        return self.get_by_ids_sync(faiss_ids)
    
    def get_by_ids_sync(self, faiss_ids: List[str]) -> List[tuple[str, str, str]]:
        """Get multiple piece vectors by their FAISS IDs (synchronous version).
        
        This method queries both piece_vector and piece_object tables to get:
        - faiss_id: The FAISS index ID
        - piece_id: The actual piece ID (from piece_vector or piece_object.piece_vector_id)
        - catalog_unique_id: The catalog ID from the piece table
        
        Returns:
            List of tuples (faiss_id_str, piece_id, catalog_unique_id)
        """
        session = self.db_manager.get_session()
        try:
            if not faiss_ids:
                return []
            
            from src.infraestructure.persistance.piece.models.piece_model import PieceModel
            from src.infraestructure.persistance.piece.models.piece_object_model import PieceObjectModel
            
            # Convertir faiss_ids de string a int
            faiss_ids_int = [int(fid) for fid in faiss_ids if fid.isdigit()]
            
            if not faiss_ids_int:
                logger.warning("⚠️ No valid faiss_ids to query")
                return []
            
            # Query unificada con UNION para obtener vectores de ambas tablas
            # Subconsulta 1: piece_vector -> piece
            query_vectors = session.query(
                PieceVectorModel.faiss_id,
                PieceVectorModel.piece_id,
                PieceModel.catalog_unique_id
            ).join(
                PieceModel,
                PieceVectorModel.piece_id == PieceModel.id
            ).filter(
                PieceVectorModel.faiss_id.in_(faiss_ids_int)
            )
            
            # Subconsulta 2: piece_object -> piece_vector -> piece
            query_objects = session.query(
                PieceObjectModel.faiss_id,
                PieceVectorModel.piece_id,
                PieceModel.catalog_unique_id
            ).join(
                PieceVectorModel,
                PieceObjectModel.piece_vector_id == PieceVectorModel.id
            ).join(
                PieceModel,
                PieceVectorModel.piece_id == PieceModel.id
            ).filter(
                PieceObjectModel.faiss_id.in_(faiss_ids_int)
            )
            
            # UNION de ambas subconsultas y DISTINCT para eliminar duplicados
            results = query_vectors.union(query_objects).distinct().all()
            
            # Convertir a lista de tuplas (faiss_id como string, piece_id, catalog_unique_id)
            vector_data = [(str(row[0]), row[1], row[2]) for row in results]
            
            logger.info(f"📊 Retrieved {len(vector_data)} unique vectors with catalog_unique_id from {len(faiss_ids)} requested faiss_ids")
            
            return vector_data
        except Exception as e:
            logger.error(f"❌ Error retrieving vectors by faiss_ids with catalog_unique_id: {e}")
            raise
        finally:
            session.close()

    async def get_by_piece_id(self, piece_id: str) -> List[PieceVector]:
        """Get all vectors for a specific piece using ORM."""
        session = self.db_manager.get_session()
        try:
            models = session.query(PieceVectorModel).filter(
                PieceVectorModel.piece_id == piece_id
            ).all()
            
            vectors = [model.to_domain() for model in models]
            logger.info(f"📊 Retrieved {len(vectors)} vectors for piece {piece_id}")
            return vectors
        except Exception as e:
            logger.error(f"❌ Error retrieving vectors for piece {piece_id}: {e}")
            raise
        finally:
            session.close()

    async def get_all_vectors(self) -> List[PieceVector]:
        """
        Get all vectors from both piece_vector and piece_object tables.
        piece_object vectors are stored in object_metadata['vector'] JSON field.
        """
        session = self.db_manager.get_session()
        try:
            from src.infraestructure.persistance.piece.models.piece_object_model import PieceObjectModel
            
            # Load from piece_vector table
            piece_vector_models = session.query(PieceVectorModel).all()
            vectors = [model.to_domain() for model in piece_vector_models]
            logger.info(f"📊 Retrieved {len(vectors)} vectors from piece_vector table")
            
            # Load from piece_object table
            piece_object_models = session.query(PieceObjectModel).filter(
                PieceObjectModel.faiss_id.isnot(None)
            ).all()
            
            # Convert PieceObject to PieceVector by extracting vector from object_metadata
            for obj_model in piece_object_models:
                obj = obj_model.to_domain()
                vector_data = obj.get_vector()
                
                if vector_data and len(vector_data) > 0:
                    # Create a PieceVector from the object's data
                    # Use the piece_vector_id as piece_id since piece_object links to piece_vector
                    piece_vector = PieceVector(
                        id=obj.id,  # Use object ID as vector ID
                        piece_id=obj.piece_vector_id,  # Link to piece_vector
                        faiss_id=obj.faiss_id,
                        vector=vector_data,
                        thumbnail=None,  # Objects don't have thumbnails
                        piece_metadata={
                            "object_name": obj.object_name,
                            "confidence": obj.get_confidence(),
                            "description": obj.get_description(),
                            "bounding_box": obj.bounding_box
                        }
                    )
                    vectors.append(piece_vector)
            
            logger.info(f"📊 Retrieved {len(piece_object_models)} vectors from piece_object table")
            logger.info(f"✅ Total vectors loaded: {len(vectors)} (piece_vector + piece_object)")
            return vectors
        except Exception as e:
            logger.error(f"❌ Error retrieving all vectors: {e}")
            raise
        finally:
            session.close()

    async def delete_by_piece_id(self, piece_id: str) -> None:
        """Delete all vectors for a specific piece using ORM."""
        session = self.db_manager.get_session()
        try:
            deleted_count = session.query(PieceVectorModel).filter(
                PieceVectorModel.piece_id == piece_id
            ).delete()
            
            session.commit()
            logger.info(f"🗑️ Deleted {deleted_count} vectors for piece {piece_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error deleting vectors for piece {piece_id}: {e}")
            raise
        finally:
            session.close()

    async def update_faiss_id(self, vector_id: str, faiss_id: int) -> None:
        """Update the FAISS ID for a piece vector using ORM."""
        session = self.db_manager.get_session()
        try:
            result = session.query(PieceVectorModel).filter(
                PieceVectorModel.id == vector_id
            ).update({PieceVectorModel.faiss_id: faiss_id})
            
            if result:
                session.commit()
                logger.info(f"✅ Updated FAISS ID for vector {vector_id} to {faiss_id}")
            else:
                logger.warning(f"⚠️ Vector {vector_id} not found. No update performed.")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error updating FAISS ID for vector {vector_id}: {e}")
            raise
        finally:
            session.close()
