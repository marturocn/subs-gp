"""
SQLAlchemy model for piece_object table.
"""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    JSON,
    CHAR,
    func,
    text,
    ForeignKey,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.infraestructure.db.db_connection import Base
from src.domain.piece_object.models.entities.piece_object import PieceObject


class PieceObjectModel(Base):
    __tablename__ = "piece_item_vector"

    id: Mapped[str] = mapped_column(
        CHAR(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        server_default=text("UUID()"),
    )
    piece_vector_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("piece_vector.id", ondelete="CASCADE"), nullable=False)
    faiss_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    object_name: Mapped[str] = mapped_column(String(255), nullable=False)
    vector: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    object_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    bounding_box: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    def to_domain(self) -> PieceObject:
        """Convert ORM model to domain entity."""
        # Copiar object_metadata y agregar vector si existe
        metadata = dict(self.object_metadata) if self.object_metadata else {}
        if self.vector is not None:
            metadata['vector'] = self.vector
        return PieceObject(
            id=self.id,
            piece_vector_id=self.piece_vector_id,
            faiss_id=self.faiss_id,
            object_name=self.object_name,
            object_metadata=metadata if metadata else None,
            bounding_box=self.bounding_box if self.bounding_box else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, piece_object: PieceObject) -> "PieceObjectModel":
        """Convert domain entity to ORM model."""
        # Extraer vector de object_metadata si existe
        vector = None
        if piece_object.object_metadata and 'vector' in piece_object.object_metadata:
            vector = piece_object.object_metadata['vector']
        return cls(
            id=piece_object.id,
            piece_vector_id=piece_object.piece_vector_id,
            faiss_id=piece_object.faiss_id,
            object_name=piece_object.object_name,
            vector=vector,
            object_metadata=piece_object.object_metadata,
            bounding_box=piece_object.bounding_box,
        )
