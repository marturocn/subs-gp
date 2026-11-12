"""
SQLAlchemy model for PieceObject table.
"""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    JSON,
    CHAR,
    ForeignKey,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.infraestructure.db.db_connection import Base
from src.domain.piece.models.entities.piece_object import PieceObject


class PieceObjectModel(Base):
    __tablename__ = "piece_item_vector"

    id: Mapped[str] = mapped_column(
        CHAR(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        server_default=text("UUID()"),
    )
    piece_vector_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("piece_vector.id", ondelete="CASCADE"), 
        nullable=False
    )
    faiss_id: Mapped[int] = mapped_column(Integer, nullable=False)
    object_name: Mapped[str] = mapped_column(String(255), nullable=False)
    vector: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    object_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    bounding_box: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    def to_domain(self) -> PieceObject:
        """Convert to domain entity."""
        return PieceObject(
            id=self.id,
            piece_vector_id=self.piece_vector_id,
            faiss_id=self.faiss_id,
            object_name=self.object_name,
            vector=list(self.vector) if self.vector else None,
            object_metadata=dict(self.object_metadata) if self.object_metadata else None,
            bounding_box=dict(self.bounding_box) if self.bounding_box else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, piece_object: PieceObject) -> "PieceObjectModel":
        """Create from domain entity."""
        return cls(
            id=piece_object.id if piece_object.id else str(uuid.uuid4()),
            piece_vector_id=piece_object.piece_vector_id,
            faiss_id=piece_object.faiss_id,
            object_name=piece_object.object_name,
            vector=piece_object.vector,
            object_metadata=piece_object.object_metadata,
            bounding_box=piece_object.bounding_box,
        )
