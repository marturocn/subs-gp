"""
SQLAlchemy model for PieceVector table.
"""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    String,
    Text,
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
from src.domain.piece.models.entities.piece_vector import PieceVector


class PieceVectorModel(Base):
    __tablename__ = "piece_vector"

    id: Mapped[str] = mapped_column(
        CHAR(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        server_default=text("UUID()"),
    )
    piece_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("piece.id", ondelete="CASCADE"), nullable=False)
    faiss_id: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[List[float]] = mapped_column(JSON, nullable=False)
    thumbnail: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    piece_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    def to_domain(self) -> PieceVector:
        """Convert to domain entity."""
        return PieceVector(
            id=self.id,
            piece_id=self.piece_id,
            faiss_id=self.faiss_id,
            vector=list(self.vector) if isinstance(self.vector, (list, tuple)) else [],
            thumbnail=self.thumbnail,
            piece_metadata=dict(self.piece_metadata) if self.piece_metadata else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, piece_vector: PieceVector) -> "PieceVectorModel":
        """Create from domain entity."""
        return cls(
            id=piece_vector.id if piece_vector.id else str(uuid.uuid4()),
            piece_id=piece_vector.piece_id,
            faiss_id=piece_vector.faiss_id,
            vector=piece_vector.vector,
            thumbnail=piece_vector.thumbnail,
            piece_metadata=piece_vector.piece_metadata,
        )
