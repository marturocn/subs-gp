"""
SQLAlchemy model for piece_vector table.
"""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import Optional, List

from sqlalchemy import (
    String,
    Text,
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
from src.domain.piece_vector.models.entities.piece_vector import PieceVector


class PieceVectorModel(Base):
    __tablename__ = "piece_vector"

    id: Mapped[str] = mapped_column(
        CHAR(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        server_default=text("UUID()"),
    )
    piece_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("piece.id", ondelete="CASCADE"), nullable=False)
    faiss_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vector: Mapped[List[float]] = mapped_column(JSON, nullable=False)
    thumbnail: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    piece_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    def to_domain(self) -> PieceVector:
        """Convert ORM model to domain entity."""
        vector_list: List[float] = []
        if self.vector is not None:
            # Handle both list/tuple and JSON string formats
            if isinstance(self.vector, (list, tuple)):
                vector_list = list(self.vector)
            elif isinstance(self.vector, str):
                # Parse JSON string to list
                import json
                try:
                    parsed = json.loads(self.vector)
                    if isinstance(parsed, (list, tuple)):
                        vector_list = list(parsed)
                    else:
                        # If it's not a list after parsing, leave empty
                        vector_list = []
                except ValueError:
                    # If parsing fails, leave empty
                    vector_list = []
            else:
                # Unknown type, leave empty
                vector_list = []
        
        return PieceVector(
            id=self.id,
            piece_id=self.piece_id,
            faiss_id=self.faiss_id,
            vector=vector_list,
            thumbnail=self.thumbnail,
            piece_metadata=self.piece_metadata if self.piece_metadata else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, piece_vector: PieceVector) -> "PieceVectorModel":
        """Convert domain entity to ORM model."""
        return cls(
            id=piece_vector.id,
            piece_id=piece_vector.piece_id,
            faiss_id=piece_vector.faiss_id,
            vector=piece_vector.vector,
            thumbnail=piece_vector.thumbnail,
            piece_metadata=piece_vector.piece_metadata,
        )
