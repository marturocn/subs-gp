"""
SQLAlchemy model for Piece table.
"""
from __future__ import annotations

from datetime import datetime
import uuid
from decimal import Decimal
from typing import Optional, List

from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    Numeric,
    JSON,
    CHAR,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.infraestructure.db.db_connection import Base
from src.domain.piece.models.entities.piece import Piece


class PieceModel(Base):
    __tablename__ = "piece"

    # Generamos UUID en Python (default) y también declaramos el server_default para MySQL (UUID())
    # Así evitamos el error de NULL identity key al crear registros sin especificar id.
    id: Mapped[str] = mapped_column(
        CHAR(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        server_default=text("UUID()"),
    )
    room_id: Mapped[Optional[str]] = mapped_column(CHAR(36), nullable=True)
    inventory_number: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    object_status: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    object_type: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    inventory_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    inventory_contact_name: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    inventory_memo: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    collection_id: Mapped[Optional[str]] = mapped_column(CHAR(36), nullable=True)
    artist_id: Mapped[Optional[str]] = mapped_column(CHAR(36), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    medium: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    catalog_unique_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    subject: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    provenance: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    dimensions: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    acquisition_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_src: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    # Converters
    def to_domain(self) -> Piece:
        return Piece(
            id=self.id,
            room_id=self.room_id,
            inventory_number=self.inventory_number,
            object_status=self.object_status,
            object_type=self.object_type,
            inventory_date=self.inventory_date,
            inventory_contact_name=self.inventory_contact_name,
            inventory_memo=self.inventory_memo,
            collection_id=self.collection_id,
            artist_id=self.artist_id,
            title=self.title,
            medium=self.medium,
            catalog_unique_id=self.catalog_unique_id,
            subject=self.subject,
            provenance=self.provenance,
            dimensions=self.dimensions,
            acquisition_price=self.acquisition_price,
            status=self.status,
            image_src=self.image_src,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, piece: Piece) -> "PieceModel":
        return cls(
            id=piece.id,
            room_id=piece.room_id,
            inventory_number=piece.inventory_number,
            object_status=piece.object_status,
            object_type=piece.object_type,
            inventory_date=piece.inventory_date,
            inventory_contact_name=piece.inventory_contact_name,
            inventory_memo=piece.inventory_memo,
            collection_id=piece.collection_id,
            artist_id=piece.artist_id,
            title=piece.title,
            medium=piece.medium,
            catalog_unique_id=piece.catalog_unique_id,
            subject=piece.subject,
            provenance=piece.provenance,
            dimensions=piece.dimensions,
            acquisition_price=piece.acquisition_price,
            status=piece.status,
            image_src=piece.image_src,
        )
