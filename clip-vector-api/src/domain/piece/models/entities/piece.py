"""
Domain Piece entity.
"""
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from decimal import Decimal


@dataclass
class Piece:
    """
    Domain Piece entity.
    Represents an artistic piece with its vector representation and metadata.
    """
    id: str
    room_id: Optional[str] = None
    inventory_number: Optional[str] = None
    object_status: Optional[str] = None
    object_type: Optional[str] = None
    inventory_date: Optional[datetime] = None
    inventory_contact_name: Optional[str] = None
    inventory_memo: Optional[str] = None
    collection_id: Optional[str] = None
    artist_id: Optional[str] = None
    title: Optional[str] = None
    medium: Optional[str] = None
    catalog_unique_id: Optional[str] = None
    subject: Optional[str] = None
    provenance: Optional[str] = None
    dimensions: Optional[str] = None
    acquisition_price: Optional[Decimal] = None
    status: Optional[int] = None
    image_src: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
