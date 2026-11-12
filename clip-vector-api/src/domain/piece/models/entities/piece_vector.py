"""
Piece Vector Entity - Domain Layer
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class PieceVector:
    """
    Represents a vector associated with a piece from a specific image/angle.
    Multiple vectors can exist for the same piece (different angles/views).
    """
    id: str
    piece_id: str
    faiss_id: int
    vector: List[float]
    thumbnail: Optional[str] = None  # URL or path to thumbnail image
    piece_metadata: Optional[Dict[str, Any]] = None  # e.g., {"angle": "frontal", "description": "..."}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None