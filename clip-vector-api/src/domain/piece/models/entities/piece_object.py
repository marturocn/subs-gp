"""
Piece Object Entity - Domain Layer
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class PieceObject:
    """
    Represents an object detected within a piece's image.
    Each object has its own vector and bounding box information.
    """
    id: str
    piece_vector_id: str
    faiss_id: int
    object_name: str  # e.g., "chair", "table", "painting"
    vector: Optional[List[float]] = None  # Vector embedding of the detected object
    object_metadata: Optional[Dict[str, Any]] = None  # e.g., {"confidence": 0.95, "description": "A red chair"}
    bounding_box: Optional[Dict[str, Any]] = None  # e.g., {"x1": 10, "y1": 20, "x2": 100, "y2": 150}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
