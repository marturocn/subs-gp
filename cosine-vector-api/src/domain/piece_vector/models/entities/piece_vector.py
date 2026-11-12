"""
Piece Vector Entity - Domain Layer
"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class PieceVector:
    """
    Domain entity representing a piece vector.
    Each piece can have multiple vectors for different angles/views.
    """
    id: str
    piece_id: str
    faiss_id: Optional[int]
    vector: List[float]
    thumbnail: Optional[str] = None  # URL o path del thumbnail
    piece_metadata: Optional[dict] = None  # JSON con metadata como {"angle": "frontal", "description": "..."}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def get_dimension(self) -> int:
        """Get the dimension of the vector."""
        return len(self.vector)
    
    def get_angle(self) -> Optional[str]:
        """Get the angle/view from metadata."""
        if self.piece_metadata:
            return self.piece_metadata.get('angle')
        return None
    
    def get_description(self) -> Optional[str]:
        """Get the description from metadata."""
        if self.piece_metadata:
            return self.piece_metadata.get('description')
        return None
