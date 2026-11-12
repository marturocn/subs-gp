"""
Piece Object Entity - Domain Layer
"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class PieceObject:
    """
    Domain entity representing an object detected within a piece vector.
    Each piece vector can have multiple detected objects.
    The vector is stored in object_metadata['vector'].
    """
    id: str
    piece_vector_id: str
    faiss_id: Optional[int]
    object_name: str  # e.g., "chair", "table", "person"
    object_metadata: Optional[dict] = None  # JSON with 'vector', 'confidence', 'description'
    bounding_box: Optional[dict] = None  # JSON with bbox coordinates
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def get_vector(self) -> List[float]:
        """Get the vector from object_metadata['vector']"""
        if self.object_metadata and "vector" in self.object_metadata:
            return self.object_metadata["vector"]
        return []
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of the vector"""
        vector = self.get_vector()
        return len(vector) if vector else 0
    
    def get_confidence(self) -> Optional[float]:
        """Get detection confidence from metadata."""
        if self.object_metadata:
            return self.object_metadata.get('confidence')
        return None
    
    def get_description(self) -> Optional[str]:
        """Get the description from metadata."""
        if self.object_metadata:
            return self.object_metadata.get('description')
        return None
