"""
Command for adding a vector to the FAISS index.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class AddVectorToIndexCommand:
    """
    Command for adding a vector to the FAISS index.
    """
    piece_id: str
    vector: List[float]
