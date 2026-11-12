"""
Command for saving a piece vector.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class SavePieceVectorCommand:
    """
    Command for saving a piece vector.
    """
    piece_id: str
    vector: List[float]
