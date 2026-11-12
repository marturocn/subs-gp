"""
Command for adding a vector to the system.
"""
from dataclasses import dataclass
from http import HTTPStatus
from typing import List
from io import BytesIO



@dataclass
class AddVectorCommand:
    """
    Command for adding a vector to the system.
    This coordinates encoding, storing, and indexing.
    """
    piece_id: str
    image_data: BytesIO



