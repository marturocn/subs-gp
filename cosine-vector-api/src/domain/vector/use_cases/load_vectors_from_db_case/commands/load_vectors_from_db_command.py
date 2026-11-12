"""
Command for loading vectors from database.
"""
from dataclasses import dataclass


@dataclass
class LoadVectorsFromDbCommand:
    """
    Command for loading vectors from database to FAISS index.
    This command has no parameters since it loads all vectors.
    """
    pass  # No parameters needed for this command
