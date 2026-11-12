from .faiss.faiss_persistence_provider import FaissPersistenceProvider
from .piece.piece_persistence_provider import PiecePersistenceProvider


BASE_ENTITY_PROVIDERS = [
    FaissPersistenceProvider(),  # FAISS persistence
    PiecePersistenceProvider(),  # Piece persistence
]
