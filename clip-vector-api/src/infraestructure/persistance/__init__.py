from .faiss.faiss_persistence_provider import FaissPersistenceProvider
from .model.model_persistence_provider import ModelPersistenceProvider
from .piece.piece_persistence_provider import PiecePersistenceProvider


BASE_ENTITY_PROVIDERS = [
    FaissPersistenceProvider(),  # FAISS persistence
    ModelPersistenceProvider(),  # Model persistence
    PiecePersistenceProvider(),  # Piece persistence
]
