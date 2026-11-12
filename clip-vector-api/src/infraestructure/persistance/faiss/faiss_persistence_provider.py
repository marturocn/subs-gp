"""
FAISS persistence module.
"""
from dishka import Provider, provide, Scope

from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.infraestructure.persistance.faiss.repositories.faiss_index_repository_impl import FaissIndexRepositoryImpl


class FaissPersistenceProvider(Provider):
    """
    Provider for FAISS persistence dependencies.
    """
    
    @provide(scope=Scope.APP)
    def get_faiss_index_repository(self) -> FaissIndexRepository:
        """Provide FAISS index repository implementation."""
        return FaissIndexRepositoryImpl()
