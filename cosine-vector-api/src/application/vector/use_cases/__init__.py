"""
Use cases module for vector application layer.
"""

from src.application.vector.use_cases.load_vectors_from_db_impl import LoadVectorsFromDbProvider
from .search_vector_impl import SearchVectorUseCaseImpl, SearchVectorProvider


# Dishka providers list
DISHKA_VECTOR_PROVIDERS = [
    SearchVectorProvider(),
    LoadVectorsFromDbProvider(),
]

__all__ = [
    "SearchVectorUseCaseImpl",
    "SearchVectorProvider",
    "LoadVectorsFromDbProvider",
    "DISHKA_VECTOR_PROVIDERS",
]