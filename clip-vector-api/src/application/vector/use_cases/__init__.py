"""
Use cases module for vector application layer.
"""
from .add_vector_impl import AddVectorUseCaseImpl, AddVectorProvider
from .generate_vector_impl import GenerateVectorUseCaseImpl, GenerateVectorProvider


# Dishka providers list
DISHKA_VECTOR_PROVIDERS = [
    AddVectorProvider(),
    GenerateVectorProvider(),
]

__all__ = [
    "AddVectorUseCaseImpl",
    "AddVectorProvider",
    "GenerateVectorUseCaseImpl",
    "GenerateVectorProvider",
    "DISHKA_VECTOR_PROVIDERS",
]