"""
Use cases module for model application layer.
"""
from .encode_image_impl.encode_image_usecase_impl import EncodeImageUseCaseImpl
from .detect_objects_impl import DetectObjectsUseCaseImpl, DetectObjectsProvider


# Dishka providers list
DISHKA_MODEL_PROVIDERS = [
    DetectObjectsProvider(),
]

__all__ = [
    "EncodeImageUseCaseImpl",
    "DetectObjectsUseCaseImpl",
    "DetectObjectsProvider",
    "DISHKA_MODEL_PROVIDERS",
]