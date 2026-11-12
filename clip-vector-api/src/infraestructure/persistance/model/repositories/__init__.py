"""
Model repository implementations.
"""
from .clip_model_repository_impl import ClipModelRepositoryImpl
from .yolo_model_repository_impl import YoloModelRepositoryImpl
from .owlvit_model_repository_impl import OwlvitRepositoryImpl
from .dinov3_model_repository_impl import DINOv3ModelRepositoryImpl

__all__ = [
    "ClipModelRepositoryImpl",
    "YoloModelRepositoryImpl",
    "OwlvitRepositoryImpl",
    "DINOv3ModelRepositoryImpl",
]