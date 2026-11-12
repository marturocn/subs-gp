"""
Domain repository interface for DINOv3 model operations.
"""
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List
from PIL import Image
import numpy as np


class DINOv3ModelRepository(ABC):
    """
    Abstract repository for DINOv3 model operations.
    DINOv3 is a vision transformer model for generating image embeddings.
    """
    
    @abstractmethod
    async def encode_image(self, image_data: BytesIO) -> np.ndarray:
        """
        Generate embedding for a full image using DINOv3 (async).
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            Normalized embedding vector as numpy array
        """
        pass

    @abstractmethod
    def encode_images_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Generate embeddings for a batch of images using DINOv3 (async).

        Args:
            images: A list of PIL Image objects

        Returns:
            A list of normalized embedding vectors as numpy arrays
        """
        pass

    @abstractmethod
    def encode_image_sync(self, image_data: BytesIO) -> np.ndarray:
        """
        Generate embedding for a full image using DINOv3 (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            Normalized embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    async def encode_cropped_image(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for a cropped/PIL image using DINOv3 (async).
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    def encode_cropped_image_sync(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for a cropped/PIL image using DINOv3 (synchronous).
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (e.g., 1024 for DINOv3-large)
        """
        pass
