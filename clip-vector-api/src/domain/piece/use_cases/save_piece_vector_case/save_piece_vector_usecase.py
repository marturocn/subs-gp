"""
Use case for saving piece vectors.
"""
from abc import ABC, abstractmethod

from reactivex import Observable

from .commands import SavePieceVectorCommand
from src.common.core.use_case import UseCase


class SavePieceVectorUseCase(UseCase[SavePieceVectorCommand, None], ABC):
    """
    Abstract use case for saving a piece vector.
    """
    
    @abstractmethod
    def execute(self, command: SavePieceVectorCommand) -> Observable[None]:
        """
        Execute the use case to save a piece vector.
        
        Args:
            command: Command with the piece ID and vector to save
            
        Returns:
            Observable that completes when the vector is saved
        """
        pass
