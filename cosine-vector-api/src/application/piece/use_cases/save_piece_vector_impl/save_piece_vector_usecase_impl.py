"""
Save Piece Vector use case implementation.
"""
import reactivex
from reactivex import Observable

from application.piece.use_cases.save_piece_vector_impl.exceptions.insert_to_database_exception import InsertToDatabaseException
from src.domain.piece.use_cases.save_piece_vector_case import SavePieceVectorUseCase, SavePieceVectorCommand
from src.domain.piece.repositories.piece_repository import PieceRepository


class SavePieceVectorUseCaseImpl(SavePieceVectorUseCase):
    """
    Implementation of SavePieceVectorUseCase.
    """
    
    def __init__(self, repository: PieceRepository):
        """
        Initialize the use case with a piece repository.
        
        Args:
            repository: Piece repository for data persistence
        """
        self.repository = repository
    
    def execute(self, command: SavePieceVectorCommand) -> Observable[None]:
        """
        Execute the save piece vector use case.
        
        Args:
            command: Command with piece ID and vector data
            
        Returns:
            Observable[None]: Observable that completes when vector is saved
        """
        try:
            # Save vector synchronously to avoid event loop issues
            self.repository.save_vector_sync(command.piece_id, command.vector)
            return reactivex.of(None)
        except Exception as e:
            raise InsertToDatabaseException(detail={"error": str(e)})
