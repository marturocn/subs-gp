"""
Add Vector Use Case Provider - Application Layer
"""
from dishka import Provider, provide, Scope

from application.vector.use_cases.add_vector_impl.add_vector_use_case_impl import AddVectorUseCaseImpl
from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.domain import AddVectorUseCase
from src.domain.piece.repositories.piece_vector_repository import PieceVectorRepository
from src.domain.piece.repositories.piece_object_repository import PieceObjectRepository



class AddVectorProvider(Provider):
    """
    Dishka provider for AddVector use case dependencies.
    Uses Grounding DINO + DINOv3 with multiple object detection and storage.
    Similar workflow to generate_vectors.py script.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_add_vector_use_case(
        self, 
        piece_vector_repository: PieceVectorRepository,
        piece_object_repository: PieceObjectRepository,
        dino_model_repository: GroundingDinoRepository,
        dinov3_model_repository: DINOv3ModelRepository
    ) -> AddVectorUseCase:
        """
        Proporciona el caso de uso para agregar vectores con detección y encoding de objetos.
        
        Complete Flow (similar a generate_vectors.py):
        1. Grounding DINO detecta objetos en la imagen (pinturas, esculturas, etc.)
        2. Cada objeto se recorta y codifica con DINOv3
        3. Los vectores se combinan en un vector principal ponderado
        4. Se guardan vector principal + vectores individuales en piece_vector
        5. Se guardan los objetos detectados con metadatos en piece_object
        6. Se actualiza FAISS con todos los vectores
        
        Uses:
        - ARTWORK_DINO_ADVANCED_CONFIG for optimized gallery detection
        - DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) for embeddings
        - Weighted combination based on detection confidence
        """
        return AddVectorUseCaseImpl(
            piece_vector_repository,
            piece_object_repository,
            dino_model_repository,
            dinov3_model_repository,
        )


__all__ = [
    "AddVectorProvider",
]
