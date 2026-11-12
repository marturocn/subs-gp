"""
Generate Vector Use Case Provider - Application Layer
"""
from dishka import Provider, provide, Scope

from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.vector.use_cases.generate_vector_case import GenerateVectorUseCase
from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from .generate_vector_use_case_impl import GenerateVectorUseCaseImpl


class GenerateVectorProvider(Provider):
    """
    Dishka provider for GenerateVector use case dependencies.
    Uses Grounding DINO for artwork detection and DINOv3 for encoding.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_generate_vector_use_case(
        self, 
        dinov3_model_repository: DINOv3ModelRepository,
        dino_model_repository: GroundingDinoRepository
    ) -> GenerateVectorUseCase:
        """
        Proporciona el caso de uso para generar vectores con detección de obras de arte.
        
        Flow:
        1. Grounding DINO detecta obras en la imagen (pinturas, esculturas, etc.)
        2. DINOv3 codifica la imagen completa (pieza)
        3. Cada obra detectada se recorta y codifica con DINOv3
        4. Se retornan los vectores individuales (imagen completa + objetos)
        
        Uses optimized configuration for gallery detection.
        """
        return GenerateVectorUseCaseImpl(
            dinov3_model_repository,
            dino_model_repository
        )


__all__ = [
    "GenerateVectorUseCaseImpl",
    "GenerateVectorProvider",
]
