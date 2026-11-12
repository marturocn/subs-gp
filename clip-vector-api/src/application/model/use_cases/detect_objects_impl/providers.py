"""
Detect Objects Use Case Provider - Application Layer
"""
from dishka import Provider, provide, Scope

from src.application.model.use_cases.detect_objects_impl.detect_objects_use_case_impl import DetectObjectsUseCaseImpl
from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.use_cases.detect_objects_case import DetectObjectsUseCase


class DetectObjectsProvider(Provider):
    """
    Dishka provider for DetectObjects use case dependencies.
    Provides Google Lens-like object detection functionality using Grounding DINO.
    """
    
    @provide(scope=Scope.REQUEST)
    def get_detect_objects_use_case(
        self, 
        grounding_dino_model_repository: GroundingDinoRepository,
    ) -> DetectObjectsUseCase:
        """
        Proporciona el caso de uso para detectar objetos en imágenes.
        
        Flow:
        1. Recibe una imagen y un prompt opcional
        2. Usa Grounding DINO para detectar objetos
        3. Devuelve los objetos detectados con sus coordenadas
        
        Args:
            grounding_dino_model_repository: Repository para el modelo Grounding DINO
            
        Returns:
            Implementación del caso de uso DetectObjects
        """
        return DetectObjectsUseCaseImpl(
            grounding_dino_model_repository
        )


__all__ = [
    "DetectObjectsProvider",
    "DetectObjectsUseCaseImpl",
]
