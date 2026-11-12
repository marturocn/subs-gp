"""
Model persistence module.
"""
from dishka import Provider, provide, Scope

from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.model.repositories.owlvit_repository import OwlvitRepository
from src.domain.model.repositories.clip_model_repository import ClipModelRepository
from src.domain.model.repositories.yolo_model_repository import YoloModelRepository
from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.infraestructure.persistance.model.repositories.clip_model_repository_impl import ClipModelRepositoryImpl
from src.infraestructure.persistance.model.repositories.yolo_model_repository_impl import YoloModelRepositoryImpl
from src.infraestructure.persistance.model.repositories.dinov3_model_repository_impl import DINOv3ModelRepositoryImpl


class ModelPersistenceProvider(Provider):
    """
    Provider for Model persistence dependencies.
    """
    

    @provide(scope=Scope.APP)
    def get_grounding_dino_repository(self) -> GroundingDinoRepository:
        """Provide Grounding DINO model repository implementation."""
        from src.infraestructure.persistance.model.repositories.grounding_dino_repository_impl import GroundingDinoRepositoryImpl
        # Instantiate repository and attempt to load the model at application startup so
        # the first request doesn't pay the model-loading latency.
        repo = GroundingDinoRepositoryImpl()
        try:
            # Attempt to eagerly load the model. This is idempotent and safe to call.
            # If it fails, we log the error but still return the repo to allow the
            # application to start; the repo will fall back to lazy loading on first use.
            repo._load_model()
        except Exception as e:
            # Import logger lazily to avoid circular imports at module import time.
            import logging
            logging.getLogger(__name__).warning(f"Failed to eagerly load Grounding DINO model at startup: {e}")
        return repo
    
    @provide(scope=Scope.APP)
    def get_owlvit_repository(self) -> OwlvitRepository:
        """Provide OWL-ViT v2 model repository implementation."""
        from src.infraestructure.persistance.model.repositories.owlvit_model_repository_impl import OwlvitRepositoryImpl
        # Instantiate repository and attempt to load the model at application startup so
        # the first request doesn't pay the model-loading latency.
        repo = OwlvitRepositoryImpl()
        try:
            # Attempt to eagerly load the model. This is idempotent and safe to call.
            # If it fails, we log the error but still return the repo to allow the
            # application to start; the repo will fall back to lazy loading on first use.
            repo._load_model()
        except Exception as e:
            # Import logger lazily to avoid circular imports at module import time.
            import logging
            logging.getLogger(__name__).warning(f"Failed to eagerly load OWL-ViT model at startup: {e}")
        return repo
    
    @provide(scope=Scope.APP)
    def get_dinov3_model_repository(self) -> DINOv3ModelRepository:
        """Provide DINOv3 model repository implementation."""
        repo = DINOv3ModelRepositoryImpl()
        try:
            # Attempt to eagerly load the model. This is idempotent and safe to call.
            # If it fails, we log the error but still return the repo to allow the
            # application to start; the repo will fall back to lazy loading on first use.
            repo._load_model()
        except Exception as e:
            # Import logger lazily to avoid circular imports at module import time.
            import logging
            logging.getLogger(__name__).warning(f"Failed to eagerly load DINOv3 model at startup: {e}")
        return repo
