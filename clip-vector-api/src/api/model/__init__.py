"""
Model API module.
"""
from dishka import Provider, Scope, provide

from src.common.middlewares.basic_auth.basic_auth_middleware import BasicAuthMiddleware
from src.common.core.module_registry import api_module
from .model_controller import ModelController
from src.domain.model.use_cases.detect_objects_case import DetectObjectsUseCase


class ModelControllerProvider(Provider):
    scope = Scope.REQUEST

    @provide(scope=Scope.REQUEST)
    def get_model_controller(
        self,
        detect_objects_use_case: DetectObjectsUseCase,
    ) -> ModelController:  # pyright: ignore[reportGeneralTypeIssues]
        print("🔎 ModelControllerProvider: creando ModelController")
        return ModelController(detect_objects_use_case)


__all__ = [
    "ModelController",
    "ModelControllerProvider",
]

@api_module(
    middlewares={
        ModelController: [BasicAuthMiddleware],  # aquí la clase
    }
)
class ModelControllerModule:
    provider = ModelControllerProvider()
    controller = ModelController
