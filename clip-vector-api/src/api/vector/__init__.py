"""
Vector API module.
"""
from dishka import Provider, Scope, provide

from src.common.middlewares.basic_auth.basic_auth_middleware import BasicAuthMiddleware
from src.common.core.module_registry import api_module
from .vector_controller import VectorController
from src.domain import AddVectorUseCase
from src.domain.vector.use_cases.generate_vector_case import GenerateVectorUseCase


class VectorControllerProvider(Provider):
    scope = Scope.REQUEST

    @provide(scope=Scope.REQUEST)
    def get_vector_controller(
        self,
        add_vector_use_case: AddVectorUseCase,
        generate_vector_use_case: GenerateVectorUseCase,
    ) -> VectorController:  # pyright: ignore[reportGeneralTypeIssues]
        print("🔎 VectorControllerProvider: creando VectorController")
        return VectorController(add_vector_use_case, generate_vector_use_case)


__all__ = [
    "VectorController",
    "VectorControllerProvider",
]

@api_module(
    middlewares={
        VectorController: [BasicAuthMiddleware],  # aquí la clase
    }
)
class VectorControllerModule:
    provider = VectorControllerProvider()
    controller = VectorController
