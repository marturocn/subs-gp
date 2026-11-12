"""
Vector API module.
"""
from dishka import Provider, Scope, provide

from src.domain.vector.use_cases.search_vector_case.search_vector_usecase import SearchVectorUseCase
from src.domain.vector.use_cases.load_vectors_from_db_case.load_vectors_from_db_usecase import LoadVectorsFromDbUseCase
from src.common.middlewares.basic_auth.basic_auth_middleware import BasicAuthMiddleware
from src.common.core.module_registry import api_module
from .vector_controller import VectorController

class VectorControllerProvider(Provider):
    scope = Scope.REQUEST

    @provide(scope=Scope.REQUEST)
    def get_vector_controller(
        self,
        search_vector_use_case: SearchVectorUseCase,
        load_vectors_from_db_use_case: LoadVectorsFromDbUseCase,
    ) -> VectorController:  # pyright: ignore[reportGeneralTypeIssues]
        print("🔎 VectorControllerProvider: creando VectorController")
        return VectorController(search_vector_use_case, load_vectors_from_db_use_case)


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
