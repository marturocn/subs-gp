"""
Middleware aplicador de rutas específicas.
Aplica middlewares basados en la configuración de ROUTE_MIDDLEWARE_REQUIREMENTS.
"""
import re
from typing import Callable, Awaitable, Dict, Set, Any, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.common.core.module_registry import ROUTE_MIDDLEWARE_REQUIREMENTS
from src.common.exceptions.custom_server_exception import CustomServerException


class RouteMiddlewareApplicator(BaseHTTPMiddleware):
    """
    Middleware que aplica middlewares específicos según la ruta y método HTTP.
    
    Este middleware lee la configuración de ROUTE_MIDDLEWARE_REQUIREMENTS y
    aplica los middlewares correspondientes a cada ruta de manera dinámica.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self._compiled_patterns: List[Dict[str, Any]] = []
        self._compile_route_patterns()
    
    def _compile_route_patterns(self):
        """
        Compila los patrones de rutas para optimizar el matching.
        Convierte rutas de FastAPI con parámetros a expresiones regulares.
        """
        self._compiled_patterns = []
        
        for requirement in ROUTE_MIDDLEWARE_REQUIREMENTS:
            path = requirement["path"]
            methods = requirement["methods"]
            middlewares = requirement["middlewares"]
            
            # Convertir rutas de FastAPI con parámetros a regex
            # Ejemplo: /vectors/add/{id} -> /vectors/add/[^/]+
            pattern = path
            pattern = re.sub(r'\{[^}]+\}', r'[^/]+', pattern)
            pattern = f"^{pattern}$"
            
            try:
                compiled_pattern = re.compile(pattern)
                self._compiled_patterns.append({
                    "pattern": compiled_pattern,
                    "original_path": path,
                    "methods": methods,
                    "middlewares": middlewares
                })
            except re.error as e:
                print(f"⚠️ Error compilando patrón de ruta {path}: {e}")
    
    def _find_matching_middlewares(self, path: str, method: str) -> List[Callable]:
        """
        Encuentra los middlewares que deben aplicarse a una ruta específica.
        
        Args:
            path: Ruta de la petición
            method: Método HTTP de la petición
            
        Returns:
            Lista de middlewares a aplicar
        """
        matching_middlewares = []
        method_upper = method.upper()
        
        for pattern_info in self._compiled_patterns:
            if pattern_info["pattern"].match(path):
                # Verificar si el método coincide
                if method_upper in pattern_info["methods"]:
                    matching_middlewares.extend(pattern_info["middlewares"])
        
        return matching_middlewares
    
    async def _apply_middleware_chain(
        self, 
        middlewares: List[Callable], 
        request: Request, 
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Aplica una cadena de middlewares de forma secuencial.
        
        Args:
            middlewares: Lista de middlewares a aplicar
            request: Request de FastAPI
            call_next: Siguiente función en la cadena
            
        Returns:
            Response procesada por todos los middlewares
        """
        if not middlewares:
            return await call_next(request)
        
        # Crear una cadena de middlewares
        async def create_middleware_chain(index: int):
            if index >= len(middlewares):
                return await call_next(request)
            
            middleware_class = middlewares[index]
            
            # Crear una instancia temporal del middleware para usar su dispatch
            # Nota: Esto crea un middleware "dummy" solo para usar su lógica de dispatch
            class TemporaryApp:
                async def __call__(self, scope, receive, send):
                    # Esta clase temporal no necesita implementación real
                    # Solo se usa como placeholder para instanciar middlewares
                    pass
            
            temp_app = TemporaryApp()
            middleware_instance = middleware_class(temp_app)
            
            async def next_in_chain(req: Request) -> Response:
                return await create_middleware_chain(index + 1)
            
            return await middleware_instance.dispatch(request, next_in_chain)
        
        return await create_middleware_chain(0)
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """
        Método principal del middleware. Determina qué middlewares aplicar y los ejecuta.
        
        Args:
            request: Request de FastAPI
            call_next: Siguiente función en la cadena
            
        Returns:
            Response procesada
        """
        path = request.url.path
        method = request.method
        
        # Buscar middlewares que coincidan con esta ruta
        matching_middlewares = self._find_matching_middlewares(path, method)
        
        if matching_middlewares:
            print(f"🔧 RouteMiddlewareApplicator: Aplicando {len(matching_middlewares)} middleware(s) a {method} {path}")
            
            try:
                # Aplicar la cadena de middlewares
                return await self._apply_middleware_chain(matching_middlewares, request, call_next)
            except CustomServerException as exc:
                # Manejar excepciones personalizadas del servidor
                print(f"🚨 RouteMiddlewareApplicator: Capturada CustomServerException: {exc}")
                return JSONResponse(
                    status_code=exc.get_status(),
                    content=exc.get_response()
                )
            except Exception as exc:
                # Manejar cualquier otra excepción
                print(f"🚨 RouteMiddlewareApplicator: Error no manejado: {exc}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal Server Error",
                        "message": "Error en middleware específico de ruta",
                        "detail": str(exc)
                    }
                )
        
        # Si no hay middlewares específicos, continuar normalmente
        return await call_next(request)
    
    def refresh_patterns(self):
        """
        Recompila los patrones de rutas. Útil si se actualizan los requirements dinámicamente.
        """
        self._compile_route_patterns()
