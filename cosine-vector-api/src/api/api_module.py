
# ⚠️ Dishka container is async
from fastapi import APIRouter
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Any, List
from src.common.core.module_registry import REGISTERED_CONTROLLERS, ROUTE_MIDDLEWARE_REQUIREMENTS
from src.application.application_module import application_container

def _apply_middlewares_to_router(controller_instance, middlewares_map):
    """Placeholder: la aplicación real de middlewares debe ocurrir en una capa
    superior (@api_module o global) usando add_middleware de FastAPI o dependencies.

    Aquí solo validamos que existan y dejamos metadata en el controller si se requiere.
    """
    if not middlewares_map:
        return
    setattr(controller_instance, "__middlewares__", middlewares_map)




def _register_class_middlewares(controller, class_middlewares, api_prefix):
    for route in controller.router.routes:
        path = getattr(route, 'path', None)
        methods = getattr(route, 'methods', [])
        if not path or not methods:
            continue
        if path.startswith(controller.router.prefix):
            combined = path
        else:
            combined = f"{controller.router.prefix}{path}".replace('//', '/')
        full_path = f"{api_prefix}{combined}".replace('//', '/')
        if class_middlewares:
            ROUTE_MIDDLEWARE_REQUIREMENTS.append({
                "path": full_path,
                "methods": {m.upper() for m in methods},
                "middlewares": class_middlewares,
            })

def _register_specific_middlewares(controller, middlewares_map, api_prefix):
    specific_keys = [k for k in middlewares_map if isinstance(k, tuple)]
    if not specific_keys:
        return
    for key in specific_keys:
        rel_path, method = key
        mws = middlewares_map[key]
        for route in controller.router.routes:
            path = getattr(route, 'path', None)
            if not path:
                continue
            if path.startswith(controller.router.prefix):
                combined = path
            else:
                combined = f"{controller.router.prefix}{path}".replace('//', '/')
            full_path = f"{api_prefix}{combined}".replace('//', '/')
            if rel_path in path or rel_path in combined:
                ROUTE_MIDDLEWARE_REQUIREMENTS.append({
                    "path": full_path,
                    "methods": {method.upper()},
                    "middlewares": mws,
                })

def _register_route_middlewares(controller, middlewares_map, api_prefix):
    """Registra los middlewares requeridos para cada ruta del controlador."""
    if not middlewares_map:
        return
    class_middlewares: List[Any] = []
    for key, mws in middlewares_map.items():
        if not isinstance(key, tuple):
            class_middlewares.extend(mws)
    _register_class_middlewares(controller, class_middlewares, api_prefix)
    _register_specific_middlewares(controller, middlewares_map, api_prefix)

def create_api_v1_router(controllers, api_prefix: str = "") -> APIRouter:
    api_v1_router = APIRouter()
    ROUTE_MIDDLEWARE_REQUIREMENTS.clear()
    for controller, entry in controllers:
        middlewares_map = entry.get('middlewares') if isinstance(entry, dict) else {}
        _apply_middlewares_to_router(controller, middlewares_map)
        _register_route_middlewares(controller, middlewares_map, api_prefix)
        api_v1_router.include_router(controller.router)
    try:
        from infraestructure import logger
        logger.debug("[api_module] Rutas protegidas registradas: " + ", ".join(r['path'] for r in ROUTE_MIDDLEWARE_REQUIREMENTS))
    except Exception:
        pass
    return api_v1_router

async def create_router(api_prefix: str = "") -> APIRouter:
    async with application_container() as request_scope:
        instances_with_meta = []
        for entry in REGISTERED_CONTROLLERS:
            controller_cls = entry.get("controller") if isinstance(entry, dict) else entry
            instance = await request_scope.get(controller_cls)
            instances_with_meta.append((instance, entry))
    return create_api_v1_router(instances_with_meta, api_prefix)