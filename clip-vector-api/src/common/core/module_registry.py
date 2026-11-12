from typing import (
    Type,
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
    cast,
)
import inspect

MiddlewareType = Callable[..., Any]
# RouteKey admite:
#  - Clase del controlador (Type)
#  - Tupla (path_relativo, metodo_http)
#  - Callable decorado que envuelve una clase (p.e. resultado de @inject)
RouteKey = Union[Type[Any], Tuple[str, str], Callable[..., Any]]

REGISTERED_PROVIDERS: List[Any] = []
REGISTERED_CONTROLLERS: List[Dict[str, Any]] = []
REGISTERED_CLASSES: set[str] = set()
# Rutas que requieren middlewares (se llena tras instanciar controllers)
ROUTE_MIDDLEWARE_REQUIREMENTS: List[Dict[str, Any]] = []


def _normalize_controller_key(key: RouteKey) -> RouteKey:
    """Normaliza la clave del diccionario de middlewares.

    - Si es una tupla (path, method) normaliza el método a MAYÚSCULAS.
    - Si es un callable decorado que en realidad representa una clase (p.e. @inject),
      intenta recuperar la clase original vía __wrapped__ o __qualname__.
    """
    if isinstance(key, tuple):
        if len(key) != 2:
            raise TypeError("Route middleware key tuple must have 2 elements: (path, method)")
        path, method = key
        if not isinstance(path, str) or not isinstance(method, str):
            raise TypeError("Route middleware key tuple must be (str, str)")
        return (path, method.upper())
    # Clase directa
    if isinstance(key, type):
        return key
    # Callable posiblemente decorando una clase
    if callable(key):
        original = getattr(key, "__wrapped__", None)
        if isinstance(original, type):
            return cast(Type[Any], original)
  
        # Lo dejamos tal cual; validación adicional ocurrirá después
        return key
    return key


def _validate_middlewares(
    middlewares: Optional[Dict[RouteKey, List[MiddlewareType]]]
) -> Dict[RouteKey, List[MiddlewareType]]:
    if middlewares is None:
        return {}
    if not isinstance(middlewares, dict):
        raise TypeError("`middlewares` must be a dict or None")

    normalized: Dict[RouteKey, List[MiddlewareType]] = {}
    for raw_key, mws in middlewares.items():
        key = _normalize_controller_key(raw_key)
        is_route = isinstance(key, tuple)
        is_class_like = isinstance(key, type) or (callable(key) and not is_route)
        if not (is_route or is_class_like):
            raise TypeError(
                "Keys in `middlewares` must be a controller class (or its decorated callable) or a (path, method) tuple"
            )
        if not isinstance(mws, list) or not all(callable(m) for m in mws):
            raise TypeError("Values in `middlewares` must be lists of callables")
        # Combinar si ya existe (p.e. misma ruta agregada varias veces)
        existing = normalized.get(key, [])
        existing.extend(mws)
        normalized[key] = existing
    return normalized


def _register_module(
    cls: Type[Any],
    middlewares: Optional[Dict[RouteKey, List[MiddlewareType]]]
) -> Type[Any]:
    if not hasattr(cls, "provider") or not hasattr(cls, "controller"):
        raise ValueError(f"{cls.__name__} must define both `provider` and `controller` attributes")

    provider = cls.provider
    controller = cls.controller

    if not callable(getattr(provider, "__class__", None)) or not isinstance(provider, object):
        raise TypeError("`provider` must be an instance of a Dishka Provider")

    class_id = f"{cls.__name__}:{provider.__class__.__name__}"
    if class_id not in REGISTERED_CLASSES:
        normalized_middlewares = _validate_middlewares(middlewares)
        REGISTERED_PROVIDERS.append(provider)
        REGISTERED_CONTROLLERS.append({
            "controller": controller,
            # Guardamos middlewares normalizados (métodos en mayúsculas, claves unificadas)
            "middlewares": normalized_middlewares
        })
        REGISTERED_CLASSES.add(class_id)

    return cls


def api_module(
    cls: Optional[Type[Any]] = None,
    *,
    middlewares: Optional[Dict[RouteKey, List[MiddlewareType]]] = None
) -> Any:
    """Decorador para registrar módulos API con su provider, controller y middlewares.

    Formatos aceptados para `middlewares`:
        middlewares={
            MiController: [AuthMiddleware, LoggingMiddleware],
            ("ruta/relativa", "post"): [OtroMiddleware]
        }

    Notas:
        - Los métodos HTTP se normalizan a mayúsculas.
        - Si la clase del controller está decorada (por ejemplo con @inject) y el decorador
          retorna un callable, igualmente puede usarse como clave.
        - Si se definen middlewares tanto a nivel de clase como de ruta, el orden de aplicación
          esperado (que debe implementar la fase de montaje) será primero los de la clase y luego
          los específicos de la ruta.
    """

    def wrapper(inner_cls: Type[Any]) -> Type[Any]:
        # _register_module internamente validará y normalizará
        return _register_module(inner_cls, middlewares)

    if cls is not None and callable(cls):  # Se usó sin paréntesis
        return wrapper(cls)
    return wrapper
