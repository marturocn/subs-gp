
"""
Main entry point for the FastAPI application.
Configures and launches the app with all necessary dependencies.
"""
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from common.exceptions.http.my_custom_exception import MyCustomException
load_dotenv()

from infraestructure import logger
import uvicorn
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Dict
import traceback

from fastapi import FastAPI
from fastapi.responses import JSONResponse


from dishka import make_async_container, Provider
from dishka.integrations.fastapi import setup_dishka

from src.api.api_module import create_api_v1_router, create_router
from src.infraestructure.framework.providers.global_filters_provider import configure_global_exception_handlers
from src.infraestructure.framework.config.config_root_module import get_app_config
from src.infraestructure.framework.middleware.exception_middleware import GlobalExceptionMiddleware
from src.infraestructure.framework.middleware.route_middleware_applicator import RouteMiddlewareApplicator

# Importar providers de Dishka
from src.application.application_module import APPLICATION_PROVIDERS
from src.infraestructure.persistance.persistance_module import PERSISTENCE_PROVIDERS


class MainProvider(Provider):
    """
    Main provider that configures all application dependencies.
    This provider can contain global configurations.
    """
    pass  # Specific providers are handled separately


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle manager.
    Runs at the start and end of the application.
    """
    # Setup when starting the application
    logger.info("🚀 Starting FastAPI application...")

    # Here you can add initialization logic such as:
    # - Database connection

    try:
        from src.infraestructure.db.db_connection import get_database_manager
        from sqlalchemy import text
        db_manager = get_database_manager()
        engine = db_manager.engine
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful.")
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    
    # Load vectors from database on startup
    try:
        from src.application.vector.use_cases.load_vectors_from_db_impl import load_vectors_from_db_handler
        from src.infraestructure.persistance.faiss.repositories.faiss_index_repository_impl import FaissIndexRepositoryImpl
        from src.infraestructure.persistance.piece.repositories.piece_vector_repository_impl import PieceVectorRepositoryImpl
        
        logger.info("📥 [STARTUP] Loading vectors from database on application startup...")
        faiss_repo = FaissIndexRepositoryImpl()
        piece_vector_repo = PieceVectorRepositoryImpl()
        
        result = await load_vectors_from_db_handler(
            faiss_index_repository=faiss_repo,
            piece_vector_repository=piece_vector_repo,
        )
        logger.info(f"✅ [STARTUP] Vectors loaded successfully: {result.data.get('loaded_count', 0)} vectors loaded")
    except Exception as e:
        logger.error(f"⚠️  [STARTUP] Could not load vectors on startup: {str(e)}")
        logger.error("⚠️  [STARTUP] Application will continue without pre-loaded vectors")
        logger.error(f"⚠️  Full traceback: {traceback.format_exc()}")
    
    # - Cache initialization
    # - External services setup
    
    yield
    
    # Cleanup when shutting down the application
    logger.info("🛑 Shutting down FastAPI application...")


async def create_app() -> FastAPI:
    """
    Factory function to create the FastAPI application.
    
    Returns:
        FastAPI: The configured application instance
    """
    # Get application config
    app_config = get_app_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="Clip Vector API",
        description="API for CLIP vector management with hexagonal architecture",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        debug=app_config.is_dev,
    )

    # IMPORTANT:UNKNOWN Exception middleware
    app.add_middleware(GlobalExceptionMiddleware)
    
    # IMPORTANT: Manejador para nuestra excepción personalizada
    from src.common.exceptions.custom_server_exception import CustomServerException
    @app.exception_handler(CustomServerException)
    async def custom_server_exception_handler(request, exc: CustomServerException):
        logger.error(f"🔥 CustomServerException handler: Capturada {type(exc).__name__}: {str(exc)}")
        logger.error(f"🔥 Full traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=exc.get_status(),
            content=exc.get_response()
        )

    # Configure Dishka container with all providers
    all_providers = [MainProvider(),*APPLICATION_PROVIDERS]
    container = make_async_container(*all_providers)
    setup_dishka(container, app)
    
    api_prefix = os.getenv("API_PREFIX", "/v1")
    router = await create_router(api_prefix)
    # Include API routers
    app.include_router(
        router=router,
        prefix=api_prefix,
        tags=["API v1"]
    )
    
    # IMPORTANT: Agregar el middleware de rutas específicas DESPUÉS de registrar las rutas
    # Esto debe ir después de create_router() para que ROUTE_MIDDLEWARE_REQUIREMENTS esté funcionando
    app.add_middleware(RouteMiddlewareApplicator)

    # Add health check endpoint and test unknown filter
    @app.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, Any]:
        """Endpoint to check application status."""
        raise RuntimeError("This is a custom exception for testing purposes.")
    
    # Register the function to avoid warnings
    _ = health_check
    
    # Print all registered routes
    logger.info("\n🛣️  Registered routes in the application:")
    logger.info("-" * 70)
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = ", ".join(sorted(getattr(route, 'methods', [])))
            path = getattr(route, 'path', '')
            # Show additional info if it's a mounted router
            route_info = ""
            if hasattr(route, 'prefix') and getattr(route, 'prefix'):
                route_info = f" (prefix: {getattr(route, 'prefix')})"
            logger.info(f"  {methods:<15} {path:<30} {route_info}")
        elif hasattr(route, 'path'):
            path = getattr(route, 'path', '')
            route_info = ""
            if hasattr(route, 'prefix') and getattr(route, 'prefix'):
                route_info = f" (prefix: {getattr(route, 'prefix')})"
            logger.info(f"  {'MOUNT':<15} {path:<30} {route_info}")
    logger.info("-" * 70)
    
    return app


import asyncio

def main() -> None:
    """
    Main function to run the application.
    """
    # Get config
    app_config = get_app_config()
    
    # Create the application (await the coroutine)
    app = asyncio.run(create_app())
    
    # Configure uvicorn
    log_level = app_config.log_level.lower() if hasattr(app_config, 'log_level') else "info"
    

    logger.info(f"🌟 Starting server in {app_config.env} mode")
    logger.info(f"🔧 Log level set to: {log_level}")
    logger.info(f"📡 Server available at: http://0.0.0.0:{app_config.api_port}")
    logger.info(f"📚 API documentation: http://0.0.0.0:{app_config.api_port}/api/docs")

    # Run the server
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=app_config.api_port,
        reload=app_config.is_dev,
        log_level=log_level,
        access_log=True,
    )
# Instancia de la aplicación para Gunicorn

# Detectar si se está ejecutando bajo Gunicorn
import os
import sys

if sys.argv[0].endswith("gunicorn"):  # Gunicorn
    import asyncio
    app = asyncio.run(create_app())
else:
    # Para Uvicorn: exponer un objeto awaitable que inicializa la app solo una vez
    class AppWrapper:
        def __init__(self):
            self._app = None
        async def __call__(self, scope, receive, send):
            if self._app is None:
                self._app = await create_app()
            await self._app(scope, receive, send)
    app = AppWrapper()

if __name__ == "__main__":
    main()