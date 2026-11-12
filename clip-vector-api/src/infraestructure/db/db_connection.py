# infrastructure/db/database.py
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from functools import lru_cache
from typing import Optional
from dishka import Provider, provide, Scope

from src.infraestructure.db.config.db_get_config_service import DbGetConfigService, DbCredentials
from src.infraestructure.framework.providers.aws.secrets_manager import ISecretManager

Base = declarative_base()


class DatabaseManager:
    """
    Database manager that uses DbGetConfigService for credentials.
    Supports both local env config and AWS Secrets Manager.
    """
    
    def __init__(self, db_config_service: DbGetConfigService):
        self.db_config_service = db_config_service
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker[Session]] = None
        self._credentials: Optional[DbCredentials] = None
    
    def _get_credentials(self) -> DbCredentials:
        """Get credentials, caching the result."""
        if self._credentials is None:
            # Subscribe and get credentials synchronously
            def on_next(creds: DbCredentials) -> None:
                self._credentials = creds
            
            def on_error(e: Exception) -> None:
                raise RuntimeError(f"Failed to get database credentials: {e}")
            
            self.db_config_service.get_credentials().subscribe(
                on_next=on_next,
                on_error=on_error
            )
        
        if self._credentials is None:
            raise RuntimeError("Failed to retrieve database credentials")
        
        return self._credentials
    
    @property
    def engine(self) -> Engine:
        if self._engine is None:
            creds = self._get_credentials()
            database_url = f"mysql+pymysql://{creds.username}:{creds.password}@{creds.host}:{creds.port}/{creds.database}"
            self._engine = create_engine(
                database_url,
                echo=False,
                pool_pre_ping=True
            )
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        return self._session_factory
    
    def get_session(self) -> Session:
        return self.session_factory()


# Instancia global del gestor de base de datos
@lru_cache()
def get_database_manager() -> DatabaseManager:
    from src.infraestructure.framework.providers.aws.secrets_manager import SecretsManagerProvider
    
    # Get secret manager from provider
    sm_provider = SecretsManagerProvider()
    secret_manager = sm_provider.get_secret_manager()
    
    # Create config service
    db_config_service = DbGetConfigService(secret_manager)
    
    return DatabaseManager(db_config_service)


# Dependency para FastAPI
def get_db_session():
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


class DatabaseProvider(Provider):
    """
    Dishka provider for database connections and entities.
    Centralizes all database-related dependencies.
    """
    
    @provide(scope=Scope.APP)
    def get_db_config_service(self, secret_manager: ISecretManager) -> DbGetConfigService:
        """Proporciona el servicio de configuración de base de datos."""
        return DbGetConfigService(secret_manager)
    
    @provide(scope=Scope.APP)
    def get_database_manager(self, db_config_service: DbGetConfigService) -> DatabaseManager:
        """Proporciona el gestor de base de datos como singleton."""
        return DatabaseManager(db_config_service)
    
    @provide(scope=Scope.APP)
    def get_engine(self, db_manager: DatabaseManager) -> Engine:
        """Proporciona el engine de SQLAlchemy como singleton."""
        return db_manager.engine
    
    @provide(scope=Scope.APP)
    def get_session_factory(self, db_manager: DatabaseManager) -> sessionmaker[Session]:
        """Proporciona el factory de sesiones como singleton."""
        return db_manager.session_factory
    
    @provide(scope=Scope.REQUEST)
    def get_session(self, session_factory: sessionmaker[Session]) -> Session:
        """Proporciona una sesión de base de datos por request."""
        session = session_factory()
        try:
            return session
        except Exception:
            session.close()
            raise


# Dishka database provider para uso en el contenedor principal
DISHKA_DB_PROVIDER = DatabaseProvider()