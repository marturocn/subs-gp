"""
Database configuration service with AWS Secrets Manager integration.
"""
import os
import json
from typing import Dict, Any
from reactivex import Observable, of
from reactivex import operators as ops

from src.infraestructure.framework.providers.aws.secrets_manager import ISecretManager
from src.infraestructure.db.config.configuration import get_env_credentials


class RdsCredentials:
    """RDS credentials structure from Secrets Manager."""
    
    def __init__(self, data: Dict[str, Any]):
        self.engine: str = data.get("engine", "mysql")
        self.host: str = data.get("host", "")
        self.port: int = int(data.get("port", 3306))
        self.dbname: str = data.get("dbname", "")
        self.username: str = data.get("username", "")
        self.password: str = data.get("password", "")


class DbCredentials:
    """Normalized database credentials."""
    
    def __init__(
        self,
        db_type: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str
    ):
        self.type = db_type
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password


class DbGetConfigService:
    """
    Database configuration service.
    Retrieves credentials from environment or AWS Secrets Manager.
    """
    
    def __init__(self, secret_manager: ISecretManager):
        """
        Initialize the service.
        
        Args:
            secret_manager: ISecretManager instance for retrieving secrets
        """
        self.sm = secret_manager
    
    @property
    def is_local(self) -> bool:
        """Check if using local database configuration (env vars instead of Secrets Manager)."""
        return os.getenv("AWS_USE_SECRETS_MANAGER", "false").lower() != "true"
    
    @property
    def secret_key(self) -> str:
        """
        Get the secret key name for database credentials.
        Format: SUBS_GP_DB_CREDENTIALS_${PY_ENV}
        """
        py_env = os.getenv("PY_ENV", "")
        prefix = os.getenv("AWS_DB_CREDENTIALS_SECRET_PREFIX", "")
        return f"{prefix}_{py_env}"
    
    def get_credentials(self) -> Observable[DbCredentials]:
        """
        Get database credentials as an Observable.
        
        Returns local credentials from env vars if AWS_USE_SECRETS_MANAGER is false,
        otherwise retrieves from AWS Secrets Manager.
        
        Returns:
            Observable[DbCredentials]: Observable that emits database credentials
        """
        if self.is_local:
            # Use local env-defined DB config
            local = get_env_credentials()
            if local is None:
                local = {}
            
            return of(DbCredentials(
                db_type=local.get("type", "mysql"),
                host=local.get("host", "localhost"),
                port=local.get("port", 3306),
                database=local.get("name", "root"),
                username=local.get("user", "root"),
                password=local.get("pass", "root"),
            ))
        
        # Use AWS Secrets Manager
        return self.sm.get_secret(self.secret_key).pipe(
            ops.map(lambda secret: json.loads(secret)),
            ops.map(lambda data: RdsCredentials(data)),
            ops.map(lambda cred: DbCredentials(
                db_type=cred.engine,
                host=cred.host,
                port=cred.port,
                database=cred.dbname,
                username=cred.username,
                password=cred.password,
            ))
        )
