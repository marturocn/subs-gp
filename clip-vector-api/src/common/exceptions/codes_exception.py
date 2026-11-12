"""
Códigos de error estándar para excepciones del servidor.
"""
from enum import IntEnum


class CodeException(IntEnum):
    """
    Códigos de error estándar para excepciones.
    """
    # Errores generales del sistema
    UNKNOWN_ERROR = 1000
    INTERNAL_SERVER_ERROR = 1001
    VALIDATION_ERROR = 1002
    
    # Errores de autenticación y autorización
    UNAUTHORIZED = 2001
    FORBIDDEN = 2002
    TOKEN_EXPIRED = 2003
    INVALID_TOKEN = 2004
    
    # Errores de recursos
    NOT_FOUND = 3001
    RESOURCE_ALREADY_EXISTS = 3002
    RESOURCE_CONFLICT = 3003
    
    # Errores de validación de datos
    INVALID_INPUT = 4001
    MISSING_REQUIRED_FIELD = 4002
    INVALID_FORMAT = 4003
    
    # Errores de base de datos
    DATABASE_ERROR = 5001
    CONNECTION_ERROR = 5002
    TRANSACTION_ERROR = 5003
    
    # Errores de servicios externos
    EXTERNAL_SERVICE_ERROR = 6001
    NETWORK_ERROR = 6002
    TIMEOUT_ERROR = 6003


class VectorErrorCodes(IntEnum):
    """
    Códigos de error específicos para operaciones con vectores.
    """
    MISSING_OR_INVALID_ID = 3001
    INVALID_IMAGE = 3002
    DATABASE_ERROR = 3003
    FAISS_INDEX_ERROR = 3008
    MODEL_LOADING_ERROR = 3009
    INTERNAL_ERROR = 5000
