"""
Standard error messages for server exceptions.
"""
from enum import Enum


class MessageException(str, Enum):
    """
    Standard error messages for exceptions.
    """
    # General system errors
    UNKNOWN_ERROR = "Unknown error"
    INTERNAL_SERVER_ERROR = "Internal server error"
    VALIDATION_ERROR = "Validation error"
    
    # Authentication and authorization errors
    UNAUTHORIZED = "Unauthorized"
    FORBIDDEN = "Forbidden access"
    TOKEN_EXPIRED = "Token expired"
    INVALID_TOKEN = "Invalid token"
    
    # Resource errors
    NOT_FOUND = "Resource not found"
    RESOURCE_ALREADY_EXISTS = "Resource already exists"
    RESOURCE_CONFLICT = "Resource conflict"
    
    # Data validation errors
    INVALID_INPUT = "Invalid input"
    MISSING_REQUIRED_FIELD = "Missing required field"
    INVALID_FORMAT = "Invalid format"
    
    # Database errors
    DATABASE_ERROR = "Database error"
    CONNECTION_ERROR = "Connection error"
    TRANSACTION_ERROR = "Transaction error"
    
    # External service errors
    EXTERNAL_SERVICE_ERROR = "External service error"
    NETWORK_ERROR = "Network error"
    TIMEOUT_ERROR = "Timeout exceeded"


class VectorErrorMessages(str, Enum):
    """
    Mensajes de error específicos para operaciones con vectores.
    """
    MISSING_OR_INVALID_ID = "Missing or invalid ID."
    INVALID_IMAGE = "Invalid or unreadable image."
    DATABASE_ERROR = "Failed to store vector in database."
    FAISS_INDEX_ERROR = "Internal error adding vector to index."
    MODEL_LOADING_ERROR = "Error loading the CLIP model."
    INTERNAL_ERROR = "Internal server error"
