"""
Improved Basic Authentication Middleware using configuration service.
"""
import base64
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Awaitable
from .exceptions.basic_auth_exception import BasicAuthException
from .config import get_basic_auth_config


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """
    Basic Authentication Middleware that loads credentials from configuration.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.auth_config = get_basic_auth_config()

    @staticmethod
    def _encode_credentials(username: str, password: str) -> str:
        """
        Encode username and password to Base64.
        
        Args:
            username: The username
            password: The password
            
        Returns:
            Base64 encoded credentials
        """
        raw = f"{username}:{password}"
        return base64.b64encode(raw.encode("utf-8")).decode("utf-8")

    @staticmethod
    def _decode_credentials(token: str) -> tuple[str, str]:
        """
        Decode Base64 token to username and password.
        
        Args:
            token: Base64 encoded token
            
        Returns:
            Tuple of (username, password)
            
        Raises:
            ValueError: If token is invalid
        """
        try:
            decoded = base64.b64decode(token.encode("utf-8")).decode("utf-8")
            if ":" not in decoded:
                raise ValueError("Invalid token format")
            username, password = decoded.split(":", 1)
            return username, password
        except Exception as e:
            raise ValueError(f"Failed to decode credentials: {e}")

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """
        Process the request and validate authentication.
        
        Args:
            request: The HTTP request
            call_next: The next middleware/handler
            
        Returns:
            HTTP response
            
        Raises:
            BasicAuthException: If authentication fails
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            raise BasicAuthException()

        token = auth_header.split(" ", 1)[1]
        
        try:
            username, password = self._decode_credentials(token)
            print(f"🔎 ImprovedBasicAuthMiddleware: Validating user: {username}")
            
            # Validate credentials using configuration service
            if not self.auth_config.validate_credentials(username, password):
                raise BasicAuthException(detail={"reason": "Invalid credentials"})
                
            print(f"✅ ImprovedBasicAuthMiddleware: User {username} authenticated successfully")
            
        except ValueError as e:
            print(f"❌ ImprovedBasicAuthMiddleware: Invalid token format: {e}")
            raise BasicAuthException(detail={"reason": "Invalid token format"})

        return await call_next(request)
