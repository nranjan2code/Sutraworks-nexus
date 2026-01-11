"""
NEXUS Authentication & Security
================================

API key authentication and optional JWT support.
Rate limiting configuration.

Security Model:
- API key authentication (required if NEXUS_API_KEY is set)
- Optional JWT for multi-user scenarios
- Rate limiting via slowapi
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# Use centralized logging
from nexus.service.logging_config import get_logger

logger = get_logger("auth")


@dataclass
class SecurityConfig:
    """Security configuration."""

    # API Key authentication
    api_key: Optional[str] = None
    api_key_header_name: str = "X-API-Key"

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10

    # Optional: JWT settings (for future multi-user support)
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("NEXUS_API_KEY"),
            api_key_header_name=os.getenv("NEXUS_API_KEY_HEADER", "X-API-Key"),
            rate_limit_enabled=os.getenv("NEXUS_RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests_per_minute=int(os.getenv("NEXUS_RATE_LIMIT_RPM", "60")),
            rate_limit_burst=int(os.getenv("NEXUS_RATE_LIMIT_BURST", "10")),
            jwt_secret=os.getenv("NEXUS_JWT_SECRET"),
            jwt_algorithm=os.getenv("NEXUS_JWT_ALGORITHM", "HS256"),
            jwt_expiry_hours=int(os.getenv("NEXUS_JWT_EXPIRY_HOURS", "24")),
        )


class AuthManager:
    """
    Manages API key and optional JWT authentication.

    Usage:
        auth = AuthManager.from_env()

        @app.get("/protected")
        async def protected_endpoint(api_key: str = Depends(auth.verify_api_key)):
            return {"status": "authenticated"}
    """

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._api_key_header = APIKeyHeader(
            name=config.api_key_header_name,
            auto_error=False,
        )

        if config.api_key:
            logger.info("API key authentication enabled")
        else:
            logger.warning("API key not set - endpoints are PUBLIC")

    @classmethod
    def from_env(cls) -> "AuthManager":
        """Create AuthManager from environment variables."""
        return cls(SecurityConfig.from_env())

    async def verify_api_key(
        self,
        request: Request,
        api_key: Optional[str] = Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
    ) -> Optional[str]:
        """
        Verify API key from request header.

        Returns:
            API key if valid, None if no auth required

        Raises:
            HTTPException: If API key is required but invalid
        """
        # If no API key configured, allow all requests
        if not self.config.api_key:
            return None

        # API key is required
        if not api_key:
            logger.warning(f"Missing API key from {request.client.host}")
            raise HTTPException(
                status_code=401,
                detail="API key required. Set X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(api_key, self.config.api_key):
            logger.warning(f"Invalid API key from {request.client.host}")
            raise HTTPException(
                status_code=403,
                detail="Invalid API key",
            )

        return api_key

    def require_auth(self) -> bool:
        """Check if authentication is required."""
        return self.config.api_key is not None

    def get_rate_limit_string(self) -> str:
        """Get rate limit string for slowapi."""
        return f"{self.config.rate_limit_requests_per_minute}/minute"


def get_client_ip(request: Request) -> str:
    """
    Get client IP address for rate limiting.

    Handles X-Forwarded-For header for proxied requests.
    """
    # Check for forwarded header (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP (original client)
        return forwarded.split(",")[0].strip()

    # Direct connection
    if request.client:
        return request.client.host

    return "unknown"


def create_api_key() -> str:
    """Generate a secure API key."""
    import secrets

    return f"nexus_{secrets.token_urlsafe(32)}"


# Optional: JWT utilities (requires python-jose)
try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True

    def create_jwt_token(
        data: dict,
        secret: str,
        algorithm: str = "HS256",
        expiry_hours: int = 24,
    ) -> str:
        """Create a JWT token."""
        to_encode = data.copy()
        expire = time.time() + (expiry_hours * 3600)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, secret, algorithm=algorithm)

    def verify_jwt_token(
        token: str,
        secret: str,
        algorithm: str = "HS256",
    ) -> Optional[dict]:
        """Verify a JWT token."""
        try:
            payload = jwt.decode(token, secret, algorithms=[algorithm])
            return payload
        except JWTError:
            return None

except ImportError:
    JWT_AVAILABLE = False

    def create_jwt_token(*args, **kwargs):
        raise RuntimeError("JWT support not available. Install python-jose.")

    def verify_jwt_token(*args, **kwargs):
        raise RuntimeError("JWT support not available. Install python-jose.")


# Singleton auth manager
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create the global auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager.from_env()
    return _auth_manager
