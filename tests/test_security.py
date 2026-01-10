"""
Security Tests
===============

Tests for authentication and rate limiting.
"""

import os
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import HTTPException

from nexus.service.auth import (
    AuthManager,
    SecurityConfig,
    get_auth_manager,
    get_client_ip,
    create_api_key,
)


class TestSecurityConfig:
    """Test security configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SecurityConfig()

        assert config.api_key is None
        assert config.api_key_header_name == "X-API-Key"
        assert config.rate_limit_enabled is True
        assert config.rate_limit_requests_per_minute == 60

    @patch.dict(
        os.environ,
        {
            "NEXUS_API_KEY": "test-key-123",
            "NEXUS_RATE_LIMIT_RPM": "30",
        },
    )
    def test_from_env(self):
        """Test loading config from environment."""
        config = SecurityConfig.from_env()

        assert config.api_key == "test-key-123"
        assert config.rate_limit_requests_per_minute == 30


class TestAuthManager:
    """Test authentication manager."""

    def test_auth_manager_no_key(self):
        """Test auth manager without API key configured."""
        config = SecurityConfig(api_key=None)
        auth = AuthManager(config)

        assert auth.require_auth() is False

    def test_auth_manager_with_key(self):
        """Test auth manager with API key configured."""
        config = SecurityConfig(api_key="secret-key")
        auth = AuthManager(config)

        assert auth.require_auth() is True

    def test_get_rate_limit_string(self):
        """Test rate limit string format."""
        config = SecurityConfig(rate_limit_requests_per_minute=100)
        auth = AuthManager(config)

        assert auth.get_rate_limit_string() == "100/minute"


class TestAPIKeyGeneration:
    """Test API key generation."""

    def test_create_api_key(self):
        """Test API key generation."""
        key = create_api_key()

        assert key.startswith("nexus_")
        assert len(key) > 20  # Should be reasonably long

    def test_keys_are_unique(self):
        """Test that generated keys are unique."""
        keys = [create_api_key() for _ in range(100)]
        assert len(set(keys)) == 100  # All unique


class TestClientIP:
    """Test client IP extraction."""

    def test_client_ip_direct(self):
        """Test getting IP from direct connection."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None
        mock_request.client.host = "192.168.1.100"

        ip = get_client_ip(mock_request)
        assert ip == "192.168.1.100"

    def test_client_ip_forwarded(self):
        """Test getting IP from X-Forwarded-For header."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "10.0.0.1, 10.0.0.2"

        ip = get_client_ip(mock_request)
        assert ip == "10.0.0.1"  # First IP in chain

    def test_client_ip_no_client(self):
        """Test fallback when no client info."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None
        mock_request.client = None

        ip = get_client_ip(mock_request)
        assert ip == "unknown"


class TestAuthManagerVerification:
    """Test API key verification."""

    @pytest.mark.asyncio
    async def test_verify_no_key_required(self):
        """Test verification when no key is required."""
        config = SecurityConfig(api_key=None)
        auth = AuthManager(config)

        mock_request = MagicMock()
        result = await auth.verify_api_key(mock_request, api_key=None)

        assert result is None  # No auth required

    @pytest.mark.asyncio
    async def test_verify_missing_key(self):
        """Test verification with missing API key."""
        config = SecurityConfig(api_key="secret")
        auth = AuthManager(config)

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        with pytest.raises(HTTPException) as exc_info:
            await auth.verify_api_key(mock_request, api_key=None)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_invalid_key(self):
        """Test verification with invalid API key."""
        config = SecurityConfig(api_key="secret")
        auth = AuthManager(config)

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        with pytest.raises(HTTPException) as exc_info:
            await auth.verify_api_key(mock_request, api_key="wrong-key")

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_verify_valid_key(self):
        """Test verification with valid API key."""
        config = SecurityConfig(api_key="correct-key")
        auth = AuthManager(config)

        mock_request = MagicMock()
        result = await auth.verify_api_key(mock_request, api_key="correct-key")

        assert result == "correct-key"


class TestGlobalAuthManager:
    """Test global auth manager singleton."""

    def test_get_auth_manager_singleton(self):
        """Test that get_auth_manager returns singleton."""
        # Clear any existing instance
        import nexus.service.auth as auth_module

        auth_module._auth_manager = None

        manager1 = get_auth_manager()
        manager2 = get_auth_manager()

        assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
