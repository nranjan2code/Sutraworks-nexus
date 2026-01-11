"""
NEXUS Configuration Validation
===============================

Pydantic-based configuration with runtime validation.

Features:
- Field validators for critical parameters
- Environment variable parsing
- Type-safe configuration objects
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# Try to import Pydantic, fall back to basic validation
try:
    from pydantic import BaseModel, Field, field_validator, model_validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Create stub classes for graceful degradation
    class BaseModel:  # type: ignore
        pass

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return kwargs.get("default")

    def field_validator(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def model_validator(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


if PYDANTIC_AVAILABLE:

    class ServiceConfig(BaseModel):
        """Service layer configuration with validation."""

        # Server settings
        host: str = Field(default="0.0.0.0", description="Server host")
        port: int = Field(default=8000, ge=1, le=65535, description="Server port")

        # Resource limits
        active_cpu_limit: float = Field(
            default=10.0, ge=1.0, le=100.0, description="CPU % limit when actively processing"
        )
        idle_cpu_limit: float = Field(
            default=25.0, ge=1.0, le=100.0, description="CPU % limit when idle/dreaming"
        )
        gpu_memory_limit_percent: float = Field(
            default=50.0, ge=10.0, le=90.0, description="GPU memory % limit"
        )
        gpu_utilization_limit: float = Field(
            default=80.0, ge=10.0, le=100.0, description="GPU utilization % limit"
        )

        # Thermal limits
        thermal_warning_celsius: float = Field(
            default=70.0, ge=40.0, le=95.0, description="Temperature warning threshold"
        )
        thermal_critical_celsius: float = Field(
            default=80.0, ge=50.0, le=100.0, description="Temperature critical threshold"
        )

        # Security
        api_key: Optional[str] = Field(default=None, description="API key for authentication")
        rate_limit_rpm: int = Field(
            default=60, ge=1, le=10000, description="Rate limit: requests per minute"
        )

        # Daemon settings
        checkpoint_interval_seconds: int = Field(
            default=300, ge=30, le=3600, description="Checkpoint save interval"
        )
        max_history_size: int = Field(
            default=100, ge=10, le=10000, description="Maximum history entries to keep"
        )

        @field_validator("thermal_critical_celsius")
        @classmethod
        def critical_above_warning(cls, v: float, info: Any) -> float:
            """Ensure critical temp is above warning temp."""
            warning = info.data.get("thermal_warning_celsius", 70.0)
            if v <= warning:
                raise ValueError(f"critical temperature ({v}) must be above warning ({warning})")
            return v

        @classmethod
        def from_env(cls) -> "ServiceConfig":
            """Load configuration from environment variables."""
            return cls(
                host=os.getenv("NEXUS_HOST", "0.0.0.0"),
                port=_get_env_int("NEXUS_PORT", 8000),
                active_cpu_limit=_get_env_float("NEXUS_ACTIVE_CPU_LIMIT", 10.0),
                idle_cpu_limit=_get_env_float("NEXUS_IDLE_CPU_LIMIT", 25.0),
                gpu_memory_limit_percent=_get_env_float("NEXUS_GPU_MEMORY_LIMIT", 50.0),
                gpu_utilization_limit=_get_env_float("NEXUS_GPU_UTIL_LIMIT", 80.0),
                thermal_warning_celsius=_get_env_float("NEXUS_THERMAL_WARNING", 70.0),
                thermal_critical_celsius=_get_env_float("NEXUS_THERMAL_CRITICAL", 80.0),
                api_key=os.getenv("NEXUS_API_KEY"),
                rate_limit_rpm=_get_env_int("NEXUS_RATE_LIMIT_RPM", 60),
                checkpoint_interval_seconds=_get_env_int("NEXUS_CHECKPOINT_INTERVAL", 300),
                max_history_size=_get_env_int("NEXUS_MAX_HISTORY", 100),
            )

    class ModelConfig(BaseModel):
        """Model configuration with validation."""

        # Model size
        d_model: int = Field(default=512, ge=64, le=8192, description="Model dimension")
        d_latent: int = Field(default=256, ge=32, le=4096, description="Latent dimension")
        n_heads: int = Field(default=8, ge=1, le=128, description="Number of attention heads")
        n_layers: int = Field(default=6, ge=1, le=128, description="Number of layers")

        # Vocabulary
        vocab_size: int = Field(default=50262, ge=100, le=500000, description="Vocabulary size")
        max_seq_len: int = Field(default=8192, ge=64, le=1000000, description="Max sequence length")

        # State space
        ssm_d_state: int = Field(default=16, ge=4, le=256, description="SSM state dimension")
        ssm_d_conv: int = Field(default=4, ge=2, le=32, description="SSM convolution size")

        # Training
        dropout: float = Field(default=0.1, ge=0.0, le=0.9, description="Dropout rate")

        @field_validator("d_model")
        @classmethod
        def d_model_divisible_by_heads(cls, v: int, info: Any) -> int:
            """Ensure d_model is divisible by n_heads."""
            n_heads = info.data.get("n_heads", 8)
            if v % n_heads != 0:
                raise ValueError(f"d_model ({v}) must be divisible by n_heads ({n_heads})")
            return v

        @field_validator("d_latent")
        @classmethod
        def d_latent_smaller_than_d_model(cls, v: int, info: Any) -> int:
            """Ensure d_latent is <= d_model."""
            d_model = info.data.get("d_model", 512)
            if v > d_model:
                raise ValueError(f"d_latent ({v}) should be <= d_model ({d_model})")
            return v

else:
    # Fallback dataclass-based config without Pydantic

    @dataclass
    class ServiceConfig:  # type: ignore
        """Service configuration (basic validation)."""

        host: str = "0.0.0.0"
        port: int = 8000
        active_cpu_limit: float = 10.0
        idle_cpu_limit: float = 25.0
        gpu_memory_limit_percent: float = 50.0
        gpu_utilization_limit: float = 80.0
        thermal_warning_celsius: float = 70.0
        thermal_critical_celsius: float = 80.0
        api_key: Optional[str] = None
        rate_limit_rpm: int = 60
        checkpoint_interval_seconds: int = 300
        max_history_size: int = 100

        def __post_init__(self) -> None:
            """Basic validation."""
            if not (1 <= self.port <= 65535):
                raise ValueError(f"port must be 1-65535, got {self.port}")
            if self.thermal_critical_celsius <= self.thermal_warning_celsius:
                raise ValueError("critical temp must be above warning temp")

        @classmethod
        def from_env(cls) -> "ServiceConfig":
            """Load from environment."""
            return cls(
                host=os.getenv("NEXUS_HOST", "0.0.0.0"),
                port=_get_env_int("NEXUS_PORT", 8000),
                api_key=os.getenv("NEXUS_API_KEY"),
            )

    @dataclass
    class ModelConfig:  # type: ignore
        """Model configuration (basic validation)."""

        d_model: int = 512
        d_latent: int = 256
        n_heads: int = 8
        n_layers: int = 6
        vocab_size: int = 50262
        max_seq_len: int = 8192
        ssm_d_state: int = 16
        ssm_d_conv: int = 4
        dropout: float = 0.1


def validate_config(config: Any) -> list[str]:
    """
    Validate configuration and return list of issues.

    Args:
        config: Configuration object to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues: list[str] = []

    # Check for common issues
    if hasattr(config, "d_model") and hasattr(config, "n_heads"):
        if config.d_model % config.n_heads != 0:
            issues.append(f"d_model ({config.d_model}) not divisible by n_heads ({config.n_heads})")

    if hasattr(config, "thermal_warning_celsius") and hasattr(config, "thermal_critical_celsius"):
        if config.thermal_critical_celsius <= config.thermal_warning_celsius:
            issues.append("thermal_critical must be above thermal_warning")

    if hasattr(config, "dropout"):
        if not 0.0 <= config.dropout <= 0.9:
            issues.append(f"dropout ({config.dropout}) should be 0.0-0.9")

    return issues


# Module-level singleton for service config
_service_config: Optional[ServiceConfig] = None


def get_service_config() -> ServiceConfig:
    """Get or create the global service configuration."""
    global _service_config
    if _service_config is None:
        _service_config = ServiceConfig.from_env()
    return _service_config
