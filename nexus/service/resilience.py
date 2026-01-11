"""
NEXUS Resilience & Error Recovery
==================================

Production-grade error handling and graceful degradation.
Prevents cascading failures and ensures continuous operation.

Features:
- Circuit breaker pattern
- Retry with exponential backoff
- Timeout protection
- Graceful degradation
- Error categorization and handling
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

# Use centralized logging
from nexus.service.logging_config import get_logger

logger = get_logger("resilience")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure threshold to open circuit
    failure_threshold: int = 5

    # Success threshold to close circuit (from half-open)
    success_threshold: int = 2

    # Timeout before transitioning from open to half-open
    timeout_seconds: float = 60.0

    # Window for counting failures (seconds)
    failure_window: float = 60.0


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject all requests immediately
    - HALF_OPEN: Testing if system recovered, allow limited requests

    Example:
        >>> breaker = CircuitBreaker("inference")
        >>> result = breaker.call(model.predict, input_data)
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name for identification/logging
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state_changed_time = time.time()

        # Track failures in time window
        self.failure_times: list[float] = []

        logger.info(f"CircuitBreaker '{name}' initialized: {self.config}")

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception if function fails
        """
        # Check if we should reject immediately
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Retry after {self._time_until_retry():.1f}s"
                )

        # Attempt execution
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
            self.failure_times.clear()

    def _on_failure(self, exception: Exception) -> None:
        """Handle failed execution."""
        current_time = time.time()

        self.failure_count += 1
        self.failure_times.append(current_time)
        self.last_failure_time = current_time

        # Remove old failures outside window
        self._cleanup_old_failures()

        # Count recent failures
        recent_failures = len(self.failure_times)

        logger.warning(f"Circuit breaker '{self.name}' failure #{recent_failures}: {exception}")

        # Check if we should open circuit
        if self.state == CircuitState.CLOSED and recent_failures >= self.config.failure_threshold:
            self._transition_to_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately reopens circuit
            self._transition_to_open()

    def _cleanup_old_failures(self) -> None:
        """Remove failures outside the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.config.failure_window

        self.failure_times = [t for t in self.failure_times if t > cutoff_time]

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        time_since_open = time.time() - self.state_changed_time
        return time_since_open >= self.config.timeout_seconds

    def _time_until_retry(self) -> float:
        """Calculate time until retry is allowed."""
        time_since_open = time.time() - self.state_changed_time
        return max(0, self.config.timeout_seconds - time_since_open)

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        logger.error(f"Circuit breaker '{self.name}' transitioning to OPEN")

        self.state = CircuitState.OPEN
        self.state_changed_time = time.time()
        self.success_count = 0

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")

        self.state = CircuitState.HALF_OPEN
        self.state_changed_time = time.time()
        self.success_count = 0
        self.failure_count = 0

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")

        self.state = CircuitState.CLOSED
        self.state_changed_time = time.time()
        self.failure_count = 0
        self.success_count = 0
        self.failure_times.clear()

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' manually reset")

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_times.clear()
        self.state_changed_time = time.time()

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": len(self.failure_times),
            "success_count": self.success_count,
            "time_in_state": time.time() - self.state_changed_time,
            "time_until_retry": (
                self._time_until_retry() if self.state == CircuitState.OPEN else None
            ),
        }

    def __repr__(self) -> str:
        return f"CircuitBreaker(name='{self.name}', state={self.state.value})"


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_attempts: int = 3
    initial_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryWithBackoff:
    """
    Retry failed operations with exponential backoff.

    Example:
        >>> retry = RetryWithBackoff()
        >>> result = retry.execute(unreliable_function, arg1, arg2)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()

    def execute(
        self,
        func: Callable[..., T],
        *args,
        retry_on: Optional[tuple[type[Exception], ...]] = None,
        **kwargs,
    ) -> T:
        """
        Execute function with retry and backoff.

        Args:
            func: Function to execute
            *args: Positional arguments
            retry_on: Exception types to retry on (None = all)
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if we should retry this exception type
                if retry_on is not None and not isinstance(e, retry_on):
                    raise

                # Don't retry on last attempt
                if attempt >= self.config.max_attempts - 1:
                    break

                # Calculate delay
                delay = self._calculate_delay(attempt)

                logger.warning(
                    f"Retry attempt {attempt + 1}/{self.config.max_attempts} "
                    f"after {delay:.2f}s delay: {e}"
                )

                time.sleep(delay)

        # All retries exhausted
        logger.error(f"All {self.config.max_attempts} retry attempts failed")
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(
            self.config.initial_delay * (self.config.exponential_base**attempt),
            self.config.max_delay,
        )

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random

            delay *= 0.5 + random.random()

        return delay


class TimeoutProtection:
    """
    Timeout protection for long-running operations.

    Note: Uses threading, not perfect but good enough for our use case.
    """

    @staticmethod
    def execute(
        func: Callable[..., T],
        timeout_seconds: float,
        *args,
        **kwargs,
    ) -> T:
        """
        Execute function with timeout.

        Args:
            func: Function to execute
            timeout_seconds: Maximum execution time
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            TimeoutError: If execution exceeds timeout
        """
        import threading

        result = [None]
        exception = [None]

        def wrapper():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            logger.error(f"Operation timed out after {timeout_seconds}s")
            raise TimeoutError(f"Operation exceeded {timeout_seconds}s timeout")

        if exception[0]:
            raise exception[0]

        return result[0]


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> Callable:
    """
    Decorator for circuit breaker protection.

    Example:
        @circuit_breaker("model_inference")
        def predict(model, input_data):
            return model(input_data)
    """
    breaker = CircuitBreaker(name, config)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        wrapper._circuit_breaker = breaker
        return wrapper

    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    retry_on: Optional[tuple[type[Exception], ...]] = None,
) -> Callable:
    """
    Decorator for retry with exponential backoff.

    Example:
        @retry_on_failure(max_attempts=3)
        def unstable_operation():
            ...
    """
    config = RetryConfig(max_attempts=max_attempts, initial_delay=initial_delay)
    retry_handler = RetryWithBackoff(config)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler.execute(func, *args, retry_on=retry_on, **kwargs)

        return wrapper

    return decorator


def with_timeout(timeout_seconds: float) -> Callable:
    """
    Decorator for timeout protection.

    Example:
        @with_timeout(5.0)
        def slow_operation():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return TimeoutProtection.execute(func, timeout_seconds, *args, **kwargs)

        return wrapper

    return decorator


class GracefulDegradation:
    """
    Graceful degradation when primary system fails.

    Provides fallback responses when main system is unavailable.
    """

    @staticmethod
    def fallback_response(reason: str = "system_unavailable") -> dict[str, Any]:
        """
        Generate fallback response.

        Args:
            reason: Reason for fallback

        Returns:
            Fallback response dictionary
        """
        messages = {
            "system_unavailable": "I'm temporarily unavailable. Please try again in a moment.",
            "overloaded": "I'm experiencing high load. Please try again shortly.",
            "timeout": "Your request took too long to process. Please try a simpler query.",
            "circuit_open": "I'm recovering from errors. Please try again in a few moments.",
        }

        message = messages.get(reason, messages["system_unavailable"])

        return {
            "responded": False,
            "message": message,
            "fallback": True,
            "reason": reason,
            "confidence": 0.0,
        }
