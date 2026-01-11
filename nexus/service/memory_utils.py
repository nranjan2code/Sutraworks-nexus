"""
NEXUS Memory Utilities
=======================

Centralized GPU and system memory cleanup utilities.

Features:
- Explicit GPU memory cleanup
- Decorator for memory-intensive operations
- Context manager for temporary allocations
"""

from __future__ import annotations

import functools
import gc
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar

# Use centralized logging
try:
    from nexus.service.logging_config import get_logger

    logger = get_logger("memory_utils")
except ImportError:
    logger = logging.getLogger("nexus.memory_utils")


# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


def cleanup_gpu_memory(verbose: bool = False) -> dict[str, Any]:
    """
    Clean up GPU memory by emptying CUDA cache.

    Works across CUDA, MPS, and ROCm.

    Args:
        verbose: Log memory stats before/after cleanup

    Returns:
        Dictionary with cleanup statistics
    """
    stats: dict[str, Any] = {
        "gpu_available": False,
        "device_type": None,
        "cleaned": False,
        "memory_freed_mb": 0.0,
    }

    try:
        import torch
    except ImportError:
        logger.debug("PyTorch not available, skipping GPU cleanup")
        return stats

    # Check CUDA (includes ROCm)
    if torch.cuda.is_available():
        stats["gpu_available"] = True
        stats["device_type"] = "cuda"

        if verbose:
            before_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            before_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(
                f"GPU memory before cleanup: "
                f"allocated={before_allocated:.1f}MB, "
                f"reserved={before_reserved:.1f}MB"
            )

        # Run garbage collection first
        gc.collect()

        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Synchronize to ensure cleanup is complete
        torch.cuda.synchronize()

        stats["cleaned"] = True

        if verbose:
            after_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            after_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            stats["memory_freed_mb"] = before_reserved - after_reserved
            logger.info(
                f"GPU memory after cleanup: "
                f"allocated={after_allocated:.1f}MB, "
                f"reserved={after_reserved:.1f}MB, "
                f"freed={stats['memory_freed_mb']:.1f}MB"
            )

    # Check MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        stats["gpu_available"] = True
        stats["device_type"] = "mps"

        # MPS doesn't have explicit cache control like CUDA
        # but we can still run garbage collection
        gc.collect()

        # Empty MPS cache if available
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            stats["cleaned"] = True

        if verbose:
            logger.info("MPS memory cleanup completed")

    else:
        logger.debug("No GPU available, running GC only")
        gc.collect()

    return stats


def cleanup_system_memory() -> dict[str, float]:
    """
    Clean up system memory via garbage collection.

    Returns:
        Dictionary with cleanup statistics
    """
    import psutil

    process = psutil.Process()
    before_mb = process.memory_info().rss / 1024 / 1024

    # Force garbage collection
    gc.collect()

    after_mb = process.memory_info().rss / 1024 / 1024
    freed_mb = before_mb - after_mb

    return {
        "before_mb": before_mb,
        "after_mb": after_mb,
        "freed_mb": max(0.0, freed_mb),
    }


def with_memory_cleanup(
    cleanup_before: bool = False, cleanup_after: bool = True
) -> Callable[[F], F]:
    """
    Decorator that cleans up GPU memory around a function call.

    Args:
        cleanup_before: Clean up before function execution
        cleanup_after: Clean up after function execution

    Returns:
        Decorated function

    Example:
        >>> @with_memory_cleanup(cleanup_after=True)
        ... def train_epoch(model, data):
        ...     # Memory-intensive training code
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if cleanup_before:
                cleanup_gpu_memory(verbose=False)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if cleanup_after:
                    cleanup_gpu_memory(verbose=False)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def memory_cleanup_context(
    cleanup_on_enter: bool = False,
    cleanup_on_exit: bool = True,
    verbose: bool = False,
) -> Generator[None, None, None]:
    """
    Context manager for memory cleanup around operations.

    Args:
        cleanup_on_enter: Clean memory when entering context
        cleanup_on_exit: Clean memory when exiting context
        verbose: Log memory statistics

    Example:
        >>> with memory_cleanup_context(cleanup_on_exit=True):
        ...     # Memory-intensive operations
        ...     process_large_batch(data)
    """
    if cleanup_on_enter:
        cleanup_gpu_memory(verbose=verbose)

    try:
        yield
    finally:
        if cleanup_on_exit:
            cleanup_gpu_memory(verbose=verbose)


def get_memory_stats() -> dict[str, Any]:
    """
    Get comprehensive memory statistics.

    Returns:
        Dictionary with CPU and GPU memory info
    """
    import psutil

    stats: dict[str, Any] = {
        "system": {},
        "process": {},
        "gpu": {},
    }

    # System memory
    mem = psutil.virtual_memory()
    stats["system"] = {
        "total_mb": mem.total / 1024 / 1024,
        "available_mb": mem.available / 1024 / 1024,
        "used_mb": mem.used / 1024 / 1024,
        "percent": mem.percent,
    }

    # Process memory
    process = psutil.Process()
    mem_info = process.memory_info()
    stats["process"] = {
        "rss_mb": mem_info.rss / 1024 / 1024,
        "vms_mb": mem_info.vms / 1024 / 1024,
    }

    # GPU memory
    try:
        import torch

        if torch.cuda.is_available():
            stats["gpu"] = {
                "device_type": "cuda",
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }

            # Get device properties
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                stats["gpu"][f"device_{i}"] = {
                    "name": props.name,
                    "total_memory_mb": props.total_memory / 1024 / 1024,
                }

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            stats["gpu"] = {
                "device_type": "mps",
                "available": True,
            }

    except ImportError:
        stats["gpu"] = {"available": False}

    return stats


def reset_peak_memory_stats() -> None:
    """Reset peak memory tracking for CUDA devices."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
    except ImportError:
        pass
