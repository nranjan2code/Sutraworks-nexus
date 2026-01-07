"""
NEXUS Memory Manager
====================

Memory management for long-running processes.
Prevents memory leaks and ensures stable operation over days/weeks.

Features:
- Periodic garbage collection
- PyTorch cache management
- Memory leak detection
- Replay buffer management
- Memory usage monitoring and alerts
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psutil
import torch

logger = logging.getLogger("nexus.memory")


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    # Cleanup intervals
    gc_interval_seconds: float = 600.0  # 10 minutes
    cache_cleanup_interval_seconds: float = 300.0  # 5 minutes

    # Memory thresholds (MB)
    warning_threshold_mb: float = 2000.0  # 2 GB
    critical_threshold_mb: float = 4000.0  # 4 GB

    # Replay buffer limits
    max_replay_buffer_size: int = 2048
    replay_buffer_cleanup_threshold: float = 0.9  # Cleanup at 90% full

    # History limits
    max_thought_history: int = 100
    max_metric_history: int = 1000

    # Enable aggressive cleanup under memory pressure
    aggressive_cleanup: bool = True


class MemoryManager:
    """
    Manages memory for long-running NEXUS processes.

    Prevents memory leaks, manages caches, and ensures
    stable operation over extended periods.

    Example:
        >>> manager = MemoryManager()
        >>> manager.periodic_cleanup()  # Call regularly
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory manager.

        Args:
            config: Memory management configuration
        """
        self.config = config or MemoryConfig()

        self.process = psutil.Process()
        self.last_gc_time = time.time()
        self.last_cache_cleanup_time = time.time()

        # Memory tracking
        self.peak_memory_mb = 0.0
        self.memory_samples: list[float] = []

        # Leak detection
        self.baseline_memory_mb: Optional[float] = None
        self.memory_growth_rate_mb_per_hour: float = 0.0

        logger.info(f"MemoryManager initialized: {self.config}")

    def periodic_cleanup(self) -> Dict[str, Any]:
        """
        Perform periodic memory cleanup.

        Should be called regularly (e.g., every iteration of daemon loop).

        Returns:
            Cleanup statistics
        """
        current_time = time.time()
        stats = {}

        # Check current memory usage
        current_memory_mb = self._get_memory_usage_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)

        # Garbage collection
        if current_time - self.last_gc_time >= self.config.gc_interval_seconds:
            gc_stats = self._run_garbage_collection()
            stats["gc"] = gc_stats
            self.last_gc_time = current_time

        # Cache cleanup
        if (
            current_time - self.last_cache_cleanup_time
            >= self.config.cache_cleanup_interval_seconds
        ):
            cache_stats = self._cleanup_caches()
            stats["cache"] = cache_stats
            self.last_cache_cleanup_time = current_time

        # Check memory pressure
        memory_status = self._check_memory_pressure(current_memory_mb)
        stats["memory_status"] = memory_status

        # Aggressive cleanup if needed
        if memory_status["level"] == "critical" and self.config.aggressive_cleanup:
            logger.warning("Critical memory pressure - running aggressive cleanup")
            aggressive_stats = self._aggressive_cleanup()
            stats["aggressive_cleanup"] = aggressive_stats

        return stats

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        return mem_info.rss / 1024 / 1024

    def _run_garbage_collection(self) -> Dict[str, Any]:
        """Run Python garbage collection."""
        logger.debug("Running garbage collection")

        mem_before = self._get_memory_usage_mb()

        # Run multiple generations
        collected = {
            "gen0": gc.collect(0),
            "gen1": gc.collect(1),
            "gen2": gc.collect(2),
        }

        mem_after = self._get_memory_usage_mb()
        freed_mb = mem_before - mem_after

        logger.info(
            f"GC completed: freed {freed_mb:.2f} MB "
            f"(collected: {sum(collected.values())} objects)"
        )

        return {
            "freed_mb": freed_mb,
            "objects_collected": sum(collected.values()),
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
        }

    def _cleanup_caches(self) -> Dict[str, Any]:
        """Clean up PyTorch and system caches."""
        logger.debug("Cleaning up caches")

        stats = {}

        # PyTorch CUDA cache
        if torch.cuda.is_available():
            cuda_mem_before = torch.cuda.memory_allocated() / 1024 / 1024
            torch.cuda.empty_cache()
            cuda_mem_after = torch.cuda.memory_allocated() / 1024 / 1024
            cuda_freed = cuda_mem_before - cuda_mem_after

            logger.info(f"CUDA cache cleared: freed {cuda_freed:.2f} MB")

            stats["cuda"] = {
                "freed_mb": cuda_freed,
                "allocated_mb": cuda_mem_after,
            }

        # PyTorch CPU cache (if using MPS on Mac)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have explicit cache clearing yet
            pass

        return stats

    def _check_memory_pressure(self, current_memory_mb: float) -> Dict[str, Any]:
        """Check if memory usage is approaching limits."""
        if current_memory_mb >= self.config.critical_threshold_mb:
            level = "critical"
            logger.error(
                f"CRITICAL memory usage: {current_memory_mb:.0f} MB "
                f">= {self.config.critical_threshold_mb:.0f} MB"
            )

        elif current_memory_mb >= self.config.warning_threshold_mb:
            level = "warning"
            logger.warning(
                f"High memory usage: {current_memory_mb:.0f} MB "
                f">= {self.config.warning_threshold_mb:.0f} MB"
            )

        else:
            level = "normal"

        # Calculate memory pressure (0-1)
        pressure = min(current_memory_mb / self.config.critical_threshold_mb, 1.0)

        return {
            "level": level,
            "current_mb": current_memory_mb,
            "peak_mb": self.peak_memory_mb,
            "pressure": pressure,
            "warning_threshold_mb": self.config.warning_threshold_mb,
            "critical_threshold_mb": self.config.critical_threshold_mb,
        }

    def _aggressive_cleanup(self) -> Dict[str, Any]:
        """Aggressive cleanup under memory pressure."""
        stats = {}

        # Force garbage collection multiple times
        for i in range(3):
            gc_stats = self._run_garbage_collection()
            stats[f"gc_round_{i}"] = gc_stats

        # Clear PyTorch cache
        cache_stats = self._cleanup_caches()
        stats["cache"] = cache_stats

        # Additional cleanup for Python internals
        # Clear weak references
        gc.collect()

        total_freed = sum(
            s.get("freed_mb", 0) for s in stats.values() if isinstance(s, dict)
        )

        logger.info(f"Aggressive cleanup completed: freed {total_freed:.2f} MB")

        return stats

    def cleanup_replay_buffer(self, replay_buffer: list) -> Dict[str, Any]:
        """
        Clean up replay buffer to prevent unbounded growth.

        Args:
            replay_buffer: Replay buffer to clean

        Returns:
            Cleanup statistics
        """
        buffer_size = len(replay_buffer)

        if buffer_size == 0:
            return {"cleaned": 0, "remaining": 0}

        # Check if cleanup needed
        fill_ratio = buffer_size / self.config.max_replay_buffer_size

        if fill_ratio < self.config.replay_buffer_cleanup_threshold:
            return {"cleaned": 0, "remaining": buffer_size}

        # Calculate how many to remove
        target_size = int(self.config.max_replay_buffer_size * 0.75)
        to_remove = max(0, buffer_size - target_size)

        if to_remove > 0:
            # Remove oldest entries (from beginning)
            del replay_buffer[:to_remove]

            logger.info(
                f"Cleaned replay buffer: removed {to_remove} entries "
                f"({buffer_size} -> {len(replay_buffer)})"
            )

        return {
            "cleaned": to_remove,
            "remaining": len(replay_buffer),
            "original_size": buffer_size,
        }

    def trim_history(self, history_list: list, max_size: int) -> int:
        """
        Trim a history list to maximum size.

        Args:
            history_list: List to trim
            max_size: Maximum allowed size

        Returns:
            Number of items removed
        """
        if len(history_list) <= max_size:
            return 0

        to_remove = len(history_list) - max_size
        del history_list[:to_remove]

        return to_remove

    def detect_memory_leak(self) -> Optional[Dict[str, Any]]:
        """
        Detect potential memory leaks.

        Returns:
            Leak detection results, or None if no leak detected
        """
        current_memory_mb = self._get_memory_usage_mb()
        self.memory_samples.append(current_memory_mb)

        # Keep only recent samples (last hour at 1 sample per 10 seconds = 360 samples)
        if len(self.memory_samples) > 360:
            self.memory_samples.pop(0)

        # Need enough samples to detect trend
        if len(self.memory_samples) < 60:
            return None

        # Set baseline if not set
        if self.baseline_memory_mb is None:
            self.baseline_memory_mb = sum(self.memory_samples[:10]) / 10

        # Calculate growth rate (simple linear regression)
        n = len(self.memory_samples)
        x = list(range(n))
        y = self.memory_samples

        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator

            # Convert slope to MB per hour
            # (slope is MB per sample, samples every 10s = 360 samples per hour)
            mb_per_hour = slope * 360

            self.memory_growth_rate_mb_per_hour = mb_per_hour

            # Detect leak: growth > 10 MB/hour
            if mb_per_hour > 10.0:
                logger.warning(
                    f"Potential memory leak detected: {mb_per_hour:.2f} MB/hour growth"
                )

                return {
                    "detected": True,
                    "growth_rate_mb_per_hour": mb_per_hour,
                    "current_memory_mb": current_memory_mb,
                    "baseline_memory_mb": self.baseline_memory_mb,
                    "samples_analyzed": n,
                }

        return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_memory_mb = self._get_memory_usage_mb()

        # System memory
        system_memory = psutil.virtual_memory()

        # CUDA memory (if available)
        cuda_stats = {}
        if torch.cuda.is_available():
            cuda_stats = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }

        return {
            "process": {
                "current_mb": current_memory_mb,
                "peak_mb": self.peak_memory_mb,
                "growth_rate_mb_per_hour": self.memory_growth_rate_mb_per_hour,
            },
            "system": {
                "total_mb": system_memory.total / 1024 / 1024,
                "available_mb": system_memory.available / 1024 / 1024,
                "percent_used": system_memory.percent,
            },
            "cuda": cuda_stats,
            "thresholds": {
                "warning_mb": self.config.warning_threshold_mb,
                "critical_mb": self.config.critical_threshold_mb,
            },
        }

    def reset_peak_tracking(self) -> None:
        """Reset peak memory tracking."""
        self.peak_memory_mb = self._get_memory_usage_mb()
        logger.info("Peak memory tracking reset")

    def __repr__(self) -> str:
        current_mb = self._get_memory_usage_mb()
        return f"MemoryManager(current={current_mb:.0f}MB, peak={self.peak_memory_mb:.0f}MB)"
