"""
NEXUS Metrics & Monitoring
===========================

Comprehensive metrics collection for production monitoring.
Tracks performance, resource usage, and system health.

Features:
- Request/response metrics with latency percentiles
- Learning cycle tracking
- Resource utilization (CPU, RAM, GPU)
- FlowingNEXUS-specific metrics (flow depth, convergence)
- Time-windowed aggregations
- Export to monitoring systems (Prometheus, etc.)
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import psutil
import torch

# Use centralized logging
from nexus.service.logging_config import get_logger

logger = get_logger("metrics")


@dataclass
class LatencyStats:
    """Latency statistics."""

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @property
    def mean(self) -> float:
        """Average latency."""
        return self.total / self.count if self.count > 0 else 0.0


@dataclass
class CounterStats:
    """Simple counter statistics."""

    total: int = 0
    success: int = 0
    failure: int = 0
    rate_per_second: float = 0.0

    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        return self.success / self.total if self.total > 0 else 0.0


class MetricsCollector:
    """
    Comprehensive metrics collection for NEXUS.

    Tracks all aspects of system behavior for monitoring,
    debugging, and optimization.

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.record_request(latency=0.15, responded=True, confidence=0.8)
        >>> summary = metrics.get_summary()
    """

    def __init__(
        self,
        window_size: int = 1000,
        enable_resource_tracking: bool = True,
    ):
        """
        Initialize metrics collector.

        Args:
            window_size: Size of rolling window for latency tracking
            enable_resource_tracking: Whether to track CPU/RAM/GPU usage
        """
        self.window_size = window_size
        self.enable_resource_tracking = enable_resource_tracking

        # Request/Response tracking
        self.total_requests = 0
        self.total_responses = 0
        self.total_refusals = 0
        self.total_errors = 0

        # Latency tracking (rolling window)
        self.request_latencies: Deque[float] = deque(maxlen=window_size)
        self.learning_latencies: Deque[float] = deque(maxlen=window_size)

        # Confidence tracking
        self.confidence_scores: Deque[float] = deque(maxlen=window_size)

        # Flow metrics (FlowingNEXUS specific)
        self.flow_depths: Deque[int] = deque(maxlen=window_size)
        self.convergence_flags: Deque[bool] = deque(maxlen=window_size)
        self.flow_energies: Deque[float] = deque(maxlen=window_size)

        # Learning tracking
        self.total_learning_cycles = 0
        self.total_learning_samples = 0
        self.learning_losses: Deque[float] = deque(maxlen=window_size)

        # Domain tracking
        self.domain_counts: Counter = Counter()
        self.domain_confidence: Dict[str, Deque[float]] = {}

        # Resource tracking
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_gpu_memory_mb = 0.0

        # Timing
        self.start_time = time.time()
        self.last_request_time = 0.0

        # Process handle for resource tracking
        if self.enable_resource_tracking:
            self.process = psutil.Process()
        else:
            self.process = None

        logger.info(
            f"MetricsCollector initialized (window={window_size}, "
            f"resource_tracking={enable_resource_tracking})"
        )

    def record_request(
        self,
        latency: float,
        responded: bool,
        confidence: Optional[float] = None,
        flow_depth: Optional[int] = None,
        converged: Optional[bool] = None,
        flow_energy: Optional[float] = None,
        domain: Optional[str] = None,
        error: bool = False,
    ) -> None:
        """
        Record a request.

        Args:
            latency: Request latency in seconds
            responded: Whether NEXUS responded (or refused)
            confidence: Confidence score (0-1)
            flow_depth: Number of flow iterations (FlowingNEXUS)
            converged: Whether flow converged
            flow_energy: Final flow energy/residual
            domain: Domain/topic of request
            error: Whether an error occurred
        """
        self.total_requests += 1
        self.last_request_time = time.time()

        # Track latency
        self.request_latencies.append(latency)

        # Track outcome
        if error:
            self.total_errors += 1
        elif responded:
            self.total_responses += 1
        else:
            self.total_refusals += 1

        # Track confidence
        if confidence is not None:
            self.confidence_scores.append(confidence)

        # Track flow metrics
        if flow_depth is not None:
            self.flow_depths.append(flow_depth)

        if converged is not None:
            self.convergence_flags.append(converged)

        if flow_energy is not None:
            self.flow_energies.append(flow_energy)

        # Track domain
        if domain is not None:
            self.domain_counts[domain] += 1

            if confidence is not None:
                if domain not in self.domain_confidence:
                    self.domain_confidence[domain] = []
                if domain not in self.domain_confidence:
                    self.domain_confidence[domain] = deque(maxlen=self.window_size)
                self.domain_confidence[domain].append(confidence)

    def record_learning_cycle(
        self,
        latency: float,
        num_samples: int,
        loss: Optional[float] = None,
    ) -> None:
        """
        Record a learning cycle.

        Args:
            latency: Learning cycle duration in seconds
            num_samples: Number of samples processed
            loss: Training loss
        """
        self.total_learning_cycles += 1
        self.total_learning_samples += num_samples
        self.learning_latencies.append(latency)

        if loss is not None:
            self.learning_losses.append(loss)

    def update_resource_metrics(self) -> None:
        """Update resource usage metrics."""
        if not self.enable_resource_tracking or self.process is None:
            return

        try:
            # CPU
            cpu_percent = self.process.cpu_percent(interval=None)
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)

            # Memory
            mem_info = self.process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

            # GPU (if available)
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, gpu_memory_mb)

        except Exception as e:
            logger.warning(f"Failed to update resource metrics: {e}")

    def get_latency_stats(self, latencies: Deque[float]) -> LatencyStats:
        """Calculate latency statistics."""
        if not latencies:
            return LatencyStats()

        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)

        return LatencyStats(
            count=count,
            total=sum(sorted_latencies),
            min=sorted_latencies[0],
            max=sorted_latencies[-1],
            p50=sorted_latencies[int(count * 0.50)],
            p95=sorted_latencies[int(count * 0.95)],
            p99=sorted_latencies[int(count * 0.99)],
        )

    def get_request_stats(self) -> CounterStats:
        """Get request statistics."""
        return CounterStats(
            total=self.total_requests,
            success=self.total_responses,
            failure=self.total_errors,
            rate_per_second=self.get_request_rate(),
        )

    def get_request_rate(self) -> float:
        """Calculate requests per second."""
        uptime = time.time() - self.start_time
        return self.total_requests / uptime if uptime > 0 else 0.0

    def get_learning_rate(self) -> float:
        """Calculate learning cycles per second."""
        uptime = time.time() - self.start_time
        return self.total_learning_cycles / uptime if uptime > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dictionary with all metrics
        """
        # Update resource metrics
        self.update_resource_metrics()

        # Calculate statistics
        request_latency = self.get_latency_stats(self.request_latencies)
        learning_latency = self.get_latency_stats(self.learning_latencies)
        request_stats = self.get_request_stats()

        # Flow metrics
        flow_metrics = {}
        if self.flow_depths:
            flow_metrics = {
                "average_depth": sum(self.flow_depths) / len(self.flow_depths),
                "min_depth": min(self.flow_depths),
                "max_depth": max(self.flow_depths),
                "convergence_rate": (
                    sum(self.convergence_flags) / len(self.convergence_flags)
                    if self.convergence_flags
                    else 0.0
                ),
            }

        if self.flow_energies:
            flow_metrics["average_energy"] = sum(self.flow_energies) / len(self.flow_energies)

        # Confidence metrics
        confidence_metrics = {}
        if self.confidence_scores:
            confidence_metrics = {
                "average": sum(self.confidence_scores) / len(self.confidence_scores),
                "min": min(self.confidence_scores),
                "max": max(self.confidence_scores),
            }

        # Domain metrics
        domain_metrics = {}
        if self.domain_counts:
            top_domains = self.domain_counts.most_common(5)
            domain_metrics = {
                "total_domains": len(self.domain_counts),
                "top_domains": [
                    {"domain": domain, "count": count} for domain, count in top_domains
                ],
            }

            if self.domain_confidence:
                domain_metrics["domain_confidence"] = {
                    domain: sum(scores) / len(scores)
                    for domain, scores in self.domain_confidence.items()
                    if scores
                }

        return {
            # System info
            "uptime_seconds": time.time() - self.start_time,
            "last_request_ago": (
                time.time() - self.last_request_time if self.last_request_time > 0 else None
            ),
            # Request metrics
            "requests": {
                "total": self.total_requests,
                "responses": self.total_responses,
                "refusals": self.total_refusals,
                "errors": self.total_errors,
                "success_rate": request_stats.success_rate,
                "refusal_rate": (
                    self.total_refusals / self.total_requests if self.total_requests > 0 else 0.0
                ),
                "rate_per_second": request_stats.rate_per_second,
            },
            # Latency metrics
            "latency": {
                "request": {
                    "mean_ms": request_latency.mean * 1000,
                    "p50_ms": request_latency.p50 * 1000,
                    "p95_ms": request_latency.p95 * 1000,
                    "p99_ms": request_latency.p99 * 1000,
                    "min_ms": request_latency.min * 1000,
                    "max_ms": request_latency.max * 1000,
                },
                "learning": {
                    "mean_ms": learning_latency.mean * 1000,
                    "p95_ms": learning_latency.p95 * 1000,
                },
            },
            # Learning metrics
            "learning": {
                "total_cycles": self.total_learning_cycles,
                "total_samples": self.total_learning_samples,
                "rate_per_second": self.get_learning_rate(),
                "average_loss": (
                    sum(self.learning_losses) / len(self.learning_losses)
                    if self.learning_losses
                    else None
                ),
            },
            # Flow metrics (FlowingNEXUS)
            "flow": flow_metrics,
            # Confidence metrics
            "confidence": confidence_metrics,
            # Domain metrics
            "domains": domain_metrics,
            # Resource metrics
            "resources": {
                "peak_memory_mb": self.peak_memory_mb,
                "peak_cpu_percent": self.peak_cpu_percent,
                "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
                "current_memory_mb": (
                    self.process.memory_info().rss / 1024 / 1024 if self.process else 0.0
                ),
                "current_cpu_percent": (
                    self.process.cpu_percent(interval=None) if self.process else 0.0
                ),
            },
        }

    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        summary = self.get_summary()

        metrics = []

        # Request metrics
        metrics.append(f"nexus_requests_total {summary['requests']['total']}")
        metrics.append(f"nexus_responses_total {summary['requests']['responses']}")
        metrics.append(f"nexus_refusals_total {summary['requests']['refusals']}")
        metrics.append(f"nexus_errors_total {summary['requests']['errors']}")
        metrics.append(f"nexus_request_rate {summary['requests']['rate_per_second']}")

        # Latency metrics
        metrics.append(f"nexus_request_latency_p50_ms {summary['latency']['request']['p50_ms']}")
        metrics.append(f"nexus_request_latency_p95_ms {summary['latency']['request']['p95_ms']}")
        metrics.append(f"nexus_request_latency_p99_ms {summary['latency']['request']['p99_ms']}")

        # Learning metrics
        metrics.append(f"nexus_learning_cycles_total {summary['learning']['total_cycles']}")
        metrics.append(f"nexus_learning_samples_total {summary['learning']['total_samples']}")

        # Flow metrics
        if summary["flow"]:
            metrics.append(f"nexus_flow_depth_average {summary['flow'].get('average_depth', 0)}")
            metrics.append(
                f"nexus_flow_convergence_rate {summary['flow'].get('convergence_rate', 0)}"
            )

        # Resource metrics
        metrics.append(f"nexus_memory_peak_mb {summary['resources']['peak_memory_mb']}")
        metrics.append(f"nexus_cpu_peak_percent {summary['resources']['peak_cpu_percent']}")

        return "\n".join(metrics)

    def reset(self) -> None:
        """Reset all metrics (use with caution!)."""
        logger.warning("Resetting all metrics")

        self.total_requests = 0
        self.total_responses = 0
        self.total_refusals = 0
        self.total_errors = 0

        self.request_latencies.clear()
        self.learning_latencies.clear()
        self.confidence_scores.clear()
        self.flow_depths.clear()
        self.convergence_flags.clear()
        self.flow_energies.clear()

        self.total_learning_cycles = 0
        self.total_learning_samples = 0
        self.learning_losses.clear()

        self.domain_counts.clear()
        self.domain_confidence.clear()

        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_gpu_memory_mb = 0.0

        self.start_time = time.time()
        self.last_request_time = 0.0

    def __repr__(self) -> str:
        return (
            f"MetricsCollector(requests={self.total_requests}, "
            f"learning_cycles={self.total_learning_cycles})"
        )


class HealthCheck:
    """
    Health check for NEXUS system.

    Determines if system is healthy based on metrics.
    """

    def __init__(
        self,
        max_error_rate: float = 0.1,  # 10% errors
        max_refusal_rate: float = 0.5,  # 50% refusals
        max_latency_p95_ms: float = 5000,  # 5 seconds
        min_convergence_rate: float = 0.8,  # 80% convergence
    ):
        """
        Initialize health check.

        Args:
            max_error_rate: Maximum acceptable error rate
            max_refusal_rate: Maximum acceptable refusal rate
            max_latency_p95_ms: Maximum acceptable P95 latency
            min_convergence_rate: Minimum acceptable convergence rate
        """
        self.max_error_rate = max_error_rate
        self.max_refusal_rate = max_refusal_rate
        self.max_latency_p95_ms = max_latency_p95_ms
        self.min_convergence_rate = min_convergence_rate

    def check_health(self, metrics: MetricsCollector) -> Dict[str, Any]:
        """
        Check system health.

        Returns:
            Health status with details
        """
        summary = metrics.get_summary()

        # Calculate health indicators
        error_rate = (
            summary["requests"]["errors"] / summary["requests"]["total"]
            if summary["requests"]["total"] > 0
            else 0.0
        )

        refusal_rate = summary["requests"]["refusal_rate"]
        latency_p95 = summary["latency"]["request"]["p95_ms"]

        convergence_rate = summary["flow"].get("convergence_rate", 1.0) if summary["flow"] else 1.0

        # Determine health status
        issues = []

        if error_rate > self.max_error_rate:
            issues.append(f"High error rate: {error_rate:.2%} > {self.max_error_rate:.2%}")

        if refusal_rate > self.max_refusal_rate:
            issues.append(f"High refusal rate: {refusal_rate:.2%} > {self.max_refusal_rate:.2%}")

        if latency_p95 > self.max_latency_p95_ms:
            issues.append(f"High latency: {latency_p95:.0f}ms > {self.max_latency_p95_ms:.0f}ms")

        if convergence_rate < self.min_convergence_rate:
            issues.append(
                f"Low convergence rate: {convergence_rate:.2%} < {self.min_convergence_rate:.2%}"
            )

        healthy = len(issues) == 0

        return {
            "healthy": healthy,
            "status": "healthy" if healthy else "degraded",
            "issues": issues,
            "metrics": {
                "error_rate": error_rate,
                "refusal_rate": refusal_rate,
                "latency_p95_ms": latency_p95,
                "convergence_rate": convergence_rate,
            },
        }
