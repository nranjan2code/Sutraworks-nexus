"""
Nexus Resource Governor
=======================

Strictly manages CPU and RAM usage to ensure Nexus Continuum remains a good citizen.
Enforces the "10% Active / 25% Idle" rule.
"""

import time
import psutil
import os
import logging
from dataclasses import dataclass
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.governor")


@dataclass
class ResourceConfig:
    # CPU Usage Limits (percentage)
    active_cpu_limit: float = 10.0
    idle_cpu_limit: float = 25.0

    # RAM Usage Limits (percentage of total system memory)
    active_ram_limit: float = 10.0
    idle_ram_limit: float = 25.0

    # Control intervals
    check_interval: float = 1.0  # Seconds between checks
    backoff_factor: float = 1.5  # Multiplier for sleep when violating


class ResourceGovernor:
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self.process = psutil.Process(os.getpid())
        self.mode: Literal["active", "idle"] = "idle"
        self._consecutive_violations = 0

    def set_mode(self, mode: Literal["active", "idle"]):
        """Switch between active (user waiting) and idle (background learning) modes."""
        if mode != self.mode:
            logger.info(f"Resource Governor switching to {mode.upper()} mode")
            self.mode = mode
            self._consecutive_violations = 0

    def check_and_throttle(self):
        """
        Check resource usage and sleep if necessary to bring average usage down.
        This should be called inside the main processing loops.
        """
        # Determine limits based on mode
        cpu_limit = (
            self.config.active_cpu_limit if self.mode == "active" else self.config.idle_cpu_limit
        )
        ram_limit = (
            self.config.active_ram_limit if self.mode == "active" else self.config.idle_ram_limit
        )

        # Check current usage
        # interval=None is non-blocking but might be spiky.
        # For a background daemon, we might want a small interval or rely on OS average.
        # Here we use a small interval for more accurate instant reading, but it blocks.
        # To avoid blocking the actual work too much, we use interval=None and average manually if needed.
        # But psutil.cpu_percent(interval=None) returns 0.0 on first call.
        try:
            current_cpu = self.process.cpu_percent(interval=0.1)
        except Exception:
            current_cpu = 0.0

        current_ram_percent = self.process.memory_percent()

        violation = False

        # CPU Violation
        if current_cpu > cpu_limit:
            violation = True
            logger.debug(f"CPU Violation: {current_cpu:.1f}% > {cpu_limit}%")

        # RAM Violation (Harder to fix instantly, but we can pause allocation/learning)
        if current_ram_percent > ram_limit:
            violation = True
            logger.warning(f"RAM Violation: {current_ram_percent:.1f}% > {ram_limit}%")

        if violation:
            self._consecutive_violations += 1
            # Sleep to lower average CPU usage
            # The more we violate, the longer we sleep
            sleep_time = 0.1 * (self.config.backoff_factor**self._consecutive_violations)
            sleep_time = min(sleep_time, 5.0)  # Cap sleep at 5 seconds

            logger.debug(f"Throttling for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        else:
            # Decay violation count
            if self._consecutive_violations > 0:
                self._consecutive_violations -= 1

    def get_stats(self):
        """Return current resource statistics."""
        return {
            "mode": self.mode,
            "cpu_percent": self.process.cpu_percent(interval=None),
            "ram_percent": self.process.memory_percent(),
            "active_limit_cpu": self.config.active_cpu_limit,
            "idle_limit_cpu": self.config.idle_cpu_limit,
            "violations": self._consecutive_violations,
        }
