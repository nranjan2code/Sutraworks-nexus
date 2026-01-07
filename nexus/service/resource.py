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


class ResourceExhaustedError(Exception):
    """Raised when resources are critically exhausted and operation must abort."""

    pass


@dataclass
class ResourceConfig:
    # CPU Usage Limits (percentage)
    active_cpu_limit: float = 10.0
    idle_cpu_limit: float = 25.0

    # RAM Usage Limits (percentage of total system memory)
    active_ram_limit: float = 30.0
    idle_ram_limit: float = 20.0
    critical_ram_limit: float = 30.0  # Emergency abort threshold

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
        Check resource usage and sleep if necessary.
        Forces GC if RAM is high. Raises exception if critical.
        """
        # Determine limits based on mode
        cpu_limit = (
            self.config.active_cpu_limit if self.mode == "active" else self.config.idle_cpu_limit
        )
        ram_limit = (
            self.config.active_ram_limit if self.mode == "active" else self.config.idle_ram_limit
        )

        try:
            current_cpu = self.process.cpu_percent(interval=None)
            # Basic smoothing to avoid blocking call
            if current_cpu <= 0.0:
                # fallback to brief blocking check if we have no prior sample
                current_cpu = self.process.cpu_percent(interval=0.05)
        except Exception:
            current_cpu = 0.0

        current_ram_percent = self.process.memory_percent()

        # 1. Critical Check (Emergency Abort)
        if current_ram_percent > self.config.critical_ram_limit:
            import gc

            gc.collect()  # Try to save it first
            new_ram = self.process.memory_percent()
            if new_ram > self.config.critical_ram_limit:
                logger.warning(
                    f"CRITICAL RAM USAGE: {new_ram:.1f}% > {self.config.critical_ram_limit}%. Throttling."
                )
                # No abort, just let it fall through to violation logic which throttles

        violation = False

        # 2. CPU Violation
        if current_cpu > cpu_limit:
            violation = True
            logger.debug(f"CPU Violation: {current_cpu:.1f}% > {cpu_limit}%")

        # 3. RAM Violation
        if current_ram_percent > ram_limit:
            violation = True
            # Active Reclamation
            import gc

            cleaned = gc.collect()
            if cleaned > 0:
                logger.debug(f"Governor forced GC: collected {cleaned} objects")

        if violation:
            self._consecutive_violations += 1
            # Sleep to lower average CPU usage
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
