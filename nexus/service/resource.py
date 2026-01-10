"""
Nexus Resource Governor
=======================

Strictly manages CPU, RAM, and thermal usage to ensure Nexus Continuum
remains a good citizen. Enforces the "10% Active / 25% Idle" rule and
monitors system temperature to prevent overheating.
"""

import gc
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.governor")


class ResourceExhaustedError(Exception):
    """Raised when resources are critically exhausted and operation must abort."""

    pass


class ThermalThrottlingError(Exception):
    """Raised when thermal limits are exceeded and operation must pause."""

    pass


class GPUOverloadError(Exception):
    """Raised when GPU resources are critically overloaded."""

    pass


class GPUMonitor:
    """
    Monitor GPU memory and utilization without hogging resources.

    Supports:
    - NVIDIA CUDA via torch.cuda
    - Apple MPS via torch.mps
    - AMD ROCm via torch.cuda (ROCm uses CUDA API)
    """

    def __init__(self):
        self._available = False
        self._device_type: Optional[str] = None
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if GPU monitoring is available."""
        try:
            import torch

            if torch.cuda.is_available():
                self._available = True
                # Check if ROCm or CUDA
                if hasattr(torch.version, "hip") and torch.version.hip:
                    self._device_type = "rocm"
                else:
                    self._device_type = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._available = True
                self._device_type = "mps"
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        """Whether GPU monitoring is available."""
        return self._available

    @property
    def device_type(self) -> Optional[str]:
        """Type of GPU: 'cuda', 'mps', or 'rocm'."""
        return self._device_type

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage in MB.

        Returns:
            Dict with 'used_mb', 'total_mb', 'percent'
        """
        if not self._available:
            return {"used_mb": 0, "total_mb": 0, "percent": 0}

        try:
            import torch

            if self._device_type in ("cuda", "rocm"):
                used = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                percent = (reserved / total * 100) if total > 0 else 0
                return {
                    "used_mb": used,
                    "reserved_mb": reserved,
                    "total_mb": total,
                    "percent": percent,
                }
            elif self._device_type == "mps":
                # MPS has limited memory introspection
                # Use current_allocated as approximation
                if hasattr(torch.mps, "current_allocated_memory"):
                    used = torch.mps.current_allocated_memory() / 1024 / 1024
                else:
                    used = 0
                # Estimate total from system memory (unified memory)
                mem = psutil.virtual_memory()
                total = mem.total / 1024 / 1024 * 0.5  # ~50% for GPU
                percent = (used / total * 100) if total > 0 else 0
                return {
                    "used_mb": used,
                    "total_mb": total,
                    "percent": percent,
                }
        except Exception as e:
            logger.debug(f"GPU memory query failed: {e}")

        return {"used_mb": 0, "total_mb": 0, "percent": 0}

    def get_utilization(self) -> float:
        """
        Get GPU compute utilization percentage.

        Note: Only available for NVIDIA GPUs via nvidia-smi.
        Returns 0 for MPS and when unavailable.
        """
        if not self._available or self._device_type != "cuda":
            return 0.0

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        return 0.0

    def get_temperature(self) -> Optional[float]:
        """
        Get GPU temperature in Celsius.

        Note: Only available for NVIDIA GPUs via nvidia-smi.
        """
        if not self._available or self._device_type != "cuda":
            return None

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        return None

    def should_throttle(
        self, memory_limit_percent: float = 80.0, utilization_limit: float = 90.0
    ) -> bool:
        """
        Check if GPU usage exceeds limits and should throttle.

        Args:
            memory_limit_percent: Max GPU memory usage %
            utilization_limit: Max GPU compute utilization %

        Returns:
            True if should throttle
        """
        if not self._available:
            return False

        mem = self.get_memory_usage()
        if mem["percent"] > memory_limit_percent:
            return True

        util = self.get_utilization()
        if util > utilization_limit:
            return True

        return False

    def get_stats(self) -> Dict:
        """Get comprehensive GPU statistics."""
        if not self._available:
            return {
                "available": False,
                "device_type": None,
            }

        mem = self.get_memory_usage()
        return {
            "available": True,
            "device_type": self._device_type,
            "memory_used_mb": mem.get("used_mb", 0),
            "memory_total_mb": mem.get("total_mb", 0),
            "memory_percent": mem.get("percent", 0),
            "utilization_percent": self.get_utilization(),
            "temperature_celsius": self.get_temperature(),
        }


@dataclass
class ResourceConfig:
    # CPU Usage Limits (percentage)
    active_cpu_limit: float = 10.0
    idle_cpu_limit: float = 25.0

    # RAM Usage Limits (percentage of total system memory)
    active_ram_limit: float = 30.0
    idle_ram_limit: float = 20.0
    critical_ram_limit: float = 30.0  # Emergency abort threshold

    # Thermal limits (degrees Celsius)
    thermal_warning: float = 70.0  # Start aggressive throttling
    thermal_critical: float = 80.0  # Emergency pause

    # Control intervals
    check_interval: float = 1.0  # Seconds between checks
    backoff_factor: float = 1.5  # Multiplier for sleep when violating

    # GPU Usage Limits (only applies when GPU is used)
    gpu_memory_limit_percent: float = 50.0  # Don't hog GPU memory
    gpu_utilization_limit: float = 80.0  # Leave headroom for other apps
    gpu_thermal_warning: float = 75.0  # GPU-specific thermal warning
    gpu_thermal_critical: float = 85.0  # GPU-specific thermal critical


class ThermalMonitor:
    """
    Cross-platform thermal monitoring.

    Supports:
    - Linux: psutil.sensors_temperatures() or /sys/class/thermal/
    - Raspberry Pi: vcgencmd or /sys/class/thermal/
    - macOS: Not supported (returns None gracefully)
    """

    def __init__(self):
        self._system = platform.system().lower()
        self._is_raspberry_pi = self._check_raspberry_pi()
        self._last_temp: Optional[float] = None
        self._available = self._check_availability()

    def _check_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi."""
        if self._system != "linux":
            return False
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "raspberry pi" in f.read().lower()
        except (FileNotFoundError, PermissionError):
            return False

    def _check_availability(self) -> bool:
        """Check if thermal monitoring is available on this platform."""
        if self._system == "darwin":
            # macOS - not supported via Python
            return False
        elif self._system == "linux":
            # Try psutil first
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    return True
            except (AttributeError, OSError):
                pass
            # Try sysfs fallback
            return os.path.exists("/sys/class/thermal/thermal_zone0/temp")
        elif self._system == "windows":
            # Windows - limited support
            return False
        return False

    def get_temperature(self) -> Optional[float]:
        """
        Get current CPU/SoC temperature in Celsius.

        Returns:
            Temperature in Celsius, or None if unavailable.
        """
        if not self._available:
            return None

        temp = None

        if self._system == "linux":
            temp = self._get_linux_temp()

        self._last_temp = temp
        return temp

    def _get_linux_temp(self) -> Optional[float]:
        """Get temperature on Linux systems."""
        # Method 1: Try psutil
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Priority order for sensor packages
                priority = ["coretemp", "k10temp", "cpu_thermal", "acpitz", "cpu-thermal"]
                for package in priority:
                    if package in temps:
                        readings = temps[package]
                        if readings:
                            # Return average of all cores for coretemp/k10temp
                            valid_temps = [r.current for r in readings if r.current > 0]
                            if valid_temps:
                                return sum(valid_temps) / len(valid_temps)
                # Fallback: use first available package
                for package, readings in temps.items():
                    if readings:
                        valid_temps = [r.current for r in readings if r.current > 0]
                        if valid_temps:
                            return sum(valid_temps) / len(valid_temps)
        except (AttributeError, OSError, KeyError):
            pass

        # Method 2: Try sysfs (works on Raspberry Pi and many Linux systems)
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_millidegrees = int(f.read().strip())
                return temp_millidegrees / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            pass

        # Method 3: Try vcgencmd (Raspberry Pi specific)
        if self._is_raspberry_pi:
            try:
                result = subprocess.run(
                    ["vcgencmd", "measure_temp"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    # Output format: temp=45.0'C
                    output = result.stdout.strip()
                    temp_str = output.replace("temp=", "").replace("'C", "")
                    return float(temp_str)
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass

        return None

    def get_status(self, warning_threshold: float = 70.0, critical_threshold: float = 80.0) -> str:
        """
        Get thermal status based on current temperature.

        Returns:
            "normal", "warning", "critical", or "unavailable"
        """
        temp = self.get_temperature()
        if temp is None:
            return "unavailable"
        elif temp >= critical_threshold:
            return "critical"
        elif temp >= warning_threshold:
            return "warning"
        return "normal"

    @property
    def available(self) -> bool:
        """Whether thermal monitoring is available."""
        return self._available

    @property
    def last_temperature(self) -> Optional[float]:
        """Last recorded temperature (cached)."""
        return self._last_temp


class ResourceGovernor:
    """
    Manages CPU, RAM, GPU, and thermal resources to ensure NEXUS is a good citizen.

    Features:
    - Mode-based limits (active vs idle)
    - Thermal monitoring with warning/critical thresholds (CPU and GPU)
    - GPU memory and utilization monitoring
    - Automatic throttling on violations
    - Graceful degradation when sensors unavailable
    - Dynamic resource allocation based on hardware
    """

    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self.process = psutil.Process(os.getpid())
        self.mode: Literal["active", "idle"] = "idle"
        self._consecutive_violations = 0
        self._thermal_monitor = ThermalMonitor()
        self._gpu_monitor = GPUMonitor()

    def set_mode(self, mode: Literal["active", "idle"]):
        """Switch between active (user waiting) and idle (background learning) modes."""
        if mode != self.mode:
            logger.info(f"Resource Governor switching to {mode.upper()} mode")
            self.mode = mode
            self._consecutive_violations = 0

    def check_and_throttle(self) -> bool:
        """
        Check resource usage and sleep if necessary.
        Forces GC if RAM is high. Raises exception if thermal critical.

        Returns:
            True if throttled, False if running normally

        Raises:
            ThermalThrottlingError: If temperature exceeds critical threshold
        """
        throttled = False

        # Determine limits based on mode
        cpu_limit = (
            self.config.active_cpu_limit if self.mode == "active" else self.config.idle_cpu_limit
        )
        ram_limit = (
            self.config.active_ram_limit if self.mode == "active" else self.config.idle_ram_limit
        )

        # 1. Thermal Check (Highest Priority)
        thermal_temp = self._thermal_monitor.get_temperature()
        if thermal_temp is not None:
            if thermal_temp >= self.config.thermal_critical:
                logger.error(
                    f"CRITICAL THERMAL: {thermal_temp:.1f}°C >= {self.config.thermal_critical}°C. "
                    "Emergency pause required!"
                )
                raise ThermalThrottlingError(
                    f"Temperature {thermal_temp:.1f}°C exceeds critical threshold "
                    f"{self.config.thermal_critical}°C"
                )
            elif thermal_temp >= self.config.thermal_warning:
                logger.warning(
                    f"THERMAL WARNING: {thermal_temp:.1f}°C >= {self.config.thermal_warning}°C. "
                    "Aggressive throttling."
                )
                # Sleep longer to allow cooling
                time.sleep(2.0)
                throttled = True

        # 2. GPU Check (if GPU is being used)
        if self._gpu_monitor.available:
            # GPU Thermal Check
            gpu_temp = self._gpu_monitor.get_temperature()
            if gpu_temp is not None:
                if gpu_temp >= self.config.gpu_thermal_critical:
                    logger.error(
                        f"CRITICAL GPU THERMAL: {gpu_temp:.1f}°C >= {self.config.gpu_thermal_critical}°C. "
                        "Emergency pause required!"
                    )
                    raise ThermalThrottlingError(
                        f"GPU temperature {gpu_temp:.1f}°C exceeds critical threshold "
                        f"{self.config.gpu_thermal_critical}°C"
                    )
                elif gpu_temp >= self.config.gpu_thermal_warning:
                    logger.warning(
                        f"GPU THERMAL WARNING: {gpu_temp:.1f}°C >= {self.config.gpu_thermal_warning}°C. "
                        "Throttling GPU usage."
                    )
                    time.sleep(1.5)
                    throttled = True

            # GPU Memory and Utilization Check
            if self._gpu_monitor.should_throttle(
                memory_limit_percent=self.config.gpu_memory_limit_percent,
                utilization_limit=self.config.gpu_utilization_limit,
            ):
                gpu_mem = self._gpu_monitor.get_memory_usage()
                gpu_util = self._gpu_monitor.get_utilization()
                logger.warning(
                    f"GPU OVERLOAD: Memory {gpu_mem['percent']:.1f}% (limit {self.config.gpu_memory_limit_percent}%), "
                    f"Utilization {gpu_util:.1f}% (limit {self.config.gpu_utilization_limit}%). Throttling."
                )
                time.sleep(0.5)
                throttled = True

        # 3. CPU Check
        try:
            current_cpu = self.process.cpu_percent(interval=None)
            # Basic smoothing to avoid blocking call
            if current_cpu <= 0.0:
                # fallback to brief blocking check if we have no prior sample
                current_cpu = self.process.cpu_percent(interval=0.05)
        except Exception:
            current_cpu = 0.0

        # 3. RAM Check
        current_ram_percent = self.process.memory_percent()

        # 5. Critical RAM Check (Emergency GC)
        if current_ram_percent > self.config.critical_ram_limit:
            gc.collect()  # Try to save it first
            new_ram = self.process.memory_percent()
            if new_ram > self.config.critical_ram_limit:
                logger.warning(
                    f"CRITICAL RAM USAGE: {new_ram:.1f}% > {self.config.critical_ram_limit}%. "
                    "Throttling hard."
                )
                # Extended sleep for memory pressure
                time.sleep(1.0)
                throttled = True

        violation = False

        # 6. CPU Violation
        if current_cpu > cpu_limit:
            violation = True
            logger.debug(f"CPU Violation: {current_cpu:.1f}% > {cpu_limit}%")

        # 6. RAM Violation
        if current_ram_percent > ram_limit:
            violation = True
            # Active Reclamation
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
            throttled = True
        else:
            # Decay violation count
            if self._consecutive_violations > 0:
                self._consecutive_violations -= 1

        return throttled

    def get_stats(self) -> Dict:
        """Return current resource statistics including thermal and GPU."""
        thermal_temp = self._thermal_monitor.get_temperature()
        thermal_status = self._thermal_monitor.get_status(
            self.config.thermal_warning,
            self.config.thermal_critical,
        )

        # GPU stats
        gpu_stats = self._gpu_monitor.get_stats()

        return {
            "mode": self.mode,
            "cpu_percent": self.process.cpu_percent(interval=None),
            "ram_percent": self.process.memory_percent(),
            "active_limit_cpu": self.config.active_cpu_limit,
            "idle_limit_cpu": self.config.idle_cpu_limit,
            "violations": self._consecutive_violations,
            # CPU Thermal stats
            "thermal_celsius": thermal_temp,
            "thermal_status": thermal_status,
            "thermal_warning_limit": self.config.thermal_warning,
            "thermal_critical_limit": self.config.thermal_critical,
            "thermal_available": self._thermal_monitor.available,
            # GPU stats
            "gpu": gpu_stats,
            "gpu_memory_limit_percent": self.config.gpu_memory_limit_percent,
            "gpu_utilization_limit": self.config.gpu_utilization_limit,
        }

    @property
    def thermal_available(self) -> bool:
        """Whether thermal monitoring is available on this platform."""
        return self._thermal_monitor.available

    @property
    def gpu_available(self) -> bool:
        """Whether GPU monitoring is available."""
        return self._gpu_monitor.available

    def should_use_gpu(self) -> bool:
        """
        Check if GPU should be used right now.

        Returns False if:
        - GPU is not available
        - GPU memory is over limit
        - GPU utilization is over limit
        - GPU temperature is in warning/critical range
        """
        if not self._gpu_monitor.available:
            return False

        # Check thermal
        gpu_temp = self._gpu_monitor.get_temperature()
        if gpu_temp is not None and gpu_temp >= self.config.gpu_thermal_warning:
            return False

        # Check limits
        if self._gpu_monitor.should_throttle(
            memory_limit_percent=self.config.gpu_memory_limit_percent,
            utilization_limit=self.config.gpu_utilization_limit,
        ):
            return False

        return True
