"""
NEXUS Hardware Abstraction Layer
=================================

Centralized hardware detection and capability management.
Detects available compute resources (CPU, GPU, accelerators) and provides
optimal device selection without resource hogging.

Supports:
- CPU: Any system (x86, ARM, Apple Silicon)
- GPU: NVIDIA CUDA, AMD ROCm, Apple MPS
- Accelerators: Apple Neural Engine (detected via MPS)

Design Principle: Detect and USE available hardware, but never REQUIRE it.
CPU-only mode always works.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import psutil

logger = logging.getLogger("nexus.hardware")


class DeviceType(Enum):
    """Available device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    ROCM = "rocm"


class DeviceStrategy(Enum):
    """Device selection strategies."""

    AUTO = "auto"  # Best available device
    PREFER_GPU = "prefer_gpu"  # GPU if available, else CPU
    PREFER_CPU = "prefer_cpu"  # Always CPU (for testing/debugging)
    PREFER_MPS = "prefer_mps"  # Apple MPS if available


@dataclass
class CPUCapabilities:
    """CPU hardware capabilities."""

    # Core information
    physical_cores: int
    logical_cores: int
    architecture: str  # x86_64, arm64, armv7l, etc.

    # Frequency (may be unavailable on some platforms)
    frequency_mhz: Optional[float] = None
    max_frequency_mhz: Optional[float] = None

    # Platform info
    processor_name: str = ""
    is_apple_silicon: bool = False
    is_raspberry_pi: bool = False
    is_arm: bool = False

    # Recommended limits
    recommended_threads: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "architecture": self.architecture,
            "frequency_mhz": self.frequency_mhz,
            "max_frequency_mhz": self.max_frequency_mhz,
            "processor_name": self.processor_name,
            "is_apple_silicon": self.is_apple_silicon,
            "is_raspberry_pi": self.is_raspberry_pi,
            "is_arm": self.is_arm,
            "recommended_threads": self.recommended_threads,
        }


@dataclass
class GPUCapabilities:
    """GPU hardware capabilities."""

    # Device info
    device_type: str  # "cuda", "mps", "rocm"
    device_name: str
    device_count: int = 1

    # Memory (in MB)
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0

    # Compute capability (CUDA only)
    compute_capability: Optional[Tuple[int, int]] = None

    # Features
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_int8: bool = False

    # Governance limits
    recommended_memory_limit_percent: float = 50.0
    recommended_utilization_limit: float = 80.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "device_count": self.device_count,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "compute_capability": self.compute_capability,
            "supports_fp16": self.supports_fp16,
            "supports_bf16": self.supports_bf16,
            "supports_int8": self.supports_int8,
            "recommended_memory_limit_percent": self.recommended_memory_limit_percent,
            "recommended_utilization_limit": self.recommended_utilization_limit,
        }


@dataclass
class AcceleratorCapabilities:
    """ML accelerator capabilities (Neural Engine, NPU, etc.)."""

    accelerator_type: str  # "neural_engine", "intel_npu", "none"
    available: bool = False

    # Detected via indirect means (e.g., MPS availability implies Neural Engine)
    detected_via: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "accelerator_type": self.accelerator_type,
            "available": self.available,
            "detected_via": self.detected_via,
        }


@dataclass
class HardwareCapabilities:
    """Complete hardware capabilities for the system."""

    # Components
    cpu: CPUCapabilities
    gpu: Optional[GPUCapabilities] = None
    accelerator: Optional[AcceleratorCapabilities] = None

    # System memory
    system_memory_total_mb: float = 0.0
    system_memory_available_mb: float = 0.0

    # Recommendations
    recommended_device: str = "cpu"
    optimal_precision: str = "fp32"  # fp32, fp16, bf16
    max_batch_size: int = 1

    # Platform
    platform: str = ""
    python_version: str = ""
    torch_version: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "cpu": self.cpu.to_dict(),
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "accelerator": self.accelerator.to_dict() if self.accelerator else None,
            "system_memory_total_mb": self.system_memory_total_mb,
            "system_memory_available_mb": self.system_memory_available_mb,
            "recommended_device": self.recommended_device,
            "optimal_precision": self.optimal_precision,
            "max_batch_size": self.max_batch_size,
            "platform": self.platform,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
        }

    def summary(self) -> str:
        """Human-readable summary of capabilities."""
        lines = [
            f"Platform: {self.platform}",
            f"CPU: {self.cpu.processor_name or self.cpu.architecture} "
            f"({self.cpu.physical_cores} cores)",
        ]
        if self.gpu:
            lines.append(
                f"GPU: {self.gpu.device_name} "
                f"({self.gpu.memory_total_mb:.0f} MB, {self.gpu.device_type})"
            )
        if self.accelerator and self.accelerator.available:
            lines.append(f"Accelerator: {self.accelerator.accelerator_type}")
        lines.append(f"Recommended: {self.recommended_device} @ {self.optimal_precision}")
        return " | ".join(lines)


class HardwareDetector:
    """
    Cross-platform hardware detection.

    Detects CPU, GPU (CUDA/MPS/ROCm), and accelerators.
    Provides recommendations for optimal device usage.
    """

    def __init__(self, strategy: DeviceStrategy = DeviceStrategy.AUTO):
        """
        Initialize hardware detector.

        Args:
            strategy: Device selection strategy
        """
        self.strategy = strategy
        self._capabilities: Optional[HardwareCapabilities] = None
        self._torch_available = self._check_torch()

    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch

            return True
        except ImportError:
            return False

    def detect(self, force_refresh: bool = False) -> HardwareCapabilities:
        """
        Detect all hardware capabilities.

        Args:
            force_refresh: Force re-detection even if cached

        Returns:
            HardwareCapabilities with all detected info
        """
        if self._capabilities is not None and not force_refresh:
            return self._capabilities

        logger.info("Detecting hardware capabilities...")

        # Detect components
        cpu = self._detect_cpu()
        gpu = self._detect_gpu()
        accelerator = self._detect_accelerator()

        # Get system memory
        mem = psutil.virtual_memory()
        system_memory_total_mb = mem.total / 1024 / 1024
        system_memory_available_mb = mem.available / 1024 / 1024

        # Determine recommendations
        recommended_device = self._recommend_device(cpu, gpu)
        optimal_precision = self._recommend_precision(cpu, gpu)
        max_batch_size = self._recommend_batch_size(cpu, gpu, system_memory_available_mb)

        # Platform info
        import sys

        platform_str = f"{platform.system()} {platform.release()}"
        python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        torch_version = "N/A"
        if self._torch_available:
            import torch

            torch_version = torch.__version__

        self._capabilities = HardwareCapabilities(
            cpu=cpu,
            gpu=gpu,
            accelerator=accelerator,
            system_memory_total_mb=system_memory_total_mb,
            system_memory_available_mb=system_memory_available_mb,
            recommended_device=recommended_device,
            optimal_precision=optimal_precision,
            max_batch_size=max_batch_size,
            platform=platform_str,
            python_version=python_version,
            torch_version=torch_version,
        )

        logger.info(f"Hardware detected: {self._capabilities.summary()}")
        return self._capabilities

    def _detect_cpu(self) -> CPUCapabilities:
        """Detect CPU capabilities."""
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1
        architecture = platform.machine()

        # Frequency info
        freq = psutil.cpu_freq()
        frequency_mhz = freq.current if freq else None
        max_frequency_mhz = freq.max if freq else None

        # Processor name
        processor_name = platform.processor() or "Unknown"

        # Platform detection
        is_arm = architecture.lower() in ("arm64", "aarch64", "armv7l", "armv8")
        is_apple_silicon = is_arm and platform.system() == "Darwin"
        is_raspberry_pi = self._check_raspberry_pi()

        # Better processor name for Apple Silicon
        if is_apple_silicon:
            processor_name = self._get_apple_chip_name() or processor_name

        # Recommended threads (leave headroom for system)
        recommended_threads = max(1, physical_cores - 1)
        if is_raspberry_pi:
            recommended_threads = max(1, physical_cores // 2)

        return CPUCapabilities(
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            architecture=architecture,
            frequency_mhz=frequency_mhz,
            max_frequency_mhz=max_frequency_mhz,
            processor_name=processor_name,
            is_apple_silicon=is_apple_silicon,
            is_raspberry_pi=is_raspberry_pi,
            is_arm=is_arm,
            recommended_threads=recommended_threads,
        )

    def _detect_gpu(self) -> Optional[GPUCapabilities]:
        """Detect GPU capabilities (CUDA, MPS, or ROCm)."""
        if not self._torch_available:
            return None

        import torch

        # Check CUDA first (most common)
        if torch.cuda.is_available():
            return self._detect_cuda_gpu()

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return self._detect_mps_gpu()

        # Check ROCm (AMD)
        # ROCm uses the same CUDA API in PyTorch, detected via HIP
        if hasattr(torch.version, "hip") and torch.version.hip:
            return self._detect_rocm_gpu()

        return None

    def _detect_cuda_gpu(self) -> GPUCapabilities:
        """Detect NVIDIA CUDA GPU capabilities."""
        import torch

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)

        # Memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_total_mb = memory_total / 1024 / 1024

        # Free memory (need to allocate to check)
        memory_free = memory_total - torch.cuda.memory_allocated(0)
        memory_free_mb = memory_free / 1024 / 1024

        # Compute capability
        props = torch.cuda.get_device_properties(0)
        compute_capability = (props.major, props.minor)

        # Feature detection
        supports_fp16 = compute_capability >= (7, 0)  # Volta+
        supports_bf16 = compute_capability >= (8, 0)  # Ampere+
        supports_int8 = compute_capability >= (7, 5)  # Turing+

        # Recommended limits (don't hog the GPU)
        recommended_memory_limit = 50.0
        if memory_total_mb > 16000:  # 16GB+, can use more
            recommended_memory_limit = 70.0
        elif memory_total_mb < 4000:  # <4GB, be conservative
            recommended_memory_limit = 40.0

        return GPUCapabilities(
            device_type="cuda",
            device_name=device_name,
            device_count=device_count,
            memory_total_mb=memory_total_mb,
            memory_free_mb=memory_free_mb,
            compute_capability=compute_capability,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            supports_int8=supports_int8,
            recommended_memory_limit_percent=recommended_memory_limit,
            recommended_utilization_limit=80.0,
        )

    def _detect_mps_gpu(self) -> GPUCapabilities:
        """Detect Apple MPS (Metal Performance Shaders) capabilities."""
        import torch

        # MPS doesn't expose detailed memory info easily
        # Use system unified memory as approximation
        mem = psutil.virtual_memory()
        # Apple Silicon shares memory; assume ~50% available for GPU
        estimated_gpu_memory_mb = (mem.total / 1024 / 1024) * 0.5

        # Get chip name for device_name
        device_name = self._get_apple_chip_name() or "Apple Silicon GPU"

        return GPUCapabilities(
            device_type="mps",
            device_name=device_name,
            device_count=1,
            memory_total_mb=estimated_gpu_memory_mb,
            memory_free_mb=estimated_gpu_memory_mb * 0.8,
            compute_capability=None,
            supports_fp16=True,
            supports_bf16=False,  # MPS has limited bf16 support
            supports_int8=False,
            recommended_memory_limit_percent=40.0,  # Conservative for unified memory
            recommended_utilization_limit=70.0,
        )

    def _detect_rocm_gpu(self) -> GPUCapabilities:
        """Detect AMD ROCm GPU capabilities."""
        import torch

        device_count = torch.cuda.device_count()  # ROCm uses CUDA API
        device_name = torch.cuda.get_device_name(0)

        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_total_mb = memory_total / 1024 / 1024
        memory_free_mb = (memory_total - torch.cuda.memory_allocated(0)) / 1024 / 1024

        return GPUCapabilities(
            device_type="rocm",
            device_name=device_name,
            device_count=device_count,
            memory_total_mb=memory_total_mb,
            memory_free_mb=memory_free_mb,
            compute_capability=None,
            supports_fp16=True,
            supports_bf16=False,
            supports_int8=False,
            recommended_memory_limit_percent=50.0,
            recommended_utilization_limit=80.0,
        )

    def _detect_accelerator(self) -> Optional[AcceleratorCapabilities]:
        """Detect ML accelerators (Neural Engine, NPU)."""
        # Apple Neural Engine (detected via MPS availability on Apple Silicon)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return AcceleratorCapabilities(
                accelerator_type="neural_engine",
                available=True,
                detected_via="Apple Silicon detection",
            )

        # Intel NPU (future: could check for Intel NPU via OpenVINO)
        # For now, return None for non-Apple systems
        return None

    def _recommend_device(self, cpu: CPUCapabilities, gpu: Optional[GPUCapabilities]) -> str:
        """Recommend optimal device based on strategy and capabilities."""

        if self.strategy == DeviceStrategy.PREFER_CPU:
            return "cpu"

        if self.strategy == DeviceStrategy.PREFER_MPS:
            if gpu and gpu.device_type == "mps":
                return "mps"
            return "cpu"

        if self.strategy == DeviceStrategy.PREFER_GPU:
            if gpu:
                return gpu.device_type
            return "cpu"

        # AUTO strategy: choose best available
        if gpu:
            # GPU available - check if it's worth using
            if gpu.device_type == "cuda" and gpu.memory_total_mb >= 2000:
                return "cuda"
            if gpu.device_type == "mps":
                return "mps"
            if gpu.device_type == "rocm" and gpu.memory_total_mb >= 2000:
                return "rocm"

        return "cpu"

    def _recommend_precision(self, cpu: CPUCapabilities, gpu: Optional[GPUCapabilities]) -> str:
        """Recommend optimal precision for computation."""
        if gpu:
            if gpu.supports_bf16:
                return "bf16"
            if gpu.supports_fp16:
                return "fp16"

        # CPU: bf16 on Apple Silicon, fp32 elsewhere
        if cpu.is_apple_silicon:
            return "bf16"

        return "fp32"

    def _recommend_batch_size(
        self,
        cpu: CPUCapabilities,
        gpu: Optional[GPUCapabilities],
        available_memory_mb: float,
    ) -> int:
        """Recommend maximum batch size based on memory."""
        if gpu:
            # Estimate based on GPU memory (rough: 500MB per batch item for small model)
            usable_memory = gpu.memory_free_mb * (gpu.recommended_memory_limit_percent / 100)
            batch_size = max(1, int(usable_memory / 500))
            return min(batch_size, 32)  # Cap at 32

        # CPU: based on system memory
        usable_memory = available_memory_mb * 0.3  # Use max 30% of available
        batch_size = max(1, int(usable_memory / 500))

        # Raspberry Pi: be very conservative
        if cpu.is_raspberry_pi:
            return min(batch_size, 2)

        return min(batch_size, 16)

    def _check_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi."""
        if platform.system() != "Linux":
            return False
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "raspberry pi" in f.read().lower()
        except (FileNotFoundError, PermissionError):
            return False

    def _get_apple_chip_name(self) -> Optional[str]:
        """Get Apple Silicon chip name (M1, M2, etc.)."""
        if platform.system() != "Darwin":
            return None
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def get_device(self) -> str:
        """Get recommended device string for PyTorch."""
        caps = self.detect()
        return caps.recommended_device

    def get_torch_device(self):
        """Get PyTorch device object."""
        if not self._torch_available:
            raise RuntimeError("PyTorch not available")
        import torch

        return torch.device(self.get_device())


# Singleton instance for convenience
_default_detector: Optional[HardwareDetector] = None


def get_hardware_detector(strategy: DeviceStrategy = DeviceStrategy.AUTO) -> HardwareDetector:
    """Get or create the default hardware detector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = HardwareDetector(strategy=strategy)
    return _default_detector


def detect_hardware() -> HardwareCapabilities:
    """Convenience function to detect hardware."""
    return get_hardware_detector().detect()


def get_optimal_device() -> str:
    """Convenience function to get optimal device string."""
    return get_hardware_detector().get_device()
