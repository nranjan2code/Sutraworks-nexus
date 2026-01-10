"""
Hardware Detection Tests
=========================

Tests for the hardware abstraction layer.
"""

import platform
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from nexus.service.hardware import (
    CPUCapabilities,
    GPUCapabilities,
    AcceleratorCapabilities,
    HardwareCapabilities,
    HardwareDetector,
    DeviceStrategy,
    detect_hardware,
    get_optimal_device,
)


class TestCPUCapabilities:
    """Test CPU capability detection."""

    def test_cpu_to_dict(self):
        """Test CPU capabilities serialization."""
        cpu = CPUCapabilities(
            physical_cores=4,
            logical_cores=8,
            architecture="x86_64",
            processor_name="Intel Core i7",
            recommended_threads=3,
        )
        d = cpu.to_dict()

        assert d["physical_cores"] == 4
        assert d["logical_cores"] == 8
        assert d["architecture"] == "x86_64"
        assert d["processor_name"] == "Intel Core i7"


class TestGPUCapabilities:
    """Test GPU capability detection."""

    def test_gpu_to_dict(self):
        """Test GPU capabilities serialization."""
        gpu = GPUCapabilities(
            device_type="cuda",
            device_name="NVIDIA RTX 3080",
            device_count=1,
            memory_total_mb=10240.0,
            memory_free_mb=8000.0,
            compute_capability=(8, 6),
            supports_fp16=True,
            supports_bf16=True,
        )
        d = gpu.to_dict()

        assert d["device_type"] == "cuda"
        assert d["device_name"] == "NVIDIA RTX 3080"
        assert d["memory_total_mb"] == 10240.0
        assert d["compute_capability"] == (8, 6)
        assert d["supports_bf16"] is True


class TestHardwareDetector:
    """Test hardware detection."""

    def test_detector_initialization(self):
        """Test detector initializes without error."""
        detector = HardwareDetector()
        assert detector.strategy == DeviceStrategy.AUTO

    def test_detector_with_strategy(self):
        """Test detector with different strategies."""
        detector = HardwareDetector(strategy=DeviceStrategy.PREFER_CPU)
        assert detector.strategy == DeviceStrategy.PREFER_CPU

    def test_detect_returns_capabilities(self):
        """Test detection returns valid capabilities."""
        caps = detect_hardware()

        assert isinstance(caps, HardwareCapabilities)
        assert caps.cpu is not None
        assert caps.cpu.physical_cores >= 1
        assert caps.cpu.logical_cores >= 1
        assert caps.recommended_device in ("cpu", "cuda", "mps", "rocm")

    def test_detect_cpu_basic(self):
        """Test CPU detection."""
        detector = HardwareDetector()
        cpu = detector._detect_cpu()

        assert cpu.physical_cores >= 1
        assert cpu.logical_cores >= cpu.physical_cores
        assert len(cpu.architecture) > 0
        assert cpu.recommended_threads >= 1

    @patch("platform.system")
    @patch("platform.machine")
    def test_detect_apple_silicon(self, mock_machine, mock_system):
        """Test Apple Silicon detection."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"

        detector = HardwareDetector()
        cpu = detector._detect_cpu()

        assert cpu.is_apple_silicon is True
        assert cpu.is_arm is True

    def test_get_device_returns_string(self):
        """Test get_device returns valid string."""
        device = get_optimal_device()
        assert device in ("cpu", "cuda", "mps", "rocm")

    def test_caching(self):
        """Test that detection is cached."""
        detector = HardwareDetector()
        caps1 = detector.detect()
        caps2 = detector.detect()

        # Should be the same object (cached)
        assert caps1 is caps2

        # Force refresh should return new object
        caps3 = detector.detect(force_refresh=True)
        # Values should be equivalent but may be new object
        assert caps3.cpu.physical_cores == caps1.cpu.physical_cores


class TestDeviceRecommendation:
    """Test device recommendation logic."""

    def test_prefer_cpu_strategy(self):
        """Test PREFER_CPU strategy always returns cpu."""
        detector = HardwareDetector(strategy=DeviceStrategy.PREFER_CPU)
        caps = detector.detect()

        assert caps.recommended_device == "cpu"

    @patch.object(HardwareDetector, "_detect_gpu")
    def test_auto_with_no_gpu(self, mock_detect_gpu):
        """Test AUTO strategy with no GPU falls back to CPU."""
        mock_detect_gpu.return_value = None

        detector = HardwareDetector(strategy=DeviceStrategy.AUTO)
        detector._capabilities = None  # Clear cache

        cpu = detector._detect_cpu()
        device = detector._recommend_device(cpu, None)

        assert device == "cpu"


class TestHardwareCapabilities:
    """Test HardwareCapabilities class."""

    def test_summary(self):
        """Test human-readable summary."""
        cpu = CPUCapabilities(
            physical_cores=4,
            logical_cores=8,
            architecture="x86_64",
            processor_name="Test CPU",
            recommended_threads=3,
        )
        caps = HardwareCapabilities(
            cpu=cpu,
            recommended_device="cpu",
            optimal_precision="fp32",
            max_batch_size=8,
            platform="Test Platform",
        )

        summary = caps.summary()
        assert "Test CPU" in summary or "x86_64" in summary
        assert "4 cores" in summary

    def test_to_dict(self):
        """Test serialization."""
        cpu = CPUCapabilities(
            physical_cores=4,
            logical_cores=8,
            architecture="x86_64",
        )
        caps = HardwareCapabilities(
            cpu=cpu,
            recommended_device="cpu",
        )

        d = caps.to_dict()
        assert "cpu" in d
        assert "recommended_device" in d
        assert d["recommended_device"] == "cpu"


class TestRaspberryPiDetection:
    """Test Raspberry Pi specific detection."""

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_not_raspberry_pi_when_file_missing(self, mock_open):
        """Test non-Pi detection when model file missing."""
        detector = HardwareDetector()
        is_pi = detector._check_raspberry_pi()
        assert is_pi is False

    @patch("platform.system")
    def test_not_raspberry_pi_on_mac(self, mock_system):
        """Test non-Pi on macOS."""
        mock_system.return_value = "Darwin"
        detector = HardwareDetector()
        is_pi = detector._check_raspberry_pi()
        assert is_pi is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
