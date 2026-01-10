import unittest
import torch
import time
from unittest.mock import MagicMock, patch, PropertyMock
from nexus.core.flowing import FlowingNEXUS, FlowingConfig
from nexus.service.resource import (
    ResourceGovernor,
    ResourceConfig,
    ResourceExhaustedError,
    ThermalThrottlingError,
    ThermalMonitor,
)


class TestGovernance(unittest.TestCase):
    def test_inner_loop_callback(self):
        """Verify step_callback is called inside FlowingNEXUS loop."""
        config = FlowingConfig(d_model=16, d_latent=8, max_flow_steps=5)
        model = FlowingNEXUS(config)

        callback = MagicMock()
        input_tensor = torch.randint(0, 10, (1, 5))

        model(input_tensor, step_callback=callback)

        # Should be called at least once per step
        self.assertEqual(callback.call_count, 5)

    @patch("nexus.service.resource.time.sleep")
    @patch("nexus.service.resource.psutil.Process")
    def test_governor_throttling(self, mock_process, mock_sleep):
        """Verify governor throttles when CPU is high."""
        gov = ResourceGovernor(ResourceConfig(active_cpu_limit=50.0))
        gov.set_mode("active")

        # Mock high CPU
        mock_process.return_value.cpu_percent.return_value = 80.0
        mock_process.return_value.memory_percent.return_value = 10.0

        # Reinitialize with mocked process
        gov.process = mock_process.return_value

        # Mock thermal monitor to be unavailable
        gov._thermal_monitor._available = False

        result = gov.check_and_throttle()

        # Should have slept (throttled)
        mock_sleep.assert_called()
        self.assertTrue(result)

    @patch("nexus.service.resource.time.sleep")
    @patch("nexus.service.resource.psutil.Process")
    def test_critical_ram_throttles(self, mock_process, mock_sleep):
        """Verify governor throttles (not aborts) on critical RAM."""
        gov = ResourceGovernor(ResourceConfig(critical_ram_limit=90.0))

        # Mock critical RAM
        mock_process.return_value.cpu_percent.return_value = 10.0
        mock_process.return_value.memory_percent.return_value = 95.0

        # Reinitialize with mocked process
        gov.process = mock_process.return_value

        # Mock thermal monitor to be unavailable
        gov._thermal_monitor._available = False

        # Should throttle, not raise exception
        result = gov.check_and_throttle()

        # Should have slept due to critical RAM
        mock_sleep.assert_called()
        self.assertTrue(result)


class TestThermalMonitor(unittest.TestCase):
    """Test thermal monitoring functionality."""

    def test_thermal_monitor_initialization(self):
        """Verify thermal monitor initializes without error."""
        monitor = ThermalMonitor()
        # Should not raise, available depends on platform
        self.assertIsInstance(monitor.available, bool)

    def test_thermal_status_unavailable(self):
        """Verify graceful handling when thermal unavailable."""
        monitor = ThermalMonitor()
        monitor._available = False

        status = monitor.get_status()
        self.assertEqual(status, "unavailable")

    @patch.object(ThermalMonitor, "get_temperature")
    def test_thermal_status_normal(self, mock_temp):
        """Verify normal status for low temperatures."""
        mock_temp.return_value = 45.0

        monitor = ThermalMonitor()
        monitor._available = True

        status = monitor.get_status(warning_threshold=70.0, critical_threshold=80.0)
        self.assertEqual(status, "normal")

    @patch.object(ThermalMonitor, "get_temperature")
    def test_thermal_status_warning(self, mock_temp):
        """Verify warning status for elevated temperatures."""
        mock_temp.return_value = 75.0

        monitor = ThermalMonitor()
        monitor._available = True

        status = monitor.get_status(warning_threshold=70.0, critical_threshold=80.0)
        self.assertEqual(status, "warning")

    @patch.object(ThermalMonitor, "get_temperature")
    def test_thermal_status_critical(self, mock_temp):
        """Verify critical status for dangerous temperatures."""
        mock_temp.return_value = 85.0

        monitor = ThermalMonitor()
        monitor._available = True

        status = monitor.get_status(warning_threshold=70.0, critical_threshold=80.0)
        self.assertEqual(status, "critical")


class TestThermalThrottling(unittest.TestCase):
    """Test thermal throttling in ResourceGovernor."""

    @patch("nexus.service.resource.time.sleep")
    @patch("nexus.service.resource.psutil.Process")
    def test_thermal_warning_throttles(self, mock_process, mock_sleep):
        """Verify governor throttles on thermal warning."""
        gov = ResourceGovernor(ResourceConfig(thermal_warning=70.0, thermal_critical=80.0))

        # Mock normal CPU/RAM
        mock_process.return_value.cpu_percent.return_value = 5.0
        mock_process.return_value.memory_percent.return_value = 10.0
        gov.process = mock_process.return_value

        # Mock thermal monitor to return warning temperature
        gov._thermal_monitor._available = True
        with patch.object(gov._thermal_monitor, "get_temperature", return_value=75.0):
            result = gov.check_and_throttle()

        # Should have throttled due to thermal warning
        self.assertTrue(result)
        mock_sleep.assert_called()

    @patch("nexus.service.resource.time.sleep")
    @patch("nexus.service.resource.psutil.Process")
    def test_thermal_critical_raises(self, mock_process, mock_sleep):
        """Verify governor raises ThermalThrottlingError on critical temp."""
        gov = ResourceGovernor(ResourceConfig(thermal_warning=70.0, thermal_critical=80.0))

        # Mock normal CPU/RAM
        mock_process.return_value.cpu_percent.return_value = 5.0
        mock_process.return_value.memory_percent.return_value = 10.0
        gov.process = mock_process.return_value

        # Mock thermal monitor to return critical temperature
        gov._thermal_monitor._available = True
        with patch.object(gov._thermal_monitor, "get_temperature", return_value=85.0):
            with self.assertRaises(ThermalThrottlingError):
                gov.check_and_throttle()

    @patch("nexus.service.resource.time.sleep")
    @patch("nexus.service.resource.psutil.Process")
    def test_thermal_unavailable_graceful(self, mock_process, mock_sleep):
        """Verify governor works when thermal reading unavailable (macOS)."""
        gov = ResourceGovernor(ResourceConfig())

        # Mock normal CPU/RAM
        mock_process.return_value.cpu_percent.return_value = 5.0
        mock_process.return_value.memory_percent.return_value = 10.0
        gov.process = mock_process.return_value

        # Mock thermal monitor as unavailable
        gov._thermal_monitor._available = False
        with patch.object(gov._thermal_monitor, "get_temperature", return_value=None):
            # Should not raise, should not throttle for thermal
            result = gov.check_and_throttle()

        # Should not have throttled (CPU/RAM are fine)
        self.assertFalse(result)


class TestGetStats(unittest.TestCase):
    """Test ResourceGovernor.get_stats() with thermal info."""

    @patch("nexus.service.resource.psutil.Process")
    def test_get_stats_includes_thermal(self, mock_process):
        """Verify get_stats includes thermal information."""
        gov = ResourceGovernor(ResourceConfig())

        # Mock CPU/RAM
        mock_process.return_value.cpu_percent.return_value = 15.0
        mock_process.return_value.memory_percent.return_value = 20.0
        gov.process = mock_process.return_value

        stats = gov.get_stats()

        # Verify thermal keys exist
        self.assertIn("thermal_celsius", stats)
        self.assertIn("thermal_status", stats)
        self.assertIn("thermal_warning_limit", stats)
        self.assertIn("thermal_critical_limit", stats)
        self.assertIn("thermal_available", stats)

        # Verify limits match config
        self.assertEqual(stats["thermal_warning_limit"], gov.config.thermal_warning)
        self.assertEqual(stats["thermal_critical_limit"], gov.config.thermal_critical)


if __name__ == "__main__":
    unittest.main()
