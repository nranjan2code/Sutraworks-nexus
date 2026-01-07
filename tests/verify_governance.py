import unittest
import torch
import time
from unittest.mock import MagicMock, patch
from nexus.core.flowing import FlowingNEXUS, FlowingConfig
from nexus.service.resource import ResourceGovernor, ResourceConfig, ResourceExhaustedError


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

        gov.check_and_throttle()

        # Should have slept
        mock_sleep.assert_called()

    @patch("nexus.service.resource.psutil.Process")
    def test_emergency_abort(self, mock_process):
        """Verify governor raises error on critical RAM."""
        gov = ResourceGovernor(ResourceConfig(critical_ram_limit=90.0))

        # Mock critical RAM
        mock_process.return_value.cpu_percent.return_value = 10.0
        mock_process.return_value.memory_percent.side_effect = [
            95.0,
            95.0,
        ]  # Initial check, then post-GC check

        with self.assertRaises(ResourceExhaustedError):
            gov.check_and_throttle()


if __name__ == "__main__":
    unittest.main()
