import os
import torch
import unittest
from nexus.training.teacher import OllamaTeacher
from nexus.core.flowing import FlowingNEXUS, FlowingConfig, DynamicsDivergenceError


class TestResilience(unittest.TestCase):
    def test_teacher_config(self):
        """Test that OllamaTeacher reads config correctly."""
        # Test explicit
        t = OllamaTeacher(base_url="http://custom-url:1234", model="custom-model")
        self.assertEqual(t.base_url, "http://custom-url:1234")
        self.assertEqual(t.model, "custom-model")

        # Test env var
        os.environ["OLLAMA_HOST"] = "http://env-url:5678"
        os.environ["OLLAMA_MODEL"] = "env-model"
        t2 = OllamaTeacher()
        self.assertEqual(t2.base_url, "http://env-url:5678")
        self.assertEqual(t2.model, "env-model")

        # Cleanup
        del os.environ["OLLAMA_HOST"]
        del os.environ["OLLAMA_MODEL"]

    def test_divergence_error(self):
        """Test that FlowingNEXUS raises DynamicsDivergenceError on NaNs."""
        config = FlowingConfig(d_model=16, d_latent=8, max_flow_steps=5)
        model = FlowingNEXUS(config)

        # Mock dynamics to produce NaNs
        # Mock dynamics to produce NaNs
        class MockDynamics(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return torch.tensor([float("nan")]).expand(1, 10, 16)

        model.dynamics = MockDynamics()

        input_tensor = torch.randint(0, 50, (1, 10))  # indices (vocab size is big enough)

        print("\nDepending on how FlowingNEXUS handles steps, this might fail efficiently...")
        with self.assertRaises(DynamicsDivergenceError):
            model(input_tensor)


if __name__ == "__main__":
    unittest.main()
