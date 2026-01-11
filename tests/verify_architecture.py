import torch
import time
import unittest
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig as NexusConfig


class TestArchitecture(unittest.TestCase):
    def test_generation_complexity(self):
        """Verify generation is faster per-token or linear scaling."""
        config = NexusConfig(d_model=64, ssm_n_layers=2)
        model = NEXUSCore(config)
        model.eval()

        # Prompt
        prompt = torch.randint(0, 100, (1, 10))

        # Warmup
        model.generate(prompt, max_new_tokens=5)

        # Measure time for 10 tokens
        start = time.time()
        model.generate(prompt, max_new_tokens=10)
        time_10 = time.time() - start

        # Measure time for 50 tokens
        start = time.time()
        model.generate(prompt, max_new_tokens=50)
        time_50 = time.time() - start

        print(f"Time 10: {time_10:.4f}s")
        print(f"Time 50: {time_50:.4f}s")

        # In O(N^2), 50 tokens would be roughly 25x slower than 10 (excluding prompt processing).
        # In O(N), it should be roughly 5x slower.
        # Let's check ratio.
        ratio = time_50 / time_10
        print(f"Ratio: {ratio:.2f}")

        # Allowing some constant overhead, but it shouldn't be quadratic
        self.assertLess(ratio, 10.0, "Generation scaling looks quadratic, expected linear(ish)")


if __name__ == "__main__":
    unittest.main()
