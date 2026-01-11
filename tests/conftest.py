"""
Pytest Configuration for NEXUS Tests
=====================================

Shared fixtures and configuration for all test modules.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import torch


# ==============================================================================
#  Configuration Constants
# ==============================================================================

# Small sizes for fast tests
TEST_BATCH_SIZES = [1, 2, 4]
TEST_SEQ_LENGTHS = [32, 64, 128]
TEST_D_MODEL = 128
TEST_D_LATENT = 64
TEST_VOCAB_SIZE = 1000
TEST_D_STATE = 16
TEST_N_HEADS = 4
TEST_N_LAYERS = 2


# ==============================================================================
#  Device Fixtures
# ==============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate test device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for specific tests."""
    return torch.device("cpu")


# ==============================================================================
#  Sample Tensor Fixtures
# ==============================================================================


@pytest.fixture
def sample_input(device: torch.device) -> torch.Tensor:
    """Create sample token input tensor."""
    return torch.randint(0, TEST_VOCAB_SIZE, (2, 64), device=device)


@pytest.fixture
def sample_hidden(device: torch.device) -> torch.Tensor:
    """Create sample hidden state tensor (continuous)."""
    return torch.randn(2, 64, TEST_D_MODEL, device=device)


@pytest.fixture
def sample_batch(device: torch.device) -> Dict[str, torch.Tensor]:
    """Create sample batch with input_ids and attention_mask."""
    input_ids = torch.randint(0, TEST_VOCAB_SIZE, (2, 64), device=device)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# ==============================================================================
#  Parameterized Fixtures
# ==============================================================================


@pytest.fixture(params=TEST_BATCH_SIZES)
def batch_size(request: pytest.FixtureRequest) -> int:
    """Parameterized batch size for comprehensive testing."""
    return request.param


@pytest.fixture(params=TEST_SEQ_LENGTHS)
def seq_length(request: pytest.FixtureRequest) -> int:
    """Parameterized sequence length for comprehensive testing."""
    return request.param


@pytest.fixture
def variable_input(
    device: torch.device,
    batch_size: int,
    seq_length: int,
) -> torch.Tensor:
    """Create input tensor with parameterized batch size and sequence length."""
    return torch.randint(0, TEST_VOCAB_SIZE, (batch_size, seq_length), device=device)


@pytest.fixture
def variable_hidden(
    device: torch.device,
    batch_size: int,
    seq_length: int,
) -> torch.Tensor:
    """Create hidden tensor with parameterized batch size and sequence length."""
    return torch.randn(batch_size, seq_length, TEST_D_MODEL, device=device)


# ==============================================================================
#  Model Configuration Fixtures
# ==============================================================================


@pytest.fixture
def small_model_config() -> Dict[str, Any]:
    """Small model configuration for fast tests."""
    return {
        "vocab_size": TEST_VOCAB_SIZE,
        "d_model": TEST_D_MODEL,
        "d_latent": TEST_D_LATENT,
        "ssm_n_layers": TEST_N_LAYERS,
        "n_heads": TEST_N_HEADS,
        "ssm_d_state": TEST_D_STATE,
    }


@pytest.fixture
def base_model_config() -> Dict[str, Any]:
    """Base model configuration."""
    return {
        "vocab_size": TEST_VOCAB_SIZE,
        "d_model": 256,
        "d_latent": 128,
        "ssm_n_layers": 4,
        "n_heads": 8,
        "ssm_d_state": 32,
    }


# ==============================================================================
#  Temporary Directory Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_checkpoint_dir(temp_dir: Path) -> Path:
    """Create a temporary checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


# ==============================================================================
#  Model Fixtures
# ==============================================================================


@pytest.fixture
def nexus_config():
    """Create a test NEXUSConfig."""
    from nexus.core.nexus_core import NEXUSConfig

    return NEXUSConfig(
        vocab_size=TEST_VOCAB_SIZE,
        d_model=TEST_D_MODEL,
        d_latent=TEST_D_LATENT,
        ssm_n_layers=TEST_N_LAYERS,
        n_heads=TEST_N_HEADS,
        ssm_d_state=TEST_D_STATE,
    )


@pytest.fixture
def nexus_model(nexus_config, device: torch.device):
    """Create a test NEXUSCore model."""
    from nexus.core.nexus_core import NEXUSCore

    model = NEXUSCore(nexus_config)
    return model.to(device)


@pytest.fixture
def flowing_config():
    """Create a test FlowingConfig."""
    from nexus.core.flowing import FlowingConfig

    return FlowingConfig(
        vocab_size=TEST_VOCAB_SIZE,
        d_model=TEST_D_MODEL,
        d_latent=TEST_D_LATENT,
        max_flow_steps=10,  # Fewer steps for faster tests
    )


@pytest.fixture
def flowing_model(flowing_config, device: torch.device):
    """Create a test FlowingNEXUS model."""
    from nexus.core.flowing import FlowingNEXUS

    model = FlowingNEXUS(flowing_config)
    return model.to(device)


@pytest.fixture
def living_nexus(device: torch.device):
    """Create a LivingNEXUS for integration tests."""
    from nexus.core.living import create_living_nexus

    nexus = create_living_nexus(size="small", architecture="flowing")
    return nexus


# ==============================================================================
#  Service Fixtures
# ==============================================================================


@pytest.fixture
def metrics_collector():
    """Create a MetricsCollector for testing."""
    from nexus.service.metrics import MetricsCollector

    return MetricsCollector()


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker for testing."""
    from nexus.service.resilience import CircuitBreaker

    return CircuitBreaker("test")


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    from nexus.core.tokenizer import NEXUSTokenizer

    return NEXUSTokenizer(model_name="gpt2")


# ==============================================================================
#  Cleanup Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Automatically clean up GPU memory after each test."""
    yield
    # Cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
#  Markers
# ==============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


# ==============================================================================
#  Skip Conditions
# ==============================================================================

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

requires_mps = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)

requires_gpu = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    ),
    reason="No GPU available",
)
