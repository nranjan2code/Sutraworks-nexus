# NEXUS Project - Copilot Instructions

## Project Overview
NEXUS (Neural EXploratory Unified Synthesis) is a next-generation AI algorithm designed to surpass Transformer/LLM limitations.

## Core Architecture Components
- **SelectiveStateSpace**: O(n) linear-time sequence modeling (inspired by Mamba/S4)
- **HierarchicalWorldModel**: Abstract predictive world representation (inspired by JEPA)
- **NeuroSymbolicReasoner**: Hybrid neural-symbolic reasoning engine
- **AdaptiveEnergyModule**: Energy-based adaptive computation
- **CausalInferenceEngine**: Causal reasoning and planning

## Configuration Parameter Names (IMPORTANT)
When working with NEXUS, use these correct parameter names:

### NEXUSConfig Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Vocabulary size (default: 32000) |
| `d_model` | int | Hidden dimension (default: 256) |
| `d_latent` | int | Latent dimension for world model (default: 128) |
| `ssm_n_layers` | int | Number of state space layers (default: 6) |
| `n_heads` | int | Number of attention heads (default: 8) |
| `ssm_d_state` | int | State space state dimension (default: 64) |
| `ssm_d_conv` | int | Convolution kernel size (default: 4) |
| `ssm_expand` | int | MLP expansion factor (default: 2) |
| `n_predicates` | int | Reasoning predicates (default: 64) |
| `n_constants` | int | Reasoning constants (default: 128) |
| `max_reasoning_steps` | int | Max reasoning iterations (default: 5) |
| `n_variables` | int | Causal graph variables (default: 32) |
| `max_energy_iters` | int | Max energy iterations (default: 10) |
| `max_seq_len` | int | Maximum sequence length (default: 8192) |
| `dropout` | float | Dropout rate (default: 0.1) |

### Usage Example
```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

# Create configuration
config = NEXUSConfig(
    vocab_size=32000,
    d_model=512,
    d_latent=256,
    ssm_n_layers=6,
    n_heads=8,
    ssm_d_state=64
)

# Create model
model = NEXUSCore(config)

# Forward pass uses continuous input (float tensor), NOT token IDs
x = torch.randn(batch_size, seq_len, config.d_model)
outputs = model(x)  # Returns dict with 'logits' key
logits = outputs['logits']
```

### Common Methods
- `model(x)` - Forward pass, returns `{'logits': tensor}`
- `model.imagine(x, n_steps=5)` - World model imagination (note: `n_steps`, not `steps`)
- `model.reason(x)` - Reasoning with proof traces
- `model.intervene(x, intervention_idx, intervention_value)` - Causal intervention

## Development Guidelines
- Use Python 3.10+ (tested with 3.10-3.12)
- Follow PEP 8 style guidelines
- Use type hints throughout
- Document all public APIs with docstrings
- Write unit tests for new functionality
- Run tests with: `PYTHONPATH=. pytest tests/ -v`
- Run demo with: `PYTHONPATH=. python examples/basic_usage.py`

## Key Dependencies
- PyTorch 2.0+ for neural components
- NumPy/SciPy for numerical operations
- NetworkX for symbolic graph operations
- einops for tensor operations

## Project Structure
```
nexus/
├── core/
│   ├── nexus_core.py    # Main NEXUSCore class & NEXUSConfig
│   ├── state_space.py   # SelectiveStateSpace & StateSpaceConfig
│   ├── world_model.py   # HierarchicalWorldModel & WorldModelConfig
│   ├── reasoning.py     # NeuroSymbolicReasoner & ReasoningConfig
│   ├── energy.py        # AdaptiveEnergyModule & EnergyConfig
│   └── causal.py        # CausalInferenceEngine & CausalConfig
├── training/
│   ├── trainer.py       # NEXUSTrainer
│   ├── losses.py        # NEXUSLoss
│   └── data.py          # Data utilities
└── evaluation/
    ├── benchmarks.py    # NEXUSBenchmark
    └── metrics.py       # Evaluation metrics
```
