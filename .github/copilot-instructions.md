# NEXUS Project - Copilot Instructions

## Project Overview
NEXUS (Neural EXploratory Unified Synthesis) is a **living AI system** that:
- **Never hallucinates** - refuses politely when uncertain
- **Learns continuously** - every interaction is a learning opportunity  
- **Evolves organically** - no stages or labels, just smooth continuous growth
- Achieves O(n) efficiency vs Transformer's O(n²)
- **Layer-free architecture** - depth emerges from input, not hyperparameters

## Core Philosophy
```
Growth is not a ladder with rungs to climb.
It is water finding its level.
The system doesn't "become" something new -
it continuously IS, shaped by all it has experienced.
```

This philosophy now extends to the architecture itself:
- No fixed layers to traverse
- Computation flows continuously toward equilibrium
- "Depth" is emergent, not predetermined

## TWO Architecture Modes

### 1. FlowingNEXUS (Layer-Free, RECOMMENDED)
```python
from nexus.core import create_flowing_nexus, create_living_nexus

# Direct model
model = create_flowing_nexus(size="base")
result = model(x, modality="continuous")
print(f"Emergent depth: {result['flow_steps']}")

# Living system with layer-free architecture
nexus = create_living_nexus(size="small", architecture="flowing")
result = nexus.interact(batch)
print(f"Flow depth: {result.flow_depth}")
```

### 2. NEXUSCore (Traditional Layered)
```python
from nexus.core import create_living_nexus

# Traditional layered architecture
nexus = create_living_nexus(size="small", architecture="layered")
```

## Core Architecture Components

### Layer-Free (FlowingNEXUS)
- **UnifiedDynamics**: Single dynamics function iterated to equilibrium
- **EquilibriumCore**: Find fixed points with convergence guarantees
- **ContinuousSSM**: State space with emergent depth
- **ImplicitDifferentiation**: O(1) memory backprop through equilibrium

### Traditional Modules (shared)
- **SelectiveStateSpace**: O(n) linear-time sequence modeling (inspired by Mamba/S4)
- **HierarchicalWorldModel**: Abstract predictive world representation (inspired by JEPA)
- **NeuroSymbolicReasoner**: Hybrid neural-symbolic reasoning engine
- **AdaptiveEnergyModule**: Energy-based adaptive computation
- **CausalInferenceEngine**: Causal reasoning and planning
- **LifecycleManager**: Continuous evolution tracking (no stages)
- **UncertaintyGate**: Anti-hallucination - refuses when not confident

## Living System Usage (PREFERRED)
```python
from nexus.core import create_living_nexus

# Create layer-free living NEXUS (default)
nexus = create_living_nexus(size="small")  # architecture="flowing" is default

# Interact - learns and responds simultaneously
result = nexus.interact(batch)

if result.responded:
    print("Answer:", result.logits)
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Flow depth: {result.flow_depth}")  # Emergent depth!
else:
    print("NEXUS: I don't know enough about this yet.")

# Check evolution status
print(nexus.get_status())
# {'total_interactions': 1523, 'average_flow_depth': 12.3, ...}
```

## Configuration Parameter Names (IMPORTANT)
When working with NEXUS, use these correct parameter names:

### FlowingConfig Parameters (Layer-Free)
| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | State dimension (default: 512) |
| `d_latent` | int | Latent space dimension (default: 256) |
| `max_flow_steps` | int | Maximum evolution steps (default: 50) |
| `convergence_threshold` | float | Fixed point tolerance (default: 1e-4) |
| `damping` | float | Update damping factor (default: 0.5) |
| `ssm_d_state` | int | State space dimension (default: 64) |
| `memory_size` | int | Co-evolving memory size (default: 128) |
| `implicit_diff` | bool | Use implicit differentiation (default: True) |
| `vocab_size` | int | Vocabulary size (default: 50000) |
| `gradient_checkpointing` | bool | Enable gradient checkpointing (default: False) |
| `checkpoint_every_n_steps` | int | Checkpoint frequency (default: 5) |
| `max_trajectory_length` | int | Max trajectory states to keep (default: 10) |

### NEXUSConfig Parameters (Layered)
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

### FlowingNEXUS Usage Example
```python
from nexus.core import create_flowing_nexus, FlowingConfig

# Using factory function (recommended)
model = create_flowing_nexus(size="base")

# Or with custom config
config = FlowingConfig(
    d_model=512,
    max_flow_steps=30,
    convergence_threshold=1e-3,
)
model = FlowingNEXUS(config)

# Forward pass - depth emerges!
x = torch.randn(batch_size, seq_len, config.d_model)
result = model(x, modality="continuous")

print(f"Logits: {result['logits'].shape}")
print(f"Flow steps: {result['flow_steps']}")  # Varies per input!
print(f"Converged: {result['converged']}")
```

### NEXUSCore Usage Example
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
- `model.get_flow_complexity(x)` - (FlowingNEXUS) Get emergent depth metrics

## Development Guidelines
- Use Python 3.10+ (tested with 3.10-3.12)
- Follow PEP 8 style guidelines
- Use type hints throughout
- Document all public APIs with docstrings
- Write unit tests for new functionality
- Run tests with: `PYTHONPATH=. pytest tests/ -v`
- Run layer-free tests: `PYTHONPATH=. pytest tests/test_layerfree.py -v`
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
│   ├── flowing.py       # 🆕 FlowingNEXUS - layer-free architecture
│   ├── equilibrium.py   # 🆕 Equilibrium dynamics, implicit diff
│   ├── continuous_ssm.py # 🆕 Continuous SSM with emergent depth
│   ├── types.py         # 🆕 Type definitions and protocols
│   ├── nexus_core.py    # Main NEXUSCore class & NEXUSConfig
│   ├── living.py        # LivingNEXUS - unified learn+respond interface
│   ├── lifecycle.py     # LifecycleManager, UncertaintyGate (continuous evolution)
│   ├── state_space.py   # SelectiveStateSpace with true parallel scan
│   ├── world_model.py   # HierarchicalWorldModel & WorldModelConfig
│   ├── reasoning.py     # NeuroSymbolicReasoner & ReasoningConfig
│   ├── energy.py        # AdaptiveEnergyModule & EnergyConfig
│   └── causal.py        # CausalInferenceEngine & CausalConfig
├── training/
│   ├── trainer.py       # NEXUSTrainer (with EMA callback)
│   ├── continual.py     # ContinualLearner + FlowingContinualLearner
│   ├── losses.py        # NEXUSLoss, JEPALoss, CausalLoss, FlowingLoss
│   └── data.py          # Data utilities
└── evaluation/
    ├── benchmarks.py    # NEXUSBenchmark, ScalingBenchmark
    └── metrics.py       # Evaluation metrics
```

## Key Principles for Development
1. **Anti-hallucination first**: Always check confidence before responding
2. **Learn continuously**: Use ContinualLearner or LivingNEXUS for online learning
3. **Evolve organically**: No stages or labels, just continuous growth
4. **Track knowledge**: Log domain confidence and refusal rates
5. **Prefer layer-free**: Use FlowingNEXUS for new development (emergent depth)
6. **Type safety**: Use types from `nexus.core.types` for better IDE support
