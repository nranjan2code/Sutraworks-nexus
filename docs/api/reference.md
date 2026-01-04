# API Reference

## Complete API Documentation for NEXUS

This document provides comprehensive API documentation for all NEXUS modules.

---

## Table of Contents

- [NEXUSCore](#nexuscore)
- [SelectiveStateSpace](#selectivestatespace)
- [HierarchicalWorldModel](#hierarchicalworldmodel)
- [NeuroSymbolicReasoner](#neurosymbolicreasoner)
- [AdaptiveEnergyModule](#adaptiveenergymodule)
- [CausalInferenceEngine](#causalinferenceengine)
- [Training](#training)
- [Evaluation](#evaluation)

---

## NEXUSCore

Main integration class combining all NEXUS components.

### Class Definition

```python
class NEXUSCore(nn.Module):
    """
    NEXUS Core Model - Unified architecture combining:
    - Selective State Space (backbone)
    - Hierarchical World Model (prediction)
    - Neuro-Symbolic Reasoner (reasoning)
    - Adaptive Energy Module (computation)
    - Causal Inference Engine (causality)
    """
```

### Constructor

```python
def __init__(
    self,
    config: NEXUSConfig
)
```

NEXUSCore takes a NEXUSConfig dataclass with the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 32000 | Size of vocabulary |
| `d_model` | int | 256 | Hidden dimension |
| `d_latent` | int | 128 | Latent dimension for world model |
| `ssm_n_layers` | int | 6 | Number of SSM layers |
| `n_heads` | int | 8 | Attention heads |
| `ssm_d_state` | int | 64 | State space dimension |
| `ssm_d_conv` | int | 4 | Convolution kernel size |
| `ssm_expand` | int | 2 | MLP expansion factor |
| `max_reasoning_steps` | int | 5 | Max reasoning iterations |
| `max_energy_iters` | int | 10 | Max energy refinement steps |
| `n_variables` | int | 32 | Number of causal variables |
| `n_predicates` | int | 64 | Number of reasoning predicates |
| `n_constants` | int | 128 | Number of reasoning constants |
| `max_seq_len` | int | 8192 | Maximum sequence length |
| `dropout` | float | 0.1 | Dropout probability |

### Methods

#### forward

```python
def forward(
    self,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_all_outputs: bool = False
) -> Union[Dict[str, torch.Tensor], Tuple[Dict, Dict]]
```

**Parameters:**
- `x`: Input tensor (continuous). Shape: `[batch_size, seq_len, d_model]`
- `attention_mask`: Optional attention mask. Shape: `[batch_size, seq_len]`
- `return_all_outputs`: Whether to return intermediate module outputs

**Returns:**
- If `return_all_outputs=False`: Dict with 'logits' tensor. Shape: `[batch_size, seq_len, vocab_size]`
- If `return_all_outputs=True`: Tuple of (outputs_dict, info_dict)

**Example:**
```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

config = NEXUSConfig(vocab_size=32000, d_model=256)
model = NEXUSCore(config)
x = torch.randn(2, 100, 256)  # Continuous input

# Basic forward
outputs = model(x)
logits = outputs['logits']

# With all outputs
outputs, info = model(x, return_all_outputs=True)
print(info['energy_info']['iterations'])
```

#### generate

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True
) -> torch.Tensor
```

**Parameters:**
- `input_ids`: Prompt token IDs
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_k`: Top-k filtering
- `top_p`: Nucleus sampling probability
- `do_sample`: Whether to sample (vs greedy)

**Returns:** Generated token IDs including prompt

#### from_pretrained

```python
@classmethod
def from_pretrained(cls, checkpoint_path: str) -> "NEXUSCore"
```

Load model from checkpoint.

#### save_pretrained

```python
def save_pretrained(self, save_path: str) -> None
```

Save model to directory.

---

## SelectiveStateSpace

O(n) linear-time sequence backbone.

### Class Definition

```python
class SelectiveStateSpace(nn.Module):
    """
    Selective State Space Model backbone.
    
    Implements content-aware state transitions for
    efficient long-range dependency modeling.
    """
```

### Constructor

```python
def __init__(
    self,
    config: StateSpaceConfig
)
```

SelectiveStateSpace takes a StateSpaceConfig dataclass:

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 256 | Model dimension |
| `d_state` | int | 64 | State space dimension |
| `n_layers` | int | 6 | Number of SSM layers |
| `d_conv` | int | 4 | Convolution kernel size |
| `expand` | int | 2 | FFN expansion |
| `dt_min` | float | 0.001 | Minimum step size |
| `dt_max` | float | 0.1 | Maximum step size |

### Methods

#### forward

```python
def forward(
    self,
    x: torch.Tensor,
    return_states: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]
```

**Parameters:**
- `x`: Input embeddings. Shape: `[batch, seq_len, d_model]`
- `return_states`: Whether to return intermediate states

**Returns:** Hidden states, optionally with intermediate layer states

---

## HierarchicalWorldModel

Multi-level predictive world model.

### Class Definition

```python
class HierarchicalWorldModel(nn.Module):
    """
    JEPA-inspired hierarchical world model.
    
    Predicts future representations at multiple
    temporal scales using EMA target encoder.
    """
```

### Constructor

```python
def __init__(
    self,
    config: WorldModelConfig
)
```

HierarchicalWorldModel takes a WorldModelConfig dataclass:

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 256 | Hidden dimension |
| `d_latent` | int | 128 | Latent dimension |
| `n_heads` | int | 8 | Attention heads |
| `n_layers` | int | 4 | Transformer layers |
| `max_seq_len` | int | 8192 | Max sequence length |
| `ema_decay` | float | 0.996 | EMA momentum |

### Methods

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `hidden_states`: Input from backbone. Shape: `[batch, seq_len, d_model]`

**Returns:** World model enhanced representations

#### predict

```python
def predict(
    self,
    hidden_states: torch.Tensor,
    horizon: int = 5
) -> torch.Tensor
```

Predict future representations.

**Parameters:**
- `hidden_states`: Current states
- `horizon`: Steps to predict ahead

**Returns:** Predicted future representations

#### update_target

```python
def update_target(self) -> None
```

Update EMA target encoder (call after each training step).

---

## NeuroSymbolicReasoner

Hybrid neural-symbolic reasoning module.

### Class Definition

```python
class NeuroSymbolicReasoner(nn.Module):
    """
    Neuro-symbolic reasoning engine.
    
    Combines neural representations with
    differentiable symbolic operations.
    """
```

### Constructor

```python
def __init__(
    self,
    config: ReasoningConfig
)
```

NeuroSymbolicReasoner takes a ReasoningConfig dataclass:

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 256 | Hidden dimension |
| `n_predicates` | int | 64 | Number of predicates |
| `n_constants` | int | 128 | Number of constants |
| `max_steps` | int | 5 | Max reasoning steps |
| `temperature` | float | 0.1 | Softmax temperature |

### Methods

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    return_proof: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]
```

**Parameters:**
- `hidden_states`: Input representations
- `return_proof`: Return proof trace

**Returns:** Reasoning-enhanced representations, optionally with proof

#### reason

```python
def reason(
    self,
    query: torch.Tensor,
    context: torch.Tensor,
    max_steps: Optional[int] = None
) -> Dict[str, Any]
```

Explicit reasoning query.

**Parameters:**
- `query`: Query embedding
- `context`: Context representations
- `max_steps`: Override max steps

**Returns:** Dict with `answer`, `proof`, `confidence`

---

## AdaptiveEnergyModule

Energy-based adaptive computation.

### Class Definition

```python
class AdaptiveEnergyModule(nn.Module):
    """
    Adaptive computation via energy minimization.
    
    Allocates more compute to harder inputs,
    enables early exit for easy ones.
    """
```

### Constructor

```python
def __init__(
    self,
    config: EnergyConfig
)
```

AdaptiveEnergyModule takes an EnergyConfig dataclass:

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 256 | Hidden dimension |
| `max_iterations` | int | 10 | Max refinement iterations |
| `step_size` | float | 0.1 | Gradient step size |
| `convergence_threshold` | float | 0.01 | Convergence criterion |

### Methods

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    force_full: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]
```

**Parameters:**
- `hidden_states`: Input representations
- `force_full`: Force all iterations

**Returns:** Tuple of (refined_states, info_dict)

Info dict contains:
- `iterations`: Number of iterations used
- `energy_history`: List of energy values
- `final_energy`: Final energy value
- `skipped`: Whether early exit was used

#### compute_energy

```python
def compute_energy(
    self,
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor
```

Compute energy for state configuration.

**Returns:** Energy values. Shape: `[batch]`

---

## CausalInferenceEngine

Causal structure and reasoning.

### Class Definition

```python
class CausalInferenceEngine(nn.Module):
    """
    Causal inference engine for NEXUS.
    
    Supports causal discovery, interventional
    queries, and counterfactual reasoning.
    """
```

### Constructor

```python
def __init__(
    self,
    config: CausalConfig
)
```

CausalInferenceEngine takes a CausalConfig dataclass:

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 256 | Hidden dimension |
| `n_variables` | int | 32 | Number of causal variables |
| `n_heads` | int | 4 | Attention heads |

### Methods

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor
) -> torch.Tensor
```

Apply causal reasoning to hidden states.

#### intervene

```python
def intervene(
    self,
    variable: int,
    value: torch.Tensor,
    target: int,
    num_samples: int = 100
) -> Dict[str, torch.Tensor]
```

Compute P(target | do(variable = value)).

**Parameters:**
- `variable`: Variable to intervene on
- `value`: Intervention value
- `target`: Target variable to query
- `num_samples`: Monte Carlo samples

**Returns:** Dict with `mean`, `std`, `samples`

#### counterfactual

```python
def counterfactual(
    self,
    observed: Dict[int, torch.Tensor],
    intervention: Tuple[int, torch.Tensor],
    target: int
) -> torch.Tensor
```

Answer "What if variable had been different?"

#### get_graph

```python
def get_graph(
    self,
    threshold: float = 0.5
) -> torch.Tensor
```

Get current causal graph adjacency matrix.

#### discover_structure

```python
def discover_structure(
    self,
    data: torch.Tensor,
    num_epochs: int = 1000
) -> torch.Tensor
```

Learn causal structure from data.

---

## Training

### NEXUSTrainer

```python
class NEXUSTrainer:
    """Main trainer for NEXUS models."""
    
    def __init__(
        self,
        model: NEXUSCore,
        config: TrainingConfig,
        loss_fn: Optional[NEXUSLoss] = None
    )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ) -> None
    
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> float
    
    def save_checkpoint(self, name: str) -> None
    
    def load_checkpoint(self, path: str) -> None
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Model
    vocab_size: int = 32000
    d_model: int = 256          # Hidden dimension
    ssm_n_layers: int = 6       # Number of layers
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batching
    batch_size: int = 32
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Loss weights
    lm_loss_weight: float = 1.0
    world_loss_weight: float = 0.1
    reason_loss_weight: float = 0.05
    energy_loss_weight: float = 0.05
    causal_loss_weight: float = 0.05
    
    # I/O
    output_dir: str = './outputs'
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Hardware
    device: str = 'cuda'
    fp16: bool = True
```

### NEXUSLoss

```python
class NEXUSLoss(nn.Module):
    """Multi-objective loss function."""
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        world_weight: float = 0.1,
        reason_weight: float = 0.05,
        energy_weight: float = 0.05,
        causal_weight: float = 0.05
    )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        module_outputs: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

---

## Evaluation

### NEXUSBenchmark

```python
class NEXUSBenchmark:
    """Comprehensive evaluation suite."""
    
    def __init__(
        self,
        model: NEXUSCore,
        device: str = 'cuda',
        batch_size: int = 16
    )
    
    def run_all(self) -> Dict[str, Any]
    
    def language_modeling(self) -> Dict[str, float]
    
    def reasoning(self) -> Dict[str, float]
    
    def causal_inference(self) -> Dict[str, float]
    
    def efficiency(self) -> Dict[str, float]
    
    def adaptive_computation(self) -> Dict[str, Any]
    
    def print_summary(self, results: Dict) -> None
```

### Metrics

```python
def compute_perplexity(
    model: NEXUSCore,
    data: DataLoader,
    device: str = 'cuda'
) -> float

def compute_accuracy(
    model: NEXUSCore,
    data: DataLoader,
    device: str = 'cuda'
) -> float

def compute_reasoning_metrics(
    model: NEXUSCore,
    test_cases: List[Dict]
) -> Dict[str, float]

def compute_efficiency_metrics(
    model: NEXUSCore,
    seq_lengths: List[int] = [128, 512, 1024, 2048]
) -> Dict[str, float]
```

---

## Type Definitions

```python
from typing import Dict, List, Optional, Tuple, Union, Any

# Common types
TensorDict = Dict[str, torch.Tensor]
InfoDict = Dict[str, Any]
MetricsDict = Dict[str, float]
```

---

## Exceptions

```python
class NEXUSError(Exception):
    """Base exception for NEXUS."""
    pass

class ConfigurationError(NEXUSError):
    """Invalid configuration."""
    pass

class CheckpointError(NEXUSError):
    """Checkpoint loading/saving error."""
    pass
```

---

## Constants

```python
# Default configurations
DEFAULT_VOCAB_SIZE = 32000
DEFAULT_D_MODEL = 256        # Hidden dimension
DEFAULT_SSM_N_LAYERS = 6     # Number of layers
DEFAULT_SSM_D_STATE = 64     # State dimension

# Model variants (parameters for NEXUSConfig)
NEXUS_TINY = dict(d_model=64, ssm_n_layers=2)
NEXUS_SMALL = dict(d_model=128, ssm_n_layers=4)
NEXUS_BASE = dict(d_model=256, ssm_n_layers=6)
NEXUS_LARGE = dict(d_model=512, ssm_n_layers=12)
```

---

*Complete API reference for NEXUS. Build with confidence.*
