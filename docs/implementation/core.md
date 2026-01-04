# Implementation Guide

## Complete Implementation Reference for NEXUS

This section provides detailed implementation documentation for developers working with NEXUS.

---

## Project Structure

```
nexus/
├── __init__.py              # Package exports
├── core/                    # Core algorithm modules
│   ├── __init__.py
│   ├── nexus_core.py        # Main NEXUSCore class
│   ├── state_space.py       # Selective State Space model
│   ├── world_model.py       # Hierarchical World Model
│   ├── reasoning.py         # Neuro-Symbolic Reasoner
│   ├── energy.py            # Adaptive Energy Module
│   └── causal.py            # Causal Inference Engine
├── training/                # Training utilities
│   ├── __init__.py
│   ├── trainer.py           # Main training loop
│   ├── losses.py            # Loss functions
│   └── data.py              # Data loading
└── evaluation/              # Evaluation utilities
    ├── __init__.py
    ├── benchmarks.py        # Benchmark suite
    └── metrics.py           # Evaluation metrics
```

---

## Core Modules

### NEXUSCore (nexus_core.py)

The main integration class that combines all components:

```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

# Initialize with config
config = NEXUSConfig(
    vocab_size=32000,
    d_model=256,           # Hidden dimension
    d_latent=128,          # Latent dimension
    ssm_n_layers=6,        # Number of layers
    ssm_d_state=64,        # State dimension
    ssm_expand=2           # Expansion factor
)
model = NEXUSCore(config)

# Forward pass with continuous input
x = torch.randn(1, 100, config.d_model)  # [batch, seq_len, d_model]
output = model(x)  # Returns dict with 'logits'

# With detailed outputs
output, info = model(x, return_all_outputs=True)
```

**Key Parameters (NEXUSConfig):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 32000 | Vocabulary size |
| `d_model` | int | 256 | Model hidden dimension |
| `d_latent` | int | 128 | Latent dimension |
| `ssm_n_layers` | int | 6 | Number of state space layers |
| `ssm_d_state` | int | 64 | State space state dimension |
| `ssm_expand` | int | 2 | MLP expansion factor |

### SelectiveStateSpace (state_space.py)

O(n) linear-time sequence model:

```python
from nexus.core.state_space import SelectiveStateSpace, StateSpaceConfig

config = StateSpaceConfig(
    d_model=256,
    d_state=64,
    n_layers=6,
    d_conv=4,
    expand=2
)
ssm = SelectiveStateSpace(config)

# Process sequence
hidden_states = ssm(embeddings)  # [batch, seq_len, d_model]
```

**Implementation Details:**

1. **Input projection**: Expands to `d_model * expand`
2. **Selection mechanism**: Computes Δ, B, C from input
3. **SSM layer**: Applies discretized state space equations
4. **Output projection**: Projects back to `d_model`

### HierarchicalWorldModel (world_model.py)

Multi-level predictive world model:

```python
from nexus.core.world_model import HierarchicalWorldModel, WorldModelConfig

config = WorldModelConfig(
    d_model=256,
    d_latent=128,
    n_heads=8,
    n_layers=4
)
world_model = HierarchicalWorldModel(config)

# Predict future representations
predictions = world_model(hidden_states)
```

**Architecture:**
- **Encoder**: Encodes current observations
- **Target Encoder**: EMA-updated for stability
- **Predictor**: Predicts future at multiple levels

### NeuroSymbolicReasoner (reasoning.py)

Hybrid neural-symbolic reasoning:

```python
from nexus.core.reasoning import NeuroSymbolicReasoner, ReasoningConfig

config = ReasoningConfig(
    d_model=256,
    n_predicates=64,
    n_constants=128,
    max_steps=5
)
reasoner = NeuroSymbolicReasoner(config)

# Perform reasoning
reasoning_output, proof = reasoner(hidden_states)
```

**Components:**
- **Predicate Embeddings**: Learnable predicate vectors
- **Constant Embeddings**: Learnable constant vectors  
- **Soft Unification**: Differentiable unification
- **Forward Chaining**: Iterative rule application

### AdaptiveEnergyModule (energy.py)

Energy-based adaptive computation:

```python
from nexus.core.energy import AdaptiveEnergyModule, EnergyConfig

config = EnergyConfig(
    d_model=256,
    max_iterations=10,
    step_size=0.1,
    convergence_threshold=0.01
)
energy = AdaptiveEnergyModule(config)

# Refine with adaptive compute
refined, info = energy(hidden_states)
print(f"Used {info['iterations']} iterations")
```

**Key Features:**
- Energy function for uncertainty
- Iterative refinement loop
- Early exit for easy inputs
- Energy history tracking

### CausalInferenceEngine (causal.py)

Causal structure and reasoning:

```python
from nexus.core.causal import CausalInferenceEngine, CausalConfig

config = CausalConfig(
    d_model=256,
    n_variables=32,
    n_heads=4
)
causal = CausalInferenceEngine(config)

# Causal reasoning
causal_output = causal(hidden_states)

# Interventional query
result = causal.intervene(var_idx=0, intervention_value=1.0, target_idx=5)
```

**Capabilities:**
- Causal graph learning
- Interventional inference: do(X)
- Counterfactual reasoning
- Causal planning

---

## Training Pipeline

### NEXUSTrainer (trainer.py)

```python
from nexus.training import NEXUSTrainer, TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10,
    warmup_steps=1000,
    gradient_accumulation_steps=4
)

trainer = NEXUSTrainer(model, config)
trainer.train(train_dataset, val_dataset)
```

**Training Configuration:**

```python
@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batching
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Loss weights
    lm_loss_weight: float = 1.0
    world_loss_weight: float = 0.1
    reason_loss_weight: float = 0.05
    energy_loss_weight: float = 0.05
    causal_loss_weight: float = 0.05
```

### Multi-Objective Loss (losses.py)

```python
from nexus.training import NEXUSLoss

loss_fn = NEXUSLoss(
    lm_weight=1.0,
    world_weight=0.1,
    reason_weight=0.05,
    energy_weight=0.05,
    causal_weight=0.05
)

# Compute loss
total_loss, loss_dict = loss_fn(
    model_output,
    targets,
    all_outputs=module_outputs
)
```

### Data Loading (data.py)

```python
from nexus.training import NEXUSDataset, create_dataloader

# Create dataset
dataset = NEXUSDataset(
    data_path="path/to/data",
    tokenizer=tokenizer,
    max_length=2048
)

# Create dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

---

## Evaluation

### Benchmarks (benchmarks.py)

```python
from nexus.evaluation import NEXUSBenchmark

benchmark = NEXUSBenchmark(model)

# Run all benchmarks
results = benchmark.run_all()

# Individual benchmarks
lm_results = benchmark.language_modeling()
reason_results = benchmark.reasoning()
causal_results = benchmark.causal_inference()
efficiency_results = benchmark.efficiency()
```

**Benchmark Categories:**

1. **Language Modeling**: Perplexity on standard datasets
2. **Reasoning**: Logic, math, common sense
3. **Causal**: Interventional accuracy, counterfactual
4. **Efficiency**: Latency, memory, throughput

### Metrics (metrics.py)

```python
from nexus.evaluation import (
    compute_perplexity,
    compute_accuracy,
    compute_reasoning_metrics,
    compute_efficiency_metrics
)

# Language modeling
ppl = compute_perplexity(model, eval_data)

# Reasoning
reason_metrics = compute_reasoning_metrics(model, reasoning_data)

# Efficiency
eff = compute_efficiency_metrics(model, seq_len=2048)
```

---

## Usage Examples

### Basic Usage

```python
import torch
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

# Initialize model
config = NEXUSConfig(
    vocab_size=32000,
    d_model=256,
    ssm_n_layers=6
)
model = NEXUSCore(config)

# Prepare continuous input
x = torch.randn(1, 100, config.d_model)

# Forward pass
outputs = model(x)
logits = outputs['logits']

# Generate (conceptual - requires implementation)
def generate(model, prompt, max_new_tokens=50):
    for _ in range(max_new_tokens):
        outputs = model(prompt)
        logits = outputs['logits']
        next_token = logits[:, -1, :].argmax(dim=-1)
        # ... (extend sequence)
    return prompt
```

### Training Loop

```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
from nexus.training import NEXUSLoss, NEXUSTrainer, TrainingConfig

# Model
model_config = NEXUSConfig(vocab_size=32000, d_model=256)
model = NEXUSCore(model_config)

# Loss and trainer
loss_fn = NEXUSLoss()
train_config = TrainingConfig(learning_rate=1e-4, batch_size=32)
trainer = NEXUSTrainer(model, train_config, loss_fn)

# Train
trainer.train(train_loader, val_loader)
```

### Custom Module Configuration

```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

# High reasoning capacity
config = NEXUSConfig(
    vocab_size=32000,
    d_model=512,
    d_latent=256,
    ssm_n_layers=12,
    max_reasoning_steps=10,
    n_variables=64
)
model = NEXUSCore(config)

# Efficiency focused
efficient_config = NEXUSConfig(
    vocab_size=32000,
    d_model=128,
    ssm_n_layers=4,
    ssm_d_state=32,
    max_energy_iters=5
)
model = NEXUSCore(efficient_config)
```
```

---

## Device and Memory

### GPU Support

```python
# Move to GPU
model = model.cuda()

# Multi-GPU with DataParallel
model = nn.DataParallel(model)

# Distributed training
model = nn.parallel.DistributedDataParallel(model)
```

### Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = loss_fn(output, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Memory Optimization

```python
# Gradient checkpointing
model.state_space.gradient_checkpointing = True

# Compile with torch 2.0
model = torch.compile(model)
```

---

## Configuration Reference

### Full Configuration

```python
from nexus.core.nexus_core import NEXUSConfig

config = NEXUSConfig(
    # Model architecture
    vocab_size=32000,
    d_model=256,              # Hidden dimension
    d_latent=128,             # Latent dimension for world model
    
    # State Space
    ssm_n_layers=6,           # Number of SSM layers
    ssm_d_state=64,           # State space state dimension
    ssm_d_conv=4,             # Convolution kernel size
    ssm_expand=2,             # MLP expansion factor
    
    # World Model
    n_heads=8,                # Attention heads
    
    # Reasoning
    n_predicates=64,          # Number of reasoning predicates
    n_constants=128,          # Number of reasoning constants  
    max_reasoning_steps=5,    # Max reasoning iterations
    
    # Energy
    max_energy_iters=10,      # Max energy refinement iterations
    
    # Causal
    n_variables=32,           # Number of causal variables
    
    # Sequence
    max_seq_len=8192,         # Maximum sequence length
    dropout=0.1               # Dropout rate
)
```

---

## Extending NEXUS

### Custom Module

```python
class CustomModule(nn.Module):
    """Custom module following NEXUS interface."""
    
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Linear(d_model, d_model)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        return self.net(hidden_states)

# Add to NEXUSCore
class CustomNEXUS(NEXUSCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom = CustomModule(self.config.d_model)
    
    def forward(self, input_ids):
        # Standard forward
        output = super().forward(input_ids)
        # Add custom processing
        output = output + self.custom(output)
        return output
```

### Custom Loss

```python
class CustomLoss(nn.Module):
    def forward(self, output, target, **kwargs):
        base_loss = F.cross_entropy(output, target)
        # Custom regularization
        custom_loss = ...
        return base_loss + custom_loss
```

---

## Debugging Tips

### Check Module Outputs

```python
output, info = model(input_ids, return_all_outputs=True)

print("Module outputs:")
for name, out in info['module_outputs'].items():
    print(f"  {name}: {out.shape}, mean={out.mean():.4f}")

print(f"Energy iterations: {info['energy_info']['iterations']}")
```

### Gradient Flow

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### Profile Performance

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Next Steps

- [API Reference](../api/nexus-core.md) - Detailed API documentation
- [Tutorials](../tutorials/) - Step-by-step guides
- [Architecture](overview.md) - Design documentation

---

*Implementation is where theory meets reality.*
