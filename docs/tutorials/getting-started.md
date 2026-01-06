# Tutorials

## Getting Started with NEXUS

Step-by-step tutorials for using NEXUS.

---

## Tutorial Index

1. [Quick Start](#quick-start) - Get up and running in 5 minutes
2. [Layer-Free Architecture](#layer-free-architecture) - 🆕 Emergent depth computation
3. [Basic Usage](#basic-usage) - Core functionality
4. [Training Your First Model](#training-your-first-model) - End-to-end training
5. [Reasoning Tasks](#reasoning-tasks) - Using reasoning capabilities
6. [Causal Inference](#causal-inference) - Causal queries and planning
7. [Custom Configurations](#custom-configurations) - Customize NEXUS
8. [Continual / Online Learning](#continual--online-learning) - Learn while serving

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sutraworks/nexus.git
cd nexus

# Install dependencies
pip install -r requirements.txt

# Install NEXUS
pip install -e .
```

### First Inference (Layer-Free - Recommended)

```python
import torch
from nexus.core import create_flowing_nexus

# Create layer-free model
model = create_flowing_nexus(size="small")

# Prepare continuous input
x = torch.randn(1, 50, model.config.d_model)

# Forward pass - depth emerges naturally!
result = model(x, modality="continuous")

print(f"Output shape: {result['logits'].shape}")
print(f"Flow steps (emergent depth): {result['flow_steps']}")
print(f"Converged: {result['converged']}")
```

### First Inference (Traditional Layered)

```python
import torch
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

# Create configuration
config = NEXUSConfig(
    vocab_size=32000,
    d_model=256,           # Hidden dimension
    ssm_n_layers=6,        # Number of layers
    ssm_d_state=64         # State dimension
)

# Create model
model = NEXUSCore(config)

# Prepare continuous input (float tensor, not token IDs)
x = torch.randn(1, 50, config.d_model)

# Forward pass
outputs = model(x)
logits = outputs['logits']
print(f"Output shape: {logits.shape}")
# Output: torch.Size([1, 50, 32000])
```

---

## Layer-Free Architecture

🆕 **NEW in NEXUS**: FlowingNEXUS introduces layer-free computation where depth emerges from input complexity.

### The Philosophy

> *Growth is not a ladder with rungs to climb.*  
> *It is water finding its level.*

Traditional neural nets have fixed depth: `input → layer₁ → layer₂ → ... → layerₙ → output`

FlowingNEXUS flows to equilibrium: `input → f(z*, input) → output` where `z* = f(z*, input)`

### Key Benefits

| Traditional Layers | Layer-Free (FlowingNEXUS) |
|--------------------|---------------------------|
| Fixed compute for all inputs | Adapts compute to input complexity |
| Depth is a hyperparameter | Depth emerges naturally |
| Same iterations always | Easy inputs converge fast |
| Parameters scale with depth | Constant parameters |

### Using FlowingNEXUS

```python
from nexus.core import create_flowing_nexus, FlowingConfig

# Simple creation
model = create_flowing_nexus(size="base")

# Or with custom config
config = FlowingConfig(
    d_model=512,
    max_flow_steps=50,         # Max iterations (not fixed depth!)
    convergence_threshold=1e-4, # When to stop
    damping=0.5,               # Update smoothing
)
model = FlowingNEXUS(config)

# Forward pass
x = torch.randn(2, 100, config.d_model)
result = model(x, modality="continuous")

# Results include flow information
print(f"Logits: {result['logits'].shape}")
print(f"Flow steps: {result['flow_steps']}")     # How many iterations
print(f"Converged: {result['converged']}")       # Did it reach equilibrium?
print(f"Final energy: {result['final_energy']}") # Residual norm
```

### Observing Emergent Depth

```python
# Different inputs need different "depths"
model = create_flowing_nexus(size="small")

# Simple input
simple_x = torch.zeros(1, 50, model.config.d_model)
simple_result = model(simple_x, modality="continuous")
print(f"Simple input depth: {simple_result['flow_steps']}")

# Complex input
complex_x = torch.randn(1, 50, model.config.d_model) * 5
complex_result = model(complex_x, modality="continuous")
print(f"Complex input depth: {complex_result['flow_steps']}")

# Depth varies based on input!
```

### Flow Trajectory Visualization

```python
# Get the full evolution trajectory
result = model(x, modality="continuous", return_trajectory=True)

trajectory = result['trajectory']
print(f"Trajectory has {len(trajectory)} states")

# Plot convergence
import matplotlib.pyplot as plt

energies = []
for i, state in enumerate(trajectory[:-1]):
    diff = (trajectory[i+1] - state).norm().item()
    energies.append(diff)

plt.plot(energies)
plt.xlabel("Iteration")
plt.ylabel("State Change (Energy)")
plt.title("Convergence to Equilibrium")
plt.show()
```

---

## Basic Usage

### Understanding the Output

```python
# Standard forward pass
outputs = model(x)
logits = outputs['logits']

# With detailed outputs
outputs, info = model(x, return_all_outputs=True)

# info contains:
# - module_outputs: dict with each module's output
# - energy_info: adaptive computation statistics
# - backbone_states: hidden states from SSM

print(f"Energy iterations: {info['energy_info']['iterations']}")
print(f"Final energy: {info['energy_info']['final_energy']:.4f}")
```

### Text Generation

```python
def generate_text(model, tokenizer, prompt, max_tokens=100):
    """Simple text generation."""
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        
        # Sample
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0])

# Usage
generated = generate_text(model, tokenizer, "The future of AI is")
print(generated)
```

### Batch Processing

```python
# Process multiple sequences (continuous input)
batch_size = 4
seq_length = 100
batch = torch.randn(batch_size, seq_length, config.d_model)

# Forward pass
outputs = model(batch)
logits = outputs['logits']
print(f"Batch output shape: {logits.shape}")
# Output: torch.Size([4, 100, 32000])

# With attention mask
attention_mask = torch.ones(batch_size, seq_length).long()
attention_mask[:, -10:] = 0  # Mask last 10 positions
outputs = model(batch, attention_mask=attention_mask)
```

---

## Training Your First Model

### Step 1: Prepare Data

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    """Simple text dataset."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            self.examples.append(torch.tensor(tokens))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }

# Create dataset
texts = [
    "This is example text for training.",
    "NEXUS is a next-generation AI algorithm.",
    # ... more examples
]

dataset = SimpleDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Step 2: Configure Training

```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
from nexus.training import TrainingConfig, NEXUSTrainer, NEXUSLoss

# Model configuration
model_config = NEXUSConfig(
    vocab_size=32000,
    d_model=256,           # Hidden dimension
    ssm_n_layers=4,        # Smaller for demo
    ssm_d_state=64         # State dimension
)

# Training configuration
train_config = TrainingConfig(
    vocab_size=32000,
    d_model=256,
    ssm_n_layers=4,
    
    learning_rate=1e-4,
    batch_size=8,
    max_steps=1000,
    warmup_steps=100,
    
    output_dir='./my_nexus_model',
    save_steps=200,
    eval_steps=100
)

# Model
model = NEXUSCore(model_config)

# Loss and trainer
loss_fn = NEXUSLoss()
trainer = NEXUSTrainer(model, train_config, loss_fn)
```

### Step 3: Train

```python
# Train!
trainer.train(train_loader, val_loader)

# Model is saved to config.output_dir
print(f"Model saved to {config.output_dir}")
```

---

## Continual / Online Learning

NEXUS is designed as a **living system** that learns and responds in parallel, never hallucinates, and evolves organically through experience.

### Philosophy

> *Growth is not a ladder with rungs to climb.*  
> *It is water finding its level.*  
> *The system doesn't "become" something new -*  
> *it continuously IS, shaped by all it has experienced.*

### Using Living NEXUS with Layer-Free Architecture (Recommended)

```python
from nexus.core import create_living_nexus

# Create a fresh NEXUS with layer-free architecture (default)
nexus = create_living_nexus(size="small")  # architecture="flowing" is default

# Interact - learns and responds simultaneously
result = nexus.interact(batch)

if result.responded:
    print("Answer:", result.logits)
    print(f"Confidence: {result.confidence:.2f}")
    
    # 🆕 Layer-free architecture provides flow information
    print(f"Flow depth: {result.flow_depth}")  # Emergent depth
    print(f"Converged: {result.converged}")    # Did it reach equilibrium?
else:
    # NEXUS wisely refuses when uncertain - this is not failure, it's wisdom
    print("NEXUS: I don't know enough about this yet.")

# Check evolution status (no stages, just continuous metrics)
status = nexus.get_status()
print(f"Total interactions: {status['total_interactions']}")
print(f"Experience factor: {status['experience_factor']:.4f}")  # 0→1 smooth curve
print(f"Wisdom ratio: {status['wisdom_ratio']:.2f}")  # How often it wisely refuses
print(f"Confidence threshold: {status['confidence_threshold']:.2f}")  # Evolves with experience
print(f"Average flow depth: {status['average_flow_depth']:.1f}")  # 🆕 Mean emergent depth
```

### Using Living NEXUS with Traditional Layered Architecture

```python
from nexus.core import create_living_nexus

# Create NEXUS with traditional layered architecture
nexus = create_living_nexus(size="small", architecture="layered")

# Interact - same interface, but fixed depth
result = nexus.interact(batch)

if result.responded:
    print("Answer:", result.logits)
    print(f"Confidence: {result.confidence:.2f}")
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **experience_factor** | 0→1 continuous measure of accumulated experience |
| **wisdom_ratio** | How often the system wisely says "I don't know" |
| **confidence_threshold** | Starts high (0.95), decreases smoothly with experience |
| **learning_rate_multiplier** | Starts high (2.5x), decreases as system matures |

### Using ContinualLearner (Lower Level)

For more control, use ContinualLearner directly:

```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
from nexus.training import TrainingConfig, ContinualConfig, ContinualLearner

model = NEXUSCore(NEXUSConfig())
train_cfg = TrainingConfig()
cont_cfg = ContinualConfig(buffer_size=2048, replay_ratio=0.5, microbatch_size=4)
learner = ContinualLearner(model, train_cfg, cont_cfg)

# Answer
outputs = learner.respond(batch)

# Learn a few safe steps with replay while continuing to answer
metrics = learner.observe_and_learn([batch])
print(metrics)
```

### Step 4: Evaluate

```python
from nexus.evaluation import NEXUSBenchmark

# Load trained model
model = NEXUSCore.from_pretrained('./my_nexus_model/best')

# Run benchmarks
benchmark = NEXUSBenchmark(model)
results = benchmark.run_all()
benchmark.print_summary(results)
```

---

## Reasoning Tasks

### Using the Reasoner

```python
# Access reasoner directly
reasoner = model.reasoner

# Perform reasoning query
hidden_states = model.state_space(model.embedding(input_ids))
reasoning_output, proof = reasoner(hidden_states, return_proof=True)

print(f"Proof steps: {len(proof['steps'])}")
```

### Logic Reasoning Example

```python
def reason_about_logic(model, premises, query):
    """Reason about logical statements."""
    
    # Encode premises
    premise_text = " ".join(premises)
    query_text = query
    full_text = f"Premises: {premise_text} Query: {query_text}"
    
    input_ids = tokenizer.encode(full_text, return_tensors='pt')
    
    # Get model output with reasoning
    logits, info = model(input_ids, return_all_outputs=True)
    
    # Extract reasoning info
    reasoning_output = info['module_outputs']['reason']
    
    return reasoning_output

# Example
premises = [
    "All mammals are animals.",
    "All dogs are mammals."
]
query = "Are dogs animals?"

result = reason_about_logic(model, premises, query)
```

### Mathematical Reasoning

```python
def solve_math(model, problem):
    """Solve math problem with reasoning."""
    
    prompt = f"Problem: {problem}\nSolution:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate with reasoning emphasis
    model.reasoner.max_steps = 10  # More reasoning steps
    
    logits, info = model(input_ids, return_all_outputs=True)
    
    # Get step-by-step solution
    return generate_text(model, tokenizer, prompt, max_tokens=200)

# Example
problem = "If x + 5 = 12, what is x?"
solution = solve_math(model, problem)
print(solution)
```

---

## Causal Inference

### Understanding Causal Queries

NEXUS supports three levels of causal queries:

1. **Observational**: P(Y|X) - What do we expect given observation?
2. **Interventional**: P(Y|do(X)) - What happens if we force X?
3. **Counterfactual**: P(Y_x|X=x',Y=y') - What if X had been different?

### Interventional Query

```python
# Access causal engine
causal = model.causal

# Query: What is effect of variable 0 on variable 5?
result = causal.intervene(
    variable=0,
    value=torch.tensor([1.0]),
    target=5,
    num_samples=100
)

print(f"Expected effect: {result['mean'].item():.4f}")
print(f"Uncertainty: {result['std'].item():.4f}")
```

### Counterfactual Query

```python
# Observe current state
observed = {
    0: torch.tensor([0.5]),  # Variable 0 = 0.5
    5: torch.tensor([0.8])   # Variable 5 = 0.8
}

# Ask: What would variable 5 be if variable 0 had been 1.0?
counterfactual_result = causal.counterfactual(
    observed=observed,
    intervention=(0, torch.tensor([1.0])),
    target=5
)

print(f"Counterfactual value: {counterfactual_result.item():.4f}")
```

### Causal Planning

```python
def plan_to_achieve_goal(model, current_state, goal_variable, goal_value):
    """Plan interventions to achieve goal."""
    
    causal = model.causal
    
    # Find best intervention
    actionable = [0, 1, 2, 3]  # Variables we can control
    
    best_intervention = None
    best_distance = float('inf')
    
    for var in actionable:
        for value in torch.linspace(-1, 1, 20):
            result = causal.intervene(var, value.unsqueeze(0), goal_variable)
            distance = (result['mean'] - goal_value).abs().item()
            
            if distance < best_distance:
                best_distance = distance
                best_intervention = (var, value.item())
    
    return best_intervention, best_distance

# Example
intervention, distance = plan_to_achieve_goal(
    model,
    current_state=None,
    goal_variable=5,
    goal_value=0.9
)

print(f"Best intervention: Set variable {intervention[0]} to {intervention[1]:.2f}")
print(f"Expected distance to goal: {distance:.4f}")
```

---

## Custom Configurations

### Small Model (For Testing)

```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

tiny_config = NEXUSConfig(
    vocab_size=10000,
    d_model=64,
    d_latent=32,
    ssm_n_layers=2,
    ssm_d_state=16,
    max_reasoning_steps=2,
    max_energy_iters=3
)
tiny_model = NEXUSCore(tiny_config)

print(f"Parameters: {sum(p.numel() for p in tiny_model.parameters()):,}")
# ~1M parameters
```

### Large Model (For Production)

```python
large_config = NEXUSConfig(
    vocab_size=50000,
    d_model=512,
    d_latent=256,
    ssm_n_layers=12,
    ssm_d_state=128,
    max_reasoning_steps=10,
    max_energy_iters=15,
    n_variables=64
)
large_model = NEXUSCore(large_config)

print(f"Parameters: {sum(p.numel() for p in large_model.parameters()):,}")
# ~100M+ parameters
```

### Reasoning-Focused Model

```python
reasoning_config = NEXUSConfig(
    vocab_size=32000,
    d_model=256,
    ssm_n_layers=6,
    
    # Emphasize reasoning
    max_reasoning_steps=15,
    n_variables=64
)
reasoning_model = NEXUSCore(reasoning_config)
```

### Efficiency-Focused Model

```python
efficient_config = NEXUSConfig(
    vocab_size=32000,
    d_model=128,
    ssm_n_layers=4,
    ssm_d_state=32,
    
    # Minimize computation
    max_reasoning_steps=3,
    max_energy_iters=5
)
efficient_model = NEXUSCore(efficient_config)
```

### Custom Module Configuration

```python
# After creating model, customize modules
config = NEXUSConfig(vocab_size=32000)
model = NEXUSCore(config)

# Adjust energy module
model.energy.refinement.max_iterations = 20
model.energy.refinement.convergence_threshold = 0.001

# Adjust reasoner
model.reasoner.config.max_steps = 15
model.reasoner.config.temperature = 0.05
```

---

## Tips and Best Practices

### 1. Start Small
```python
# Prototype with small model
config = NEXUSConfig(d_model=64, ssm_n_layers=2)
model = NEXUSCore(config)
# Then scale up
```

### 2. Monitor Energy Iterations
```python
# Check if model is using adaptive computation
x = torch.randn(1, 100, config.d_model)
_, info = model(x, return_all_outputs=True)
print(f"Iterations: {info['energy_info']['iterations']}")
# Should vary with input complexity
```

### 3. Use Return All Outputs for Debugging
```python
x = torch.randn(1, 100, config.d_model)
outputs, info = model(x, return_all_outputs=True)
for name, output in info['module_outputs'].items():
    print(f"{name}: mean={output.mean():.4f}, std={output.std():.4f}")
```

### 4. Validate Causal Structure
```python
# Check if causal graph is DAG
adjacency = model.causal.get_graph()
is_valid = model.causal.graph.is_dag()
print(f"Valid DAG: {is_valid}")
```

### 5. Balance Loss Weights
```python
# If one loss dominates, adjust weights
loss_fn = NEXUSLoss(
    lm_weight=1.0,
    world_weight=0.1,  # Reduce if too high
    reason_weight=0.05,
    energy_weight=0.05,
    causal_weight=0.05
)
```

---

## Next Steps

- [API Reference](../api/reference.md) - Complete API documentation
- [Architecture](../architecture/overview.md) - Understand the design
- [Research](../research/motivation.md) - Learn the theory

---

*Learn by doing. Build with NEXUS.*
