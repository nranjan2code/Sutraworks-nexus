# Tutorials

## Getting Started with NEXUS

Step-by-step tutorials for using NEXUS.

---

## Tutorial Index

1. [Quick Start](#quick-start) - Get up and running in 5 minutes
2. [Basic Usage](#basic-usage) - Core functionality
3. [Training Your First Model](#training-your-first-model) - End-to-end training
4. [Reasoning Tasks](#reasoning-tasks) - Using reasoning capabilities
5. [Causal Inference](#causal-inference) - Causal queries and planning
6. [Custom Configurations](#custom-configurations) - Customize NEXUS

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

### First Inference

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

That's it! You've run your first NEXUS inference.

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
