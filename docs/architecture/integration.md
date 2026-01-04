# Component Integration

## How NEXUS Components Work Together

This document explains how the five core NEXUS components—State Space, World Model, Reasoning, Energy, and Causal—integrate into a unified, coherent architecture.

---

## The Integration Challenge

Each component excels at different capabilities:

| Component | Strength |
|-----------|----------|
| State Space | Efficient sequence processing |
| World Model | Predictive representations |
| Reasoning | Symbolic manipulation |
| Energy | Adaptive computation |
| Causal | Cause-effect understanding |

**The challenge**: Combine them without losing individual strengths.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEXUS Integration Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input Sequence: x₁, x₂, ..., xₙ                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                    Input Embedding                          │          │
│   │                    d_model = 256                            │          │
│   └─────────────────────────────────────────────────────────────┘          │
│          │                                                                  │
│          ▼                                                                  │
│   ╔═════════════════════════════════════════════════════════════╗          │
│   ║              BACKBONE: Selective State Space                 ║          │
│   ║                                                             ║          │
│   ║    • O(n) linear-time processing                            ║          │
│   ║    • Long-range dependencies                                ║          │
│   ║    • Content-aware gating                                   ║          │
│   ║                                                             ║          │
│   ║    Output: h₁, h₂, ..., hₙ  (hidden states)                ║          │
│   ╚═════════════════════════════════════════════════════════════╝          │
│          │                                                                  │
│          ├────────────────┬────────────────┬────────────────┐              │
│          │                │                │                │              │
│          ▼                ▼                ▼                ▼              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │   World     │  │   Neuro-    │  │   Energy    │  │   Causal    │      │
│   │   Model     │  │   Symbolic  │  │   Module    │  │   Engine    │      │
│   │             │  │   Reasoner  │  │             │  │             │      │
│   │  Predict    │  │  Symbolic   │  │  Adaptive   │  │  Cause &    │      │
│   │  futures    │  │  reasoning  │  │  compute    │  │  effect     │      │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│          │                │                │                │              │
│          └────────────────┴────────────────┴────────────────┘              │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                    Fusion Layer                              │          │
│   │                                                              │          │
│   │    Combines outputs via gated attention:                     │          │
│   │    output = Σ αᵢ · moduleᵢ(h)                               │          │
│   │    where α = softmax(gate(h))                               │          │
│   │                                                              │          │
│   └─────────────────────────────────────────────────────────────┘          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                    Output Head                               │          │
│   │             (Task-specific: LM, Classification, etc.)        │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Detail

### Phase 1: Sequence Encoding

```
Input tokens → Embedding → State Space Backbone
                              │
                              ▼
                        Hidden states h ∈ ℝ^(B×N×D)
```

The State Space module provides the **foundation**:
- Processes entire sequence in O(n) time
- Captures long-range dependencies
- Produces rich hidden representations

### Phase 2: Parallel Module Processing

All specialist modules receive the hidden states **in parallel**:

```python
class ParallelModules(nn.Module):
    """Process hidden states through all modules in parallel."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.world_model = HierarchicalWorldModel(hidden_dim)
        self.reasoner = NeuroSymbolicReasoner(hidden_dim)
        self.energy = AdaptiveEnergyModule(hidden_dim)
        self.causal = CausalInferenceEngine(hidden_dim)
    
    def forward(self, hidden_states):
        # All run in parallel (can be batched on GPU)
        world_out = self.world_model(hidden_states)
        reason_out = self.reasoner(hidden_states)
        energy_out, energy_info = self.energy(hidden_states)
        causal_out = self.causal(hidden_states)
        
        return {
            'world': world_out,
            'reason': reason_out,
            'energy': energy_out,
            'causal': causal_out,
            'energy_info': energy_info
        }
```

### Phase 3: Gated Fusion

Combine module outputs intelligently:

```python
class GatedFusion(nn.Module):
    """Dynamically weight module contributions."""
    
    def __init__(self, hidden_dim, num_modules=4):
        super().__init__()
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modules)
        )
        
        # Per-module projection
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_modules)
        ])
    
    def forward(self, hidden_states, module_outputs):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] - original
            module_outputs: dict with module outputs
            
        Returns:
            fused: [batch, seq_len, hidden_dim]
        """
        # Compute gate weights based on content
        gate_input = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # [batch, 4]
        
        # Project and weight each module output
        outputs = [
            module_outputs['world'],
            module_outputs['reason'],
            module_outputs['energy'],
            module_outputs['causal']
        ]
        
        fused = torch.zeros_like(hidden_states)
        for i, (proj, out) in enumerate(zip(self.projections, outputs)):
            weight = gate_weights[:, i].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            fused += weight * proj(out)
        
        return fused
```

---

## Information Flow Between Modules

### Cross-Module Communication

Modules can inform each other:

```
                    ┌──────────────────────────────────────┐
                    │          Cross-Module Bus            │
                    └──────────────────────────────────────┘
                         ▲          ▲          ▲          ▲
                         │          │          │          │
              ┌──────────┴──────────┴──────────┴──────────┴──────────┐
              │                                                       │
    ┌─────────┴─────────┐      ┌──────────┐      ┌──────────┐      ┌─────────┐
    │    World Model    │ ◄──► │ Reasoner │ ◄──► │  Energy  │ ◄──► │ Causal  │
    │                   │      │          │      │          │      │         │
    │ • Future states   │      │ • Rules  │      │ • Cost   │      │ • Graph │
    │ • Uncertainty     │      │ • Proofs │      │ • Halt   │      │ • do()  │
    └───────────────────┘      └──────────┘      └──────────┘      └─────────┘
```

### Example Interactions

**World Model → Reasoner**:
```python
# Predicted future state helps reasoning
future_state = world_model.predict(hidden_states, horizon=5)
# Reasoner uses future context
reason_output = reasoner(hidden_states, context=future_state)
```

**Energy → All Modules**:
```python
# Energy determines computation budget
_, energy_info = energy_module(hidden_states)
iterations_needed = energy_info['iterations']

# Other modules adjust complexity
if iterations_needed < 3:
    # Simple input - use fast paths
    reasoner.use_shallow_search()
    causal.use_cached_graph()
```

**Causal → World Model**:
```python
# Causal structure informs predictions
causal_graph = causal_engine.get_graph(hidden_states)
# World model uses causal ordering for prediction
world_model.predict(hidden_states, causal_structure=causal_graph)
```

---

## NEXUSCore Implementation

```python
class NEXUSCore(nn.Module):
    """Complete NEXUS integration."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 256,
        num_layers: int = 6,
        state_dim: int = 64,
        expansion_factor: int = 2,
        num_reasoning_steps: int = 5,
        num_energy_iterations: int = 10,
        num_causal_vars: int = 32
    ):
        super().__init__()
        
        # Input embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # === BACKBONE ===
        self.state_space = SelectiveStateSpace(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor
        )
        
        # === SPECIALIST MODULES ===
        self.world_model = HierarchicalWorldModel(
            hidden_dim=hidden_dim,
            num_levels=3,
            prediction_horizon=10
        )
        
        self.reasoner = NeuroSymbolicReasoner(
            hidden_dim=hidden_dim,
            num_predicates=64,
            num_constants=128,
            max_steps=num_reasoning_steps
        )
        
        self.energy = AdaptiveEnergyModule(
            hidden_dim=hidden_dim,
            max_iterations=num_energy_iterations,
            convergence_threshold=0.01
        )
        
        self.causal = CausalInferenceEngine(
            hidden_dim=hidden_dim,
            num_variables=num_causal_vars
        )
        
        # === INTEGRATION ===
        self.fusion = GatedFusion(hidden_dim, num_modules=4)
        
        # === OUTPUT ===
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_all_outputs: bool = False
    ):
        """
        Full NEXUS forward pass.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            return_all_outputs: Whether to return module outputs
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            (optional) all_outputs: dict with module outputs
        """
        # Embedding
        x = self.embedding(input_ids)  # [B, N, D]
        
        # Phase 1: Backbone processing
        hidden_states = self.state_space(x)  # [B, N, D]
        
        # Phase 2: Parallel module processing
        world_out = self.world_model(hidden_states)
        reason_out = self.reasoner(hidden_states)
        energy_out, energy_info = self.energy(hidden_states)
        causal_out = self.causal(hidden_states)
        
        module_outputs = {
            'world': world_out,
            'reason': reason_out,
            'energy': energy_out,
            'causal': causal_out
        }
        
        # Phase 3: Gated fusion
        fused = self.fusion(hidden_states, module_outputs)
        
        # Residual connection with backbone
        output = hidden_states + fused
        
        # Output projection
        output = self.output_norm(output)
        logits = self.output_head(output)
        
        if return_all_outputs:
            return logits, {
                'module_outputs': module_outputs,
                'energy_info': energy_info,
                'backbone_states': hidden_states
            }
        
        return logits
```

---

## Training Integration

### Multi-Objective Loss

```python
class NEXUSLoss(nn.Module):
    """Combined training objective for all modules."""
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        world_weight: float = 0.1,
        reason_weight: float = 0.05,
        energy_weight: float = 0.05,
        causal_weight: float = 0.05
    ):
        super().__init__()
        self.weights = {
            'lm': lm_weight,
            'world': world_weight,
            'reason': reason_weight,
            'energy': energy_weight,
            'causal': causal_weight
        }
        
        # Individual losses
        self.lm_loss = nn.CrossEntropyLoss()
        self.world_loss = WorldModelLoss()
        self.reason_loss = ReasoningLoss()
        self.energy_loss = EnergyLoss()
        self.causal_loss = CausalLoss()
    
    def forward(self, model_output, targets, all_outputs=None):
        """Compute combined loss."""
        
        losses = {}
        
        # Main language modeling loss
        losses['lm'] = self.lm_loss(
            model_output.view(-1, model_output.size(-1)),
            targets.view(-1)
        )
        
        if all_outputs is not None:
            # World model: prediction accuracy
            losses['world'] = self.world_loss(
                all_outputs['module_outputs']['world'],
                targets
            )
            
            # Reasoning: consistency
            losses['reason'] = self.reason_loss(
                all_outputs['module_outputs']['reason']
            )
            
            # Energy: refinement quality
            losses['energy'] = self.energy_loss(
                all_outputs['energy_info']
            )
            
            # Causal: DAG constraint + reconstruction
            losses['causal'] = self.causal_loss(
                all_outputs['module_outputs']['causal']
            )
        
        # Weighted combination
        total_loss = sum(
            self.weights.get(name, 0.0) * loss
            for name, loss in losses.items()
        )
        
        return total_loss, losses
```

---

## Inference Modes

### Standard Inference

```python
def standard_inference(model, input_ids):
    """Normal forward pass."""
    return model(input_ids)
```

### Reasoning-Heavy Inference

```python
def reasoning_inference(model, input_ids, max_reasoning_steps=10):
    """Emphasize reasoning for complex queries."""
    # Increase reasoning iterations
    original_steps = model.reasoner.max_steps
    model.reasoner.max_steps = max_reasoning_steps
    
    output, info = model(input_ids, return_all_outputs=True)
    
    # Restore
    model.reasoner.max_steps = original_steps
    
    return output, info['module_outputs']['reason']
```

### Planning Inference

```python
def planning_inference(model, input_ids, goal_representation):
    """Use causal planning to achieve goal."""
    # Get current state
    hidden = model.state_space(model.embedding(input_ids))
    
    # Causal planning
    plan = model.causal.plan(
        current_state=hidden,
        goal=goal_representation
    )
    
    return plan
```

### Adaptive Inference

```python
def adaptive_inference(model, input_ids, budget=None):
    """Adapt computation to input complexity."""
    output, info = model(input_ids, return_all_outputs=True)
    
    # Check energy iterations as complexity proxy
    iterations = info['energy_info']['iterations']
    
    if iterations < 3:
        # Simple - output directly
        return output
    elif iterations < 7:
        # Medium - one refinement pass
        return model.energy.refine(output, steps=1)
    else:
        # Complex - full reasoning
        return reasoning_inference(model, input_ids, max_reasoning_steps=15)
```

---

## Module Scheduling

### Sequential Scheduling

Run modules in sequence for dependencies:

```python
def sequential_forward(model, hidden_states):
    """Modules in sequence, each informs next."""
    
    # 1. World model predicts futures
    world_out = model.world_model(hidden_states)
    
    # 2. Causal extracts structure from predictions
    causal_out = model.causal(torch.cat([hidden_states, world_out], dim=-1))
    
    # 3. Reasoner uses causal structure
    reason_out = model.reasoner(hidden_states, causal_graph=causal_out)
    
    # 4. Energy refines final output
    energy_out, _ = model.energy(reason_out)
    
    return energy_out
```

### Parallel Scheduling (Default)

Run modules in parallel for speed:

```python
def parallel_forward(model, hidden_states):
    """All modules in parallel, fuse at end."""
    
    # Parallel execution
    outputs = {
        'world': model.world_model(hidden_states),
        'reason': model.reasoner(hidden_states),
        'energy': model.energy(hidden_states)[0],
        'causal': model.causal(hidden_states)
    }
    
    # Fuse
    return model.fusion(hidden_states, outputs)
```

### Iterative Scheduling

Iteratively refine through modules:

```python
def iterative_forward(model, hidden_states, num_iterations=3):
    """Iterate through modules for refinement."""
    
    current = hidden_states
    
    for _ in range(num_iterations):
        # Each iteration refines through all modules
        world_out = model.world_model(current)
        reason_out = model.reasoner(world_out)
        energy_out, _ = model.energy(reason_out)
        causal_out = model.causal(energy_out)
        
        # Update current state
        current = current + 0.1 * causal_out  # Small update
    
    return current
```

---

## Best Practices

### 1. Start with Backbone

Always let State Space process first:
```python
hidden = state_space(x)  # Foundation
# Then specialist modules
```

### 2. Use Gates for Routing

Let the model learn when to use each module:
```python
# Good: learned gating
fused = gated_fusion(hidden, outputs)

# Bad: hardcoded combination  
fused = 0.25 * world + 0.25 * reason + 0.25 * energy + 0.25 * causal
```

### 3. Monitor Module Usage

Track which modules are active:
```python
gate_stats = model.fusion.get_gate_statistics()
print(f"World: {gate_stats['world']:.2%}")
print(f"Reason: {gate_stats['reason']:.2%}")
# etc.
```

### 4. Balance Training

Ensure all modules receive gradients:
```python
# Check gradient flow
for name, module in model.named_children():
    if any(p.grad is not None for p in module.parameters()):
        print(f"✓ {name} receiving gradients")
```

---

## Summary

The NEXUS integration follows these principles:

1. **Backbone first**: State Space provides the foundation
2. **Parallel specialists**: Each module processes independently
3. **Gated fusion**: Content-aware combination of outputs
4. **Cross-module communication**: Modules can inform each other
5. **Multi-objective training**: All modules trained jointly

This architecture allows NEXUS to dynamically leverage the right capabilities for each input while maintaining efficient, unified processing.

---

*The whole is greater than the sum of its parts. NEXUS proves it.*
