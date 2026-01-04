# Adaptive Energy Module

## Dynamic Computation Through Energy-Based Refinement

The Adaptive Energy Module enables NEXUS to allocate computation based on input complexity—thinking harder on difficult problems and efficiently handling simple ones.

---

## Motivation

### The Fixed Computation Problem

Standard neural networks apply identical computation to all inputs:

```
"What is 2+2?" → 12 layers × N operations
"Prove the Riemann hypothesis" → 12 layers × N operations
```

This is **wasteful** for simple inputs and **insufficient** for complex ones.

### The Energy-Based Solution

NEXUS uses energy as a measure of "difficulty" or "uncertainty":
- **High energy** = uncertain, needs more processing
- **Low energy** = confident, can stop early

---

## Energy-Based Models: Core Concepts

### Energy Function

An energy function maps configurations to scalar values:

$$
E_\theta(x, y) : \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}
$$

**Interpretation**:
- Lower energy = more compatible/likely configuration
- Higher energy = less compatible/unlikely configuration

### Inference as Optimization

Instead of direct prediction, find the output that minimizes energy:

$$
\hat{y} = \arg\min_y E_\theta(x, y)
$$

This is solved via iterative refinement (gradient descent on y).

---

## Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Adaptive Energy Module                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input State (from backbone)                                           │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │              Initial Refinement State                    │          │
│   │                    y₀ = f(x)                            │          │
│   └─────────────────────────────────────────────────────────┘          │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │              Iterative Refinement Loop                   │          │
│   │                                                         │          │
│   │   for t = 1 to T:                                       │          │
│   │     ┌─────────────────────────────────────────┐        │          │
│   │     │   Compute Energy: E_t = E_θ(x, y_{t-1}) │        │          │
│   │     └─────────────────────────────────────────┘        │          │
│   │                        │                                │          │
│   │                        ▼                                │          │
│   │     ┌─────────────────────────────────────────┐        │          │
│   │     │   Check Convergence:                    │        │          │
│   │     │   if E_t < ε or ‖∇E‖ < δ: STOP        │        │          │
│   │     └─────────────────────────────────────────┘        │          │
│   │                        │                                │          │
│   │                        ▼ (if not converged)             │          │
│   │     ┌─────────────────────────────────────────┐        │          │
│   │     │   Gradient Step:                        │        │          │
│   │     │   y_t = y_{t-1} - η · ∇_y E_θ(x, y)   │        │          │
│   │     └─────────────────────────────────────────┘        │          │
│   │                                                         │          │
│   └─────────────────────────────────────────────────────────┘          │
│          │                                                              │
│          ├─────────────────────────────────────────┐                   │
│          │                                         │                   │
│          ▼                                         ▼                   │
│   ┌─────────────────┐                    ┌─────────────────┐           │
│   │ Refined Output  │                    │ Energy History  │           │
│   │      y_T        │                    │ [E₁, E₂, ..Eₜ]  │           │
│   │                 │                    │                 │           │
│   │ (final state)   │                    │ (for analysis)  │           │
│   └─────────────────┘                    └─────────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Energy Function Design

### Architecture

```python
class EnergyFunction(nn.Module):
    """Neural energy function E(x, y) → scalar."""
    
    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        
        # Joint encoder for (x, y)
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Energy network
        self.energy_net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])
        
        # Final scalar output
        self.energy_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, y):
        """Compute energy for configuration (x, y).
        
        Args:
            x: [batch, hidden_dim] - input/context
            y: [batch, hidden_dim] - output/state to evaluate
            
        Returns:
            energy: [batch] - scalar energy values
        """
        # Concatenate and encode jointly
        joint = torch.cat([x, y], dim=-1)
        h = self.joint_encoder(joint)
        
        # Compute energy
        h = self.energy_net(h) + h  # Residual
        energy = self.energy_head(h).squeeze(-1)
        
        return energy
```

### Energy Landscape Properties

Good energy functions should have:

1. **Low energy at correct solutions**
2. **High energy far from solutions**
3. **Smooth gradients for optimization**
4. **Multiple valid minima (if appropriate)**

```python
class EnergyRegularizer:
    """Regularization for good energy landscapes."""
    
    @staticmethod
    def smoothness_loss(energy_fn, x, y, epsilon=0.01):
        """Encourage smooth energy landscape."""
        # Perturb y slightly
        y_perturbed = y + torch.randn_like(y) * epsilon
        
        # Energy should change smoothly
        e1 = energy_fn(x, y)
        e2 = energy_fn(x, y_perturbed)
        
        return (e1 - e2).abs().mean()
    
    @staticmethod
    def contrastive_loss(energy_fn, x, y_pos, y_neg, margin=1.0):
        """Push positive examples lower, negative higher."""
        e_pos = energy_fn(x, y_pos)
        e_neg = energy_fn(x, y_neg)
        
        # Margin loss
        return F.relu(margin + e_pos - e_neg).mean()
```

---

## Iterative Refinement

### Gradient-Based Refinement

```python
class IterativeRefinement(nn.Module):
    """Iterative refinement via energy minimization."""
    
    def __init__(
        self,
        energy_fn,
        max_iterations=10,
        step_size=0.1,
        convergence_threshold=0.01,
        gradient_threshold=0.001
    ):
        super().__init__()
        self.energy_fn = energy_fn
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.convergence_threshold = convergence_threshold
        self.gradient_threshold = gradient_threshold
    
    def forward(self, x, y_init):
        """Refine y_init by minimizing energy.
        
        Args:
            x: [batch, hidden_dim] - conditioning input
            y_init: [batch, hidden_dim] - initial state
            
        Returns:
            y_refined: [batch, hidden_dim] - refined state
            info: dict with energy history and iterations
        """
        y = y_init.clone()
        y.requires_grad_(True)
        
        energy_history = []
        
        for t in range(self.max_iterations):
            # Compute energy
            energy = self.energy_fn(x, y)
            energy_history.append(energy.mean().item())
            
            # Check convergence
            if energy.mean() < self.convergence_threshold:
                break
            
            # Compute gradient
            grad = torch.autograd.grad(
                energy.sum(),
                y,
                create_graph=self.training
            )[0]
            
            # Check gradient magnitude
            if grad.norm() < self.gradient_threshold:
                break
            
            # Gradient descent step
            y = y - self.step_size * grad
            
            # Optional: projection to valid space
            y = self.project(y)
        
        return y.detach() if not self.training else y, {
            'energy_history': energy_history,
            'iterations': t + 1,
            'final_energy': energy_history[-1]
        }
    
    def project(self, y):
        """Project y to valid range (optional)."""
        # Example: L2 normalization
        return F.normalize(y, dim=-1)
```

### Langevin Dynamics (Stochastic Refinement)

```python
class LangevinRefinement(nn.Module):
    """Stochastic refinement using Langevin dynamics."""
    
    def __init__(self, energy_fn, step_size=0.01, noise_scale=0.01):
        super().__init__()
        self.energy_fn = energy_fn
        self.step_size = step_size
        self.noise_scale = noise_scale
    
    def forward(self, x, y_init, num_steps=50):
        """
        Langevin dynamics: y_t = y_{t-1} - η∇E + √(2η)ε
        
        The noise term helps escape local minima.
        """
        y = y_init.clone()
        
        for _ in range(num_steps):
            y.requires_grad_(True)
            
            energy = self.energy_fn(x, y)
            grad = torch.autograd.grad(energy.sum(), y)[0]
            
            # Langevin update
            noise = torch.randn_like(y) * self.noise_scale
            y = y - self.step_size * grad + (2 * self.step_size) ** 0.5 * noise
            
            y = y.detach()
        
        return y
```

---

## Adaptive Computation Depth

### Early Exit Mechanism

```python
class AdaptiveEnergyModule(nn.Module):
    """Full adaptive energy module with early exit."""
    
    def __init__(
        self,
        hidden_dim,
        max_iterations=10,
        energy_threshold=0.1,
        confidence_threshold=0.95
    ):
        super().__init__()
        
        # Energy function
        self.energy_fn = EnergyFunction(hidden_dim)
        
        # Initial state generator
        self.init_state = nn.Linear(hidden_dim, hidden_dim)
        
        # Refinement
        self.refiner = IterativeRefinement(
            self.energy_fn,
            max_iterations=max_iterations
        )
        
        # Confidence predictor (optional fast path)
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.energy_threshold = energy_threshold
        self.confidence_threshold = confidence_threshold
    
    def forward(self, x, force_full=False):
        """
        Adaptive forward pass.
        
        Args:
            x: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            force_full: If True, always do full refinement
            
        Returns:
            output: Refined representation
            info: Computation statistics
        """
        # Pool if sequence
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)
        else:
            x_pooled = x
        
        # Quick confidence check (fast path)
        if not force_full:
            confidence = self.confidence_predictor(x_pooled)
            
            # If confident, skip refinement
            if (confidence > self.confidence_threshold).all():
                return self.init_state(x_pooled), {
                    'iterations': 0,
                    'skipped': True,
                    'confidence': confidence.mean().item()
                }
        
        # Initialize refinement state
        y_init = self.init_state(x_pooled)
        
        # Iterative refinement
        y_refined, refine_info = self.refiner(x_pooled, y_init)
        
        # Expand back to sequence if needed
        if x.dim() == 3:
            y_refined = y_refined.unsqueeze(1).expand(-1, x.size(1), -1)
            # Or use attention to distribute back
        
        return y_refined, {
            **refine_info,
            'skipped': False
        }
```

### Computation Budget Aware

```python
class BudgetAwareEnergy(nn.Module):
    """Energy refinement with computation budget."""
    
    def __init__(self, energy_module, default_budget=5):
        super().__init__()
        self.energy_module = energy_module
        self.default_budget = default_budget
    
    def forward(self, x, budget=None):
        """
        Refine within computation budget.
        
        Args:
            x: Input
            budget: Max iterations (None = use default)
        """
        budget = budget or self.default_budget
        
        # Adjust refinement based on budget
        self.energy_module.refiner.max_iterations = budget
        
        return self.energy_module(x)
    
    def estimate_required_budget(self, x):
        """Estimate iterations needed for this input."""
        with torch.no_grad():
            # Quick probe: one step, check energy drop
            y_init = self.energy_module.init_state(x)
            e_init = self.energy_module.energy_fn(x, y_init)
            
            # One refinement step
            y_step, _ = self.energy_module.refiner(x, y_init)
            self.energy_module.refiner.max_iterations = 1
            e_step = self.energy_module.energy_fn(x, y_step)
            
            # Estimate based on energy reduction rate
            reduction_rate = (e_init - e_step) / e_init
            
            if reduction_rate > 0.3:
                return 3  # Fast convergence
            elif reduction_rate > 0.1:
                return 7  # Medium
            else:
                return 15  # Slow convergence
```

---

## Training Energy-Based Models

### Contrastive Learning

```python
class ContrastiveEnergyLoss(nn.Module):
    """Train energy function with contrastive loss."""
    
    def __init__(self, margin=1.0, num_negatives=10):
        super().__init__()
        self.margin = margin
        self.num_negatives = num_negatives
    
    def forward(self, energy_fn, x, y_positive):
        """
        Push positive pairs to low energy,
        negative pairs to high energy.
        """
        batch_size = x.size(0)
        
        # Positive energy
        e_pos = energy_fn(x, y_positive)
        
        # Generate negatives
        negatives = []
        for _ in range(self.num_negatives):
            # Option 1: Random perturbation
            y_neg = y_positive + torch.randn_like(y_positive) * 0.5
            
            # Option 2: Shuffle within batch
            # y_neg = y_positive[torch.randperm(batch_size)]
            
            negatives.append(y_neg)
        
        # Negative energies
        e_neg = torch.stack([
            energy_fn(x, y_neg) for y_neg in negatives
        ], dim=1)  # [batch, num_neg]
        
        # Contrastive loss: e_pos should be lower than e_neg by margin
        loss = F.relu(self.margin + e_pos.unsqueeze(1) - e_neg).mean()
        
        return loss
```

### Score Matching

```python
class ScoreMatchingLoss(nn.Module):
    """Train with denoising score matching."""
    
    def __init__(self, noise_scale=0.1):
        super().__init__()
        self.noise_scale = noise_scale
    
    def forward(self, energy_fn, x, y):
        """
        Denoising score matching:
        Learn to denoise corrupted samples.
        """
        # Add noise
        noise = torch.randn_like(y) * self.noise_scale
        y_noisy = y + noise
        
        # Compute score (gradient of energy w.r.t. y)
        y_noisy.requires_grad_(True)
        energy = energy_fn(x, y_noisy)
        score = torch.autograd.grad(energy.sum(), y_noisy)[0]
        
        # Score should point toward clean sample
        # Optimal score = -noise / noise_scale^2
        target_score = -noise / (self.noise_scale ** 2)
        
        loss = F.mse_loss(score, target_score)
        
        return loss
```

---

## Integration with NEXUS

### Energy-Guided Generation

```python
class EnergyGuidedGeneration:
    """Use energy for generation guidance."""
    
    def __init__(self, nexus_model, energy_module, guidance_weight=1.0):
        self.model = nexus_model
        self.energy = energy_module
        self.guidance_weight = guidance_weight
    
    def generate(self, prompt, max_tokens=100):
        """Generate with energy-based guidance."""
        
        generated = prompt.clone()
        
        for _ in range(max_tokens):
            # Get model predictions
            hidden = self.model.state_space(generated)
            logits = self.model.output_head(hidden[:, -1:])
            
            # Get top-k candidates
            top_k_logits, top_k_indices = logits.topk(50)
            
            # Score candidates by energy
            candidate_energies = []
            for idx in top_k_indices[0, 0]:
                # Simulate adding this token
                candidate_hidden = hidden.clone()
                candidate_hidden[:, -1] = self.model.embedding(idx.unsqueeze(0))
                
                # Get energy (lower = better)
                _, energy_info = self.energy(candidate_hidden)
                candidate_energies.append(energy_info['final_energy'])
            
            candidate_energies = torch.tensor(candidate_energies)
            
            # Combine logits with negative energy
            adjusted_logits = top_k_logits[0, 0] - self.guidance_weight * candidate_energies
            
            # Sample
            probs = F.softmax(adjusted_logits, dim=-1)
            selected = torch.multinomial(probs, 1)
            next_token = top_k_indices[0, 0, selected]
            
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return generated
```

### Information Flow

```
State Space Output
       │
       ▼
┌──────────────────────┐
│  Initial State       │
│  y₀ = Linear(h)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Energy Check        │──── Low energy? → Early exit
│  E(h, y₀)            │
└──────────┬───────────┘
           │ High energy
           ▼
┌──────────────────────┐
│  Iterative Refine    │
│  y_t = y_{t-1} - η∇E │
│  until converged     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Refined Output      │
│  + Energy History    │
└──────────────────────┘
```

---

## Configuration

```python
energy_config = {
    # Energy function
    'energy_hidden_dim': 256,
    'energy_num_layers': 3,
    
    # Refinement
    'max_iterations': 10,
    'step_size': 0.1,
    'convergence_threshold': 0.01,
    'gradient_threshold': 0.001,
    
    # Adaptive computation
    'use_early_exit': True,
    'confidence_threshold': 0.95,
    'energy_threshold': 0.1,
    
    # Training
    'contrastive_margin': 1.0,
    'num_negatives': 10,
    'energy_loss_weight': 0.05,
}
```

---

## Benefits and Trade-offs

### Benefits

| Benefit | Description |
|---------|-------------|
| Adaptive | More compute for harder inputs |
| Efficient | Early exit for easy inputs |
| Principled | Energy = uncertainty measure |
| Flexible | Works with any backbone |

### Trade-offs

| Trade-off | Mitigation |
|-----------|------------|
| Iterative cost | Early exit reduces overhead |
| Training complexity | Contrastive loss is stable |
| Hyperparameter sensitivity | Defaults work well |

---

## Experimental Observations

### Iteration Distribution

Typical iteration counts on various inputs:
```
Simple factual questions: 1-2 iterations
Moderate reasoning: 3-5 iterations
Complex multi-step: 7-10 iterations
Ambiguous/uncertain: 10+ iterations (max)
```

### Energy-Accuracy Correlation

```
Final Energy    Accuracy
< 0.1           95%+
0.1 - 0.3       85-95%
0.3 - 0.5       70-85%
> 0.5           < 70%

Energy serves as reliable confidence estimate.
```

---

## Further Reading

- [Energy-Based Models Tutorial](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)
- [Score Matching](https://www.jmlr.org/papers/v6/hyvarinen05a.html)
- [Architecture Overview](overview.md)
- [Causal Engine](causal.md)

---

*Energy is nature's measure of change. NEXUS uses it to measure thought.*
