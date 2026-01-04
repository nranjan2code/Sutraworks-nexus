# Hierarchical World Model

## JEPA-Style Abstract Prediction in NEXUS

The Hierarchical World Model enables NEXUS to build abstract representations of the world and predict future states—a capability fundamental to intelligence. This document explains the architecture, theory, and implementation.

---

## Why World Models?

### The Token Prediction Paradigm

Traditional language models predict tokens:
$$
P(\text{next token} | \text{previous tokens})
$$

**Limitations**:
- Predicts surface form, not meaning
- No abstract understanding
- Cannot simulate consequences
- Brittle to paraphrasing

### The World Model Paradigm

NEXUS predicts in representation space:
$$
P(\text{future representation} | \text{current representation})
$$

**Benefits**:
- Captures abstract semantics
- Enables mental simulation
- Robust to surface variation
- Supports planning

---

## JEPA: Joint Embedding Predictive Architecture

### Core Principle

Instead of generating raw observations (tokens/pixels), predict in a learned representation space:

```
Traditional Generative:
  Input → Encoder → Latent → Decoder → Predicted Output
  (must model all details)

JEPA:
  Input → Encoder → Representation
                         ↓
  Target → Encoder → Representation ← Predictor
                    (compare in this space)
```

### Why Representation Space?

1. **Compression**: Representations capture essence, not details
2. **Invariance**: Same meaning maps to similar representations
3. **Efficiency**: Lower-dimensional prediction
4. **Robustness**: Noise in observation doesn't affect representation

---

## Architecture Deep Dive

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hierarchical World Model                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │  Context Input   │         │  Target Input    │                     │
│  │    (visible)     │         │   (to predict)   │                     │
│  └────────┬─────────┘         └────────┬─────────┘                     │
│           │                            │                                │
│           ▼                            ▼                                │
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │  Context Encoder │         │  Target Encoder  │                     │
│  │      f_θ         │         │      f_ξ (EMA)   │                     │
│  │  (trainable)     │         │  (exponential    │                     │
│  │                  │         │   moving avg)    │                     │
│  └────────┬─────────┘         └────────┬─────────┘                     │
│           │                            │                                │
│           │ context repr               │ target repr                   │
│           │    z_c                     │    z_t                        │
│           │                            │                                │
│           ▼                            │                                │
│  ┌──────────────────┐                  │                                │
│  │    Predictor     │                  │                                │
│  │       g_φ        │──────────────────┼── Compare                     │
│  │                  │    predicted     │   (L2 loss)                   │
│  └────────┬─────────┘        ↓         ↓                               │
│           │              ┌─────────────────┐                           │
│           │              │   JEPA Loss     │                           │
│           │              │ ||pred - tgt||² │                           │
│           │              └─────────────────┘                           │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │              Temporal Abstraction Hierarchy               │          │
│  │                                                          │          │
│  │  Level 3: ○─────○─────○─────○  (coarse, abstract)       │          │
│  │  Level 2: ○──○──○──○──○──○──○  (medium)                 │          │
│  │  Level 1: ○○○○○○○○○○○○○○○○○○○  (fine, detailed)          │          │
│  │                                                          │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Context Encoder

### Architecture

```python
class ContextEncoder(nn.Module):
    """Encode visible context into representation."""
    
    def __init__(self, hidden_dim, num_layers, num_heads):
        super().__init__()
        
        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Pooling for global context
        self.pool = AttentionPooling(hidden_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: Optional mask for context
            
        Returns:
            z: [batch, seq_len, hidden_dim] - contextualized representations
            z_global: [batch, hidden_dim] - global context vector
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        z = x
        z_global = self.pool(z)
        
        return z, z_global
```

### Key Design Choices

1. **Bidirectional Processing**: Context encoder sees all visible tokens
2. **Multiple Layers**: Deep encoding for rich representations
3. **Global Pooling**: Summary vector for predictor conditioning

---

## Target Encoder

### Exponential Moving Average (EMA)

The target encoder is an EMA copy of the context encoder:

$$
\xi \leftarrow \tau \xi + (1 - \tau) \theta
$$

Where:
- $\xi$ = target encoder parameters
- $\theta$ = context encoder parameters
- $\tau$ = decay rate (typically 0.99-0.999)

### Why EMA?

1. **Prevents Collapse**: Without EMA, both encoders can collapse to constant output
2. **Stable Targets**: Slowly-moving targets provide stable learning signal
3. **Self-Distillation**: Student (context) learns from teacher (target)

```python
class TargetEncoder(nn.Module):
    """EMA copy of context encoder."""
    
    def __init__(self, context_encoder, ema_decay=0.99):
        super().__init__()
        self.encoder = copy.deepcopy(context_encoder)
        self.ema_decay = ema_decay
        
        # Freeze parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, context_encoder):
        """Update with EMA."""
        for target_param, source_param in zip(
            self.encoder.parameters(),
            context_encoder.parameters()
        ):
            target_param.data = (
                self.ema_decay * target_param.data +
                (1 - self.ema_decay) * source_param.data
            )
    
    def forward(self, x):
        return self.encoder(x)
```

---

## Predictor

### Architecture

The predictor maps context representation to target representation:

```python
class Predictor(nn.Module):
    """Predict target representations from context."""
    
    def __init__(self, hidden_dim, predictor_dim, num_layers):
        super().__init__()
        
        # Project to predictor space
        self.input_proj = nn.Linear(hidden_dim, predictor_dim)
        
        # Prediction layers
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(predictor_dim, predictor_dim * 4),
                nn.GELU(),
                nn.Linear(predictor_dim * 4, predictor_dim),
                nn.LayerNorm(predictor_dim),
            )
            for _ in range(num_layers)
        ])
        
        # Project to output space
        self.output_proj = nn.Linear(predictor_dim, hidden_dim)
    
    def forward(self, z_context, z_global=None):
        """
        Args:
            z_context: [batch, seq_len, hidden_dim] - context representations
            z_global: [batch, hidden_dim] - optional global context
            
        Returns:
            prediction: [batch, hidden_dim] - predicted target representation
        """
        # Pool context
        z = z_context.mean(dim=1)  # or use attention pooling
        
        # Add global context if available
        if z_global is not None:
            z = z + z_global
        
        # Project and transform
        z = self.input_proj(z)
        z = self.layers(z) + z  # Residual
        prediction = self.output_proj(z)
        
        return prediction
```

### Asymmetric Design

The predictor is intentionally **simpler** than the encoders:
- Forces encoders to learn rich representations
- Prevents predictor from memorizing

---

## JEPA Loss

### Formulation

$$
\mathcal{L}_{\text{JEPA}} = \|g_\phi(f_\theta(x_{\text{context}})) - \text{sg}[f_\xi(x_{\text{target}})]\|^2
$$

Where:
- $g_\phi$ = predictor
- $f_\theta$ = context encoder
- $f_\xi$ = target encoder (EMA)
- $\text{sg}[\cdot]$ = stop gradient

```python
def jepa_loss(self, context, target):
    """Compute JEPA loss.
    
    Args:
        context: [batch, context_len, hidden_dim]
        target: [batch, target_len, hidden_dim]
        
    Returns:
        loss: scalar
    """
    # Encode context
    z_context, z_global = self.context_encoder(context)
    
    # Predict target representation
    prediction = self.predictor(z_context, z_global)
    
    # Encode target (no gradient)
    with torch.no_grad():
        z_target = self.target_encoder(target)
        z_target = z_target.mean(dim=1)  # Pool to single vector
    
    # L2 loss in representation space
    loss = F.mse_loss(prediction, z_target)
    
    return loss
```

---

## Temporal Abstraction Hierarchy

### Multi-Scale Processing

Real understanding requires multiple temporal scales:

| Scale | Processes | Example |
|-------|-----------|---------|
| Fine | Individual tokens | Word meanings |
| Medium | Phrases/clauses | Syntactic structures |
| Coarse | Sentences/paragraphs | Discourse themes |

### Implementation

```python
class TemporalAbstraction(nn.Module):
    """Multi-scale temporal abstraction."""
    
    def __init__(self, hidden_dim, num_levels=3, pool_sizes=[1, 4, 16]):
        super().__init__()
        self.num_levels = num_levels
        self.pool_sizes = pool_sizes
        
        # Level-specific projections
        self.level_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_levels)
        ])
        
        # Cross-level attention
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8)
            for _ in range(num_levels - 1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            
        Returns:
            hierarchical: List of [batch, level_len, hidden_dim]
        """
        levels = []
        
        for i, (proj, pool_size) in enumerate(zip(self.level_projs, self.pool_sizes)):
            # Pool to this level's resolution
            if pool_size > 1:
                x_pooled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=pool_size,
                    stride=pool_size
                ).transpose(1, 2)
            else:
                x_pooled = x
            
            # Project
            level_repr = proj(x_pooled)
            levels.append(level_repr)
        
        # Bottom-up aggregation
        for i in range(self.num_levels - 1):
            # Higher level attends to lower level
            levels[i + 1] = levels[i + 1] + self.cross_attention[i](
                levels[i + 1].transpose(0, 1),
                levels[i].transpose(0, 1),
                levels[i].transpose(0, 1)
            )[0].transpose(0, 1)
        
        return levels
```

---

## Multi-Step Prediction

### Imagination / Mental Simulation

The world model can predict multiple steps into the future:

```python
def predict_multi_step(self, context, num_steps=5):
    """Predict multiple future states.
    
    Args:
        context: [batch, seq_len, hidden_dim]
        num_steps: Number of future steps to predict
        
    Returns:
        predictions: List of [batch, hidden_dim] for each step
    """
    predictions = []
    
    # Initial encoding
    z_context, z_global = self.context_encoder(context)
    
    for step in range(num_steps):
        # Predict next state
        z_pred = self.predictor(z_context, z_global)
        predictions.append(z_pred)
        
        # Use prediction as new context for next step
        z_context = z_pred.unsqueeze(1).expand(-1, z_context.size(1), -1)
        # (In practice, would concatenate or use attention)
    
    return predictions
```

### Applications

1. **Planning**: Simulate actions before taking them
2. **Reasoning**: Trace consequences of assumptions
3. **Generation**: Guide generation toward desired outcomes
4. **Evaluation**: Score options by predicted consequences

---

## Training Strategy

### Masking Strategies

```python
class MaskingStrategy:
    """Different masking strategies for JEPA training."""
    
    @staticmethod
    def random_mask(seq_len, mask_ratio=0.3):
        """Random token masking."""
        mask = torch.rand(seq_len) < mask_ratio
        return mask
    
    @staticmethod
    def block_mask(seq_len, num_blocks=4, block_size=None):
        """Contiguous block masking."""
        if block_size is None:
            block_size = seq_len // (num_blocks * 2)
        
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for _ in range(num_blocks):
            start = torch.randint(0, seq_len - block_size, (1,)).item()
            mask[start:start + block_size] = True
        return mask
    
    @staticmethod
    def causal_mask(seq_len, context_ratio=0.5):
        """Predict future from past."""
        context_len = int(seq_len * context_ratio)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[context_len:] = True  # Mask future
        return mask
```

### Loss Schedule

```python
class WorldModelTrainer:
    """Training schedule for world model."""
    
    def __init__(self, world_model, optimizer):
        self.world_model = world_model
        self.optimizer = optimizer
        
        # EMA schedule
        self.ema_schedule = lambda step: min(0.999, 0.99 + step * 1e-5)
    
    def train_step(self, batch, step):
        # Sample masking strategy
        strategy = random.choice(['random', 'block', 'causal'])
        mask = MaskingStrategy.get_mask(strategy, batch.shape[1])
        
        # Split context and target
        context = batch[:, ~mask]
        target = batch[:, mask]
        
        # Forward and loss
        loss = self.world_model.jepa_loss(context, target)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update EMA
        ema_decay = self.ema_schedule(step)
        self.world_model.target_encoder.update(
            self.world_model.context_encoder,
            decay=ema_decay
        )
        
        return loss.item()
```

---

## Integration with NEXUS

### Information Flow

```
State Space Output
       │
       ├───────────────────┐
       │                   │
       ▼                   ▼
  World Model         Other Modules
       │
       ├── Predictions (for planning)
       ├── Context repr (for reasoning)
       └── Abstraction levels (for multi-scale)
```

### Usage in Generation

```python
def generate_with_world_model(self, prompt, max_tokens):
    """Generate using world model guidance."""
    
    generated = prompt.clone()
    
    for _ in range(max_tokens):
        # Get state space representation
        hidden = self.state_space(generated)
        
        # Get world model prediction of where we should go
        target_repr = self.world_model.predict_next(hidden)
        
        # Score candidate tokens by alignment with prediction
        logits = self.output_head(hidden[:, -1:])
        
        # Adjust logits based on world model
        candidate_reprs = self.embedding.weight  # [vocab, hidden]
        alignment = F.cosine_similarity(
            target_repr.unsqueeze(1),
            candidate_reprs.unsqueeze(0),
            dim=-1
        )
        logits = logits + alignment * self.world_model_weight
        
        # Sample
        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

---

## Configuration

```python
world_model_config = {
    # Encoder
    'encoder_layers': 4,
    'encoder_heads': 8,
    
    # Predictor
    'predictor_layers': 2,
    'predictor_dim': 256,
    
    # EMA
    'ema_decay': 0.99,
    'ema_warmup_steps': 1000,
    
    # Temporal abstraction
    'num_levels': 3,
    'pool_sizes': [1, 4, 16],
    
    # Training
    'mask_ratio': 0.3,
    'loss_weight': 0.1,
}
```

---

## Key Insights

### Why JEPA Works

1. **No Decoder Needed**: Don't need to reconstruct raw input
2. **Collapse Prevention**: EMA target + predictor asymmetry
3. **Semantic Learning**: Representations capture meaning, not form
4. **Efficient**: Lower-dimensional prediction target

### Connection to Human Cognition

- **Mental Models**: Humans maintain internal world models
- **Prediction**: Constantly predicting what comes next
- **Abstraction**: Multiple levels of abstraction
- **Imagination**: Simulating hypotheticals

---

## Further Reading

- [JEPA Paper](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [I-JEPA Implementation](https://github.com/facebookresearch/ijepa)
- [World Models](https://worldmodels.github.io/)
- [Architecture Overview](overview.md)
- [Reasoning Module](reasoning.md)

---

*The world model is NEXUS's imagination—its ability to see what isn't yet.*
