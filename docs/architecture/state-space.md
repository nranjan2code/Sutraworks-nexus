# Selective State Space Architecture

## The O(n) Backbone of NEXUS

The Selective State Space module is the computational backbone of NEXUS, providing efficient sequence processing with linear complexity. This document explains the architecture, mathematics, and implementation in detail.

---

## Why State Space Models?

### The Attention Problem

Traditional Transformers compute attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The $QK^T$ computation requires **O(n²)** operations and memory.

### The State Space Solution

State space models process sequences by maintaining a compressed state:

$$
\begin{aligned}
h_t &= \bar{A}h_{t-1} + \bar{B}x_t \\
y_t &= Ch_t
\end{aligned}
$$

This requires only **O(n)** operations.

---

## Architecture Deep Dive

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Selective State Space Layer                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input x                                                   │
│      │                                                      │
│      ▼                                                      │
│   ┌─────────────────────────────────────┐                  │
│   │         Input Projection            │                  │
│   │    Linear(hidden_dim → expand_dim)  │                  │
│   └─────────────────────────────────────┘                  │
│      │                                                      │
│      ├──────────────────┬───────────────┐                  │
│      │                  │               │                  │
│      ▼                  ▼               ▼                  │
│   ┌──────┐         ┌──────┐        ┌──────┐               │
│   │  Δ   │         │  B   │        │  C   │               │
│   │Linear│         │Linear│        │Linear│               │
│   └──────┘         └──────┘        └──────┘               │
│      │                  │               │                  │
│      ▼                  ▼               ▼                  │
│   softplus           (input-dependent projections)         │
│      │                                                      │
│      └──────────────────┼───────────────┘                  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────┐                  │
│   │        Discretization                │                  │
│   │   Ā = exp(Δ·A), B̄ = Δ·B            │                  │
│   └─────────────────────────────────────┘                  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────┐                  │
│   │        Selective Scan                │                  │
│   │   h_t = Ā·h_{t-1} + B̄·x_t          │                  │
│   │   y_t = C·h_t                        │                  │
│   └─────────────────────────────────────┘                  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────┐                  │
│   │         Output Projection            │                  │
│   │    Linear(expand_dim → hidden_dim)   │                  │
│   └─────────────────────────────────────┘                  │
│                         │                                   │
│                         ▼                                   │
│   Output y (+ residual from input)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The Selection Mechanism

### Why Selection Matters

In traditional SSMs, parameters A, B, C are fixed. This limits expressivity:
- Cannot ignore irrelevant tokens
- Cannot emphasize important tokens
- Same processing regardless of content

### Input-Dependent Parameters

NEXUS makes B, C, and Δ depend on the input:

```python
def selective_parameters(self, x):
    """Compute input-dependent parameters."""
    # x: [batch, seq_len, hidden_dim]
    
    # Input-dependent projections
    B = self.B_proj(x)      # [batch, seq_len, state_dim]
    C = self.C_proj(x)      # [batch, seq_len, state_dim]
    delta = F.softplus(self.delta_proj(x))  # [batch, seq_len, hidden_dim]
    
    return B, C, delta
```

### The Role of Δ (Delta)

Delta controls the discretization step size:

| Δ Value | Effect |
|---------|--------|
| Small (→0) | Ignore input, preserve state |
| Large | Incorporate input, reset state |

This provides **content-aware gating**:
- Small Δ for context words (preserve state)
- Large Δ for content words (update state)

---

## Mathematical Foundation

### Continuous to Discrete

The continuous system:
$$
\frac{dh}{dt} = Ah(t) + Bx(t)
$$

Discretized using zero-order hold:
$$
\bar{A} = e^{\Delta A}, \quad \bar{B} = (e^{\Delta A} - I)A^{-1}B
$$

### Structured State Matrix

For efficiency, A is structured (diagonal):
$$
A = \text{diag}(a_1, a_2, \ldots, a_N)
$$

This enables:
- O(N) state updates instead of O(N²)
- Parallel computation across state dimensions

### HiPPO Initialization

A is initialized using HiPPO matrices for optimal memory:

```python
def hippo_initialization(state_dim):
    """Initialize A with HiPPO-LegS."""
    A = np.zeros((state_dim, state_dim))
    for i in range(state_dim):
        for j in range(state_dim):
            if i > j:
                A[i, j] = (2*i + 1)**0.5 * (2*j + 1)**0.5
            elif i == j:
                A[i, j] = i + 1
    return -A  # Negative for stability
```

---

## Implementation Details

### The Parallel Scan Algorithm

Sequential scan is O(n) but not parallelizable:
```
h_1 = Ā_1·h_0 + B̄_1·x_1
h_2 = Ā_2·h_1 + B̄_2·x_2
...
```

**Parallel scan** enables GPU parallelism:

```python
def parallel_scan(A_bar, B_bar_x):
    """
    Parallel prefix scan for state space computation.
    
    Args:
        A_bar: [batch, seq_len, state_dim] - discretized A
        B_bar_x: [batch, seq_len, state_dim] - B * x term
    
    Returns:
        h: [batch, seq_len, state_dim] - hidden states
    """
    # Work-efficient parallel scan
    # Complexity: O(n) work, O(log n) depth
    
    n = A_bar.shape[1]
    
    # Up-sweep (reduce phase)
    for d in range(int(np.log2(n))):
        stride = 2 ** (d + 1)
        for i in range(0, n, stride):
            j = i + 2**d - 1
            k = i + stride - 1
            # Combine elements
            A_bar[:, k] = A_bar[:, k] * A_bar[:, j]
            B_bar_x[:, k] = A_bar[:, j] * B_bar_x[:, k] + B_bar_x[:, j]
    
    # Down-sweep (distribute phase)
    for d in range(int(np.log2(n)) - 1, -1, -1):
        stride = 2 ** (d + 1)
        for i in range(0, n, stride):
            j = i + 2**d - 1
            k = i + stride - 1
            # Distribute results
            temp = B_bar_x[:, j].clone()
            B_bar_x[:, j] = B_bar_x[:, k]
            B_bar_x[:, k] = A_bar[:, j] * B_bar_x[:, k] + temp
    
    return B_bar_x  # Final hidden states
```

### Memory-Efficient Implementation

For very long sequences, we use chunked processing:

```python
def chunked_scan(self, x, chunk_size=2048):
    """Process sequence in chunks for memory efficiency."""
    batch, seq_len, dim = x.shape
    
    outputs = []
    state = torch.zeros(batch, self.state_dim, device=x.device)
    
    for i in range(0, seq_len, chunk_size):
        chunk = x[:, i:i+chunk_size]
        chunk_out, state = self.scan_chunk(chunk, state)
        outputs.append(chunk_out)
    
    return torch.cat(outputs, dim=1)
```

---

## Multi-Head State Space

Like multi-head attention, we use multiple parallel state spaces:

```python
class MultiHeadStateSpace(nn.Module):
    def __init__(self, hidden_dim, state_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Each head has its own state space
        self.heads = nn.ModuleList([
            SelectiveStateSpace(self.head_dim, state_dim)
            for _ in range(num_heads)
        ])
        
        # Combine heads
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # Split into heads
        x_heads = x.chunk(self.num_heads, dim=-1)
        
        # Process each head
        outputs = [head(x_h) for head, x_h in zip(self.heads, x_heads)]
        
        # Concatenate and project
        return self.output_proj(torch.cat(outputs, dim=-1))
```

---

## Layer Stacking

### Full Layer Architecture

```python
class StateSpaceLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2):
        super().__init__()
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # State space block
        self.ssm = SelectiveSSMBlock(d_model, d_state, d_conv, expand)
        
        # FFN block
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(0.1)
        )
        
        # Gating
        self.gate = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # State space with residual
        residual = x
        x = self.norm1(x)
        ssm_out = self.ssm(x)
        gate = torch.sigmoid(self.gate(x))
        x = residual + gate * ssm_out
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x
```

### Stack Configuration

```python
class StateSpaceStack(nn.Module):
    def __init__(self, config: StateSpaceConfig):
        super().__init__()
        
        self.layers = nn.ModuleList([
            StateSpaceLayer(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )
            for _ in range(config.n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Input projection | O(L × D × E) |
| Parameter computation | O(L × D × N) |
| Discretization | O(L × N) |
| Selective scan | O(L × N) |
| Output projection | O(L × E × D) |

Where L=length, D=hidden_dim, E=expansion, N=state_dim

**Total per layer**: O(L × D²) = **O(n)**

### Memory Complexity

| Storage | Size |
|---------|------|
| Input | O(L × D) |
| Hidden states | O(L × N) |
| Parameters | O(D × N) |

**Total**: O(L × D) = **O(n)**

### Comparison with Attention

| Metric | Attention | State Space |
|--------|-----------|-------------|
| Time | O(L² × D) | O(L × D²) |
| Memory | O(L² + L × D) | O(L × D) |
| Parallelism | Full | Scan (log L depth) |

---

## Configuration Recommendations

### For Different Sequence Lengths

| Max Length | d_state | n_heads | n_layers |
|------------|---------|---------|----------|
| 1K | 32 | 4 | 6 |
| 8K | 64 | 8 | 12 |
| 32K | 128 | 16 | 18 |
| 128K | 256 | 32 | 24 |

### For Different Compute Budgets

| Budget | d_model | d_state | Params |
|--------|---------|---------|--------|
| Tiny | 256 | 32 | ~10M |
| Small | 512 | 64 | ~50M |
| Medium | 1024 | 128 | ~200M |
| Large | 2048 | 256 | ~800M |

---

## Code Example

```python
from nexus.core.state_space import SelectiveStateSpace, StateSpaceConfig

# Create configuration
config = StateSpaceConfig(
    d_model=512,           # Hidden dimension
    d_state=64,            # State space state dimension
    n_layers=12,           # Number of layers
    d_conv=4,              # Convolution kernel size  
    expand=2,              # Expansion factor
    dt_min=0.001,
    dt_max=0.1,
)

# Initialize model
ssm = SelectiveStateSpace(config)

# Forward pass
x = torch.randn(batch_size, seq_len, config.d_model)
y = ssm(x)  # [batch_size, seq_len, d_model]

# Check linear scaling
import time
for length in [1000, 2000, 4000, 8000]:
    x = torch.randn(1, length, 512).cuda()
    start = time.time()
    _ = ssm(x)
    torch.cuda.synchronize()
    print(f"Length {length}: {time.time()-start:.3f}s")
# Should show roughly linear scaling
```

---

## Further Reading

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [S4 Paper](https://arxiv.org/abs/2111.00396)
- [HiPPO Paper](https://arxiv.org/abs/2008.07669)
- [Architecture Overview](overview.md)
- [World Model](world-model.md)

---

*State spaces: The linear path to long-range understanding.*
