# Architecture Overview

## NEXUS System Architecture

This document provides a comprehensive view of the NEXUS architecture, explaining how all components integrate into a unified system.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NEXUS CORE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         INPUT PROCESSING                               │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │ │
│  │  │   Token     │    │  Position   │    │   Modal     │               │ │
│  │  │  Embedding  │ +  │  Encoding   │ +  │  Encoding   │ = Input Embed │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    SELECTIVE STATE SPACE BACKBONE                      │ │
│  │                                                                        │ │
│  │    ┌─────┐    ┌─────┐    ┌─────┐           ┌─────┐                   │ │
│  │    │ SSS │ -> │ SSS │ -> │ SSS │ -> ... -> │ SSS │                   │ │
│  │    │  1  │    │  2  │    │  3  │           │  L  │                   │ │
│  │    └─────┘    └─────┘    └─────┘           └─────┘                   │ │
│  │                     O(n) Linear Complexity                            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                       │
│                    │               │               │                       │
│                    ▼               ▼               ▼                       │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐           │
│  │   WORLD MODEL    │ │    REASONER      │ │  CAUSAL ENGINE   │           │
│  │                  │ │                  │ │                  │           │
│  │ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │           │
│  │ │   Context    │ │ │ │    Rule      │ │ │ │     SCM      │ │           │
│  │ │   Encoder    │ │ │ │    Base      │ │ │ │   Learner    │ │           │
│  │ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │           │
│  │ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │           │
│  │ │   Target     │ │ │ │    Soft      │ │ │ │   Causal     │ │           │
│  │ │   Encoder    │ │ │ │ Unification  │ │ │ │  Attention   │ │           │
│  │ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │           │
│  │ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌──────────────┐ │           │
│  │ │  Predictor   │ │ │ │   Proof      │ │ │ │ Counterfact  │ │           │
│  │ │              │ │ │ │   Tracer     │ │ │ │   Reasoner   │ │           │
│  │ └──────────────┘ │ │ └──────────────┘ │ │ └──────────────┘ │           │
│  │                  │ │                  │ │                  │           │
│  │  JEPA-Style     │ │  Neuro-Symbolic  │ │    Causal       │           │
│  │  Prediction     │ │    Reasoning     │ │   Inference     │           │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘           │
│                    │               │               │                       │
│                    └───────────────┼───────────────┘                       │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       ENERGY-BASED REFINEMENT                          │ │
│  │                                                                        │ │
│  │    Input ──► Energy Function ──► Gradient ──► Refined Output          │ │
│  │              E(x, y)           ∇E           (iterate until converge)  │ │
│  │                                                                        │ │
│  │    Adaptive computation: more iterations for harder inputs            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         OUTPUT GENERATION                              │ │
│  │                                                                        │ │
│  │    Refined Repr ──► Output Projection ──► Softmax ──► Logits          │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interactions

### Information Flow Diagram

```
                              ┌─────────────────┐
                              │     Input       │
                              │   (tokens)      │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   Embedding     │
                              │     Layer       │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  State Space    │
                              │    Backbone     │◄──── O(n) processing
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
             ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
             │   World     │    │  Reasoner   │    │   Causal    │
             │   Model     │    │             │    │   Engine    │
             └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                    │                  │                  │
                    │    Predictions   │   Proofs        │  Causal
                    │                  │                  │  Structure
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    Fusion       │
                              │    Module       │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    Energy       │◄──── Adaptive depth
                              │   Refinement    │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    Output       │
                              │   Projection    │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    Logits       │
                              │   (vocab_size)  │
                              └─────────────────┘
```

---

## Detailed Component Specifications

### 1. Input Processing

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| Token Embedding | `[B, L]` indices | `[B, L, D]` | Map tokens to vectors |
| Position Encoding | `[L]` positions | `[L, D]` | Add position information |
| Modal Encoding | Modal type | `[D]` | Distinguish modalities |

**Configuration**:
```python
embedding_config = {
    'vocab_size': 32000,
    'd_model': 512,        # Hidden dimension
    'max_seq_len': 8192,   # Maximum sequence length
    'dropout': 0.1,
}
```

### 2. State Space Backbone

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| d_model | 512-2048 | Model width |
| ssm_d_state | 64-256 | State space dimension |
| n_heads | 8-32 | Attention heads (world model) |
| ssm_n_layers | 6-24 | Depth |

**Layer Structure**:
```
┌──────────────────────────────────┐
│     State Space Layer            │
├──────────────────────────────────┤
│ Input Norm ──► Selective SSM     │
│              ──► Output Project  │
│              ──► Residual Add    │
│              ──► FFN             │
│              ──► Residual Add    │
└──────────────────────────────────┘
```

### 3. Auxiliary Modules

**World Model**:
```
Context Encoder: Transformer layers (4-8)
Target Encoder: EMA copy of context encoder
Predictor: MLP (2-4 layers)
Temporal Abstraction: Pooling at multiple scales
```

**Reasoner**:
```
Rule Base: Learnable embeddings (50-500 rules)
Unification: Attention-based soft matching
Proof Tracer: Stack-based derivation recording
Knowledge Graph: Optional external grounding
```

**Causal Engine**:
```
SCM Learner: Differentiable DAG learning
Causal Attention: Masked attention following DAG
Counterfactual: Abduction-action-prediction pipeline
```

### 4. Energy Module

```
Energy Function: MLP mapping (x, y) → scalar
Refinement: Gradient descent on y
Convergence: Energy threshold or iteration limit
Output: Refined representation + energy history
```

---

## Memory and Compute Profiles

### Memory Usage (Approximate)

| Component | Memory | Notes |
|-----------|--------|-------|
| Embeddings | O(V × D) | V=vocab, D=dim |
| State Space | O(L × D) | L=length, per layer |
| World Model | O(L × D) | Encoder representations |
| Reasoner | O(R × D) | R=rules |
| Causal | O(V² + L × D) | V=variables |
| Energy | O(L × D) | Refinement states |

**Total**: O(L × D × Layers) ≈ **Linear in sequence length**

### Compute Profile (FLOPs)

| Component | FLOPs | Complexity |
|-----------|-------|------------|
| State Space | 6 × L × D² | O(n) |
| World Model | 4 × L × D² | O(n) |
| Reasoner | R × L × D | O(n) |
| Causal | V² × D + L × D² | O(n + V²) |
| Energy | I × L × D² | O(n × I) |

Where I = refinement iterations (typically 1-10)

---

## Configuration Hierarchy

```yaml
nexus_config:
  # Core dimensions (NEXUSConfig dataclass parameters)
  vocab_size: 32000
  d_model: 512            # Hidden dimension
  d_latent: 256           # Latent dimension for world model
  ssm_n_layers: 12        # Number of state space layers
  n_heads: 8              # Attention heads
  ssm_d_state: 64         # State space state dimension
  ssm_d_conv: 4           # Convolution kernel size
  ssm_expand: 2           # Expansion factor
  
  # Reasoning
  n_predicates: 64        # Number of reasoning predicates
  n_constants: 128        # Number of reasoning constants
  max_reasoning_steps: 5  # Maximum reasoning iterations
  
  # Causal
  n_variables: 32         # Number of causal variables
  
  # Energy
  max_energy_iters: 10    # Maximum energy iterations
  
  # Sequence
  max_seq_len: 8192       # Maximum sequence length
  dropout: 0.1            # Dropout rate
```

---

## Scaling Properties

### Model Size Configurations

| Config | ssm_n_layers | d_model | n_heads | Params |
|--------|--------------|---------|---------|--------|
| Tiny | 4 | 256 | 4 | ~10M |
| Small | 6 | 512 | 8 | ~50M |
| Medium | 12 | 1024 | 16 | ~200M |
| Large | 24 | 2048 | 32 | ~800M |
| XL | 32 | 4096 | 64 | ~3B |

### Scaling Laws

Based on empirical observations:

**Compute-Optimal Training**:
```
Optimal tokens ≈ 20 × Parameters
(Similar to Chinchilla scaling)
```

**Loss Scaling**:
```
L(N, D) = A/N^α + B/D^β + C
Where N = params, D = data
α ≈ 0.5, β ≈ 0.5
```

---

## Deployment Modes

### 1. Full NEXUS (All Components)
- Maximum capability
- Highest compute
- Use for: Research, complex reasoning

### 2. Fast NEXUS (State Space + Energy)
- High efficiency
- Skip world model and reasoner
- Use for: Production inference

### 3. Reasoning NEXUS (State Space + Reasoner)
- Focused on explainability
- Include proof traces
- Use for: Verified reasoning tasks

### 4. Causal NEXUS (State Space + Causal)
- Focused on interventions
- Include causal discovery
- Use for: Decision-making, planning

---

## Extension Points

NEXUS is designed for extensibility:

```python
class NEXUSCore:
    def register_module(self, name: str, module: nn.Module):
        """Add custom auxiliary module."""
        
    def register_loss(self, name: str, loss_fn: Callable):
        """Add custom loss term."""
        
    def register_callback(self, event: str, callback: Callable):
        """Add training callbacks."""
```

**Example Extensions**:
- Retrieval-Augmented Generation
- Multi-modal encoders
- Custom reasoning engines
- Domain-specific losses

---

## Further Reading

- [State Space Details](state-space.md)
- [World Model Details](world-model.md)
- [Reasoning Details](reasoning.md)
- [Energy Module Details](energy.md)
- [Causal Engine Details](causal.md)
- [Integration Layer](integration.md)

---

*Architecture is frozen music. NEXUS orchestrates computation.*
