# Architecture Overview

## NEXUS System Architecture

This document provides a comprehensive view of the NEXUS architecture, explaining how all components integrate into a unified system.

NEXUS offers **two architecture modes**:
1. **FlowingNEXUS (Layer-Free)** - Emergent depth, recommended for new development
2. **NEXUSCore (Layered)** - Traditional stacked layers, well-tested baseline

---

## Layer-Free Architecture (FlowingNEXUS) 🆕

The layer-free architecture represents a paradigm shift where **depth emerges from input complexity** rather than being a fixed hyperparameter.

### Key Concept

Traditional neural networks: `input → layer₁ → layer₂ → ... → layerₙ → output`

FlowingNEXUS: `input → f(z*, input) → output` where `z* = f(z*, input)` (fixed point)

### Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLOWING NEXUS (LAYER-FREE)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT ──► UnifiedDynamics f(z, x) ──► ITERATE ──► Equilibrium z* ──► OUT │
│                    ↑                        │                               │
│                    └────────────────────────┘                               │
│                                                                             │
│   UnifiedDynamics contains:                                                 │
│   • Continuous SSM (state space evolution)                                  │
│   • Continuous Attention (global context)                                   │
│   • Co-evolving Memory (persistent state)                                   │
│   • Feed-forward transformation                                             │
│                                                                             │
│   Training uses implicit differentiation: O(1) memory backprop!             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from nexus.core import create_flowing_nexus

model = create_flowing_nexus(size="base")
result = model(x, modality="continuous")

print(f"Flow steps (emergent depth): {result['flow_steps']}")
print(f"Converged: {result['converged']}")
```

---

## Living System Layer

NEXUS operates as a **living system** that evolves continuously through experience.

### Philosophy

> *Growth is not a ladder with rungs to climb.*  
> *It is water finding its level.*  
> *The system doesn't "become" something new -*  
> *it continuously IS, shaped by all it has experienced.*

### Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIVING NEXUS LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │  UncertaintyGate  │  │ LifecycleManager  │  │  ContinualLearner │       │
│  │                   │  │                   │  │                   │       │
│  │ Anti-hallucination│  │ Continuous        │  │ Learn while       │       │
│  │ Refuse when       │  │ evolution         │  │ serving           │       │
│  │ uncertain         │  │ (no stages)       │  │                   │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                             │
│  Key Metrics (all continuous, no discrete stages):                          │
│  ├── experience_factor: 0→1 smooth curve of accumulated wisdom              │
│  ├── confidence_threshold: 0.95→0.35 (cautious when new, knows limits)     │
│  ├── learning_rate_mult: 2.5→0.1 (absorbs fast, then selective)            │
│  └── wisdom_ratio: how often it wisely says "I don't know"                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Production Infrastructure Layer (v2.0)

NEXUS v2.0 includes a comprehensive production infrastructure for enterprise deployment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRODUCTION INFRASTRUCTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Control Interfaces:                                                        │
│  ├── nexusctl (CLI): start/stop/pause/resume/status/logs/dashboard         │
│  ├── Web Dashboard: Real-time monitoring, interaction, controls             │
│  └── REST API: /api/status, /api/interact, /api/control                    │
│                                                                             │
│  Production Components:                                                     │
│  ├── NEXUSTokenizer: HuggingFace transformers with NEXUS special tokens    │
│  ├── CheckpointManager: Atomic saves, SHA256 validation, auto-rotation     │
│  ├── MetricsCollector: Prometheus export, P50/P95/P99, health checks       │
│  ├── CircuitBreaker: 3-state pattern (CLOSED/OPEN/HALF_OPEN)               │
│  ├── MemoryManager: Leak detection, auto-cleanup, GC orchestration         │
│  ├── ResourceGovernor: CPU/RAM limits (Active: 10%, Idle: 25%)             │
│  └── NexusDaemon: Main orchestrator integrating all components             │
│                                                                             │
│  Deployment Modes:                                                          │
│  ├── Development: uvicorn --reload                                          │
│  ├── Production: systemd service (Linux) or nexusctl (Mac/Windows)         │
│  ├── Edge: Raspberry Pi optimized deployment                               │
│  └── Remote: SSH tunnel, Tailscale, ngrok support                          │
│                                                                             │
│  See: docs/architecture/production.md for complete details                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Production Features**:
- **Zero Technical Debt**: All features implemented to completion
- **Real Tokenization**: No mock implementations
- **Checkpoint Persistence**: Crash-safe atomic saves
- **Comprehensive Metrics**: Production-grade observability
- **Error Recovery**: Circuit breaker, retry, graceful degradation
- **Memory Safety**: Long-running stability with leak detection
- **Resource Governance**: Strict CPU/RAM limits
- **Multiple Control Interfaces**: CLI, Dashboard, API

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

### Core Architecture
- [State Space Details](state-space.md)
- [World Model Details](world-model.md)
- [Reasoning Details](reasoning.md)
- [Energy Module Details](energy.md)
- [Causal Engine Details](causal.md)
- [Integration Layer](integration.md)

### Production Infrastructure (v2.0)
- [Production Architecture](production.md) - Complete production infrastructure guide
- [Deployment Guide](../deployment/deployment-guide.md)
- [Operations Runbook](../operations/runbook.md)

---

*Architecture is frozen music. NEXUS orchestrates computation.*
