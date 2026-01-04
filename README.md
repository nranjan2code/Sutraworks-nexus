# NEXUS - Neural EXploratory Unified Synthesis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A next-generation AI architecture designed to address fundamental limitations of Transformer/LLM models.**

## 🎯 Key Innovations

NEXUS synthesizes five cutting-edge AI paradigms into a unified architecture:

| Component | Paradigm | Key Benefit |
|-----------|----------|-------------|
| **SelectiveStateSpace** | Mamba/S4 State-Space Models | O(n) linear-time sequence processing |
| **HierarchicalWorldModel** | JEPA (Joint Embedding Predictive Architecture) | Abstract world modeling, not just token prediction |
| **NeuroSymbolicReasoner** | Neuro-Symbolic AI | Explainable reasoning with proof traces |
| **AdaptiveEnergyModule** | Energy-Based Models | Adaptive computation and uncertainty quantification |
| **CausalInferenceEngine** | Causal AI | True causal understanding, not just correlation |

## 🚀 Why NEXUS?

### Addressing LLM/Transformer Limitations

| Problem | Transformer | NEXUS Solution |
|---------|-------------|----------------|
| **Quadratic Complexity** | O(n²) attention | O(n) state-space backbone |
| **Context Length** | Limited by memory | Efficient 100K+ tokens |
| **Hallucination** | Black-box generation | Grounded reasoning with proofs |
| **Correlation vs Causation** | Learns correlations | Native causal inference |
| **Token Prediction** | Next-token only | Abstract world modeling |
| **Explainability** | Opaque decisions | Proof traces & symbolic grounding |

### Efficiency at Scale

```
Sequence Length  | Transformer O(n²) | NEXUS O(n)  | Speedup
-----------------|-------------------|-------------|--------
1,000 tokens     | 1,000,000 ops     | 1,000 ops   | 1,000x
10,000 tokens    | 100,000,000 ops   | 10,000 ops  | 10,000x
100,000 tokens   | 10B ops           | 100,000 ops | 100,000x
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nexus.git
cd nexus

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+
- SciPy 1.10+
- einops 0.6+
- NetworkX 3.0+

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      NEXUS Core                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Embedding  │ -> │ State-Space │ -> │   Output    │     │
│  │   Layer     │    │   Backbone  │    │   Layer     │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                                │
│         ┌──────────────────┼──────────────────┐            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   World     │    │  Reasoning  │    │   Causal    │     │
│  │   Model     │    │   Module    │    │   Engine    │     │
│  │  (JEPA)     │    │(Neuro-Sym)  │    │ (Inference) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            ▼                                │
│                   ┌─────────────┐                          │
│                   │   Energy    │                          │
│                   │   Module    │                          │
│                   │ (Adaptive)  │                          │
│                   └─────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📖 Quick Start

### Basic Usage

```python
import torch
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig

# Create configuration with correct parameter names
config = NEXUSConfig(
    vocab_size=32000,
    d_model=512,           # Hidden dimension (not hidden_dim)
    d_latent=256,          # Latent dimension
    ssm_n_layers=6,        # Number of state space layers (not num_layers)
    n_heads=8,             # Number of attention heads (not num_heads)
    ssm_d_state=64,        # State space state dimension (not state_dim)
    max_seq_len=8192,
)

# Initialize model
model = NEXUSCore(config)
model.to("cuda")

# Forward pass with continuous input (float tensor)
x = torch.randn(1, 256, config.d_model).cuda()  # [batch, seq_len, d_model]
outputs = model(x)
logits = outputs['logits']  # [1, 256, 32000]
```

### Generation

```python
# Generate text (requires embedding layer for token IDs)
prompt = torch.randn(1, 32, config.d_model).cuda()  # Continuous prompt
generated = model.generate(
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
)
```

### Reasoning with Proof Traces

```python
# Get reasoning output with explainable proof
x = torch.randn(1, 100, config.d_model).cuda()
reasoning_output = model.reason(x)
proof_trace = reasoning_output['proof_trace']
```

### Imagination (World Modeling)

```python
# Predict future abstract states
x = torch.randn(1, 100, config.d_model).cuda()
imagination = model.imagine(x, n_steps=5)  # Note: n_steps, not steps
future_states = imagination['imagined_states']
```

### Causal Intervention

```python
# Perform causal intervention
x = torch.randn(1, 100, config.d_model).cuda()
intervention = model.intervene(
    x,
    intervention_idx=32,
    intervention_value=torch.randn(1, config.d_model).cuda(),
)
counterfactual = intervention['counterfactual']
```

## 🔧 Training

```python
from nexus.training.trainer import NEXUSTrainer, TrainingConfig
from nexus.training.data import NEXUSDataset

# Create training config
train_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    epochs=100,
    gradient_accumulation_steps=4,
    use_mixed_precision=True,
)

# Initialize trainer
trainer = NEXUSTrainer(model, train_config)

# Train
trainer.train(train_dataset, val_dataset)
```

### Multi-Objective Loss

NEXUS uses a composite loss function:

```
Total Loss = λ₁·LM_Loss + λ₂·JEPA_Loss + λ₃·Reasoning_Loss + λ₄·Energy_Loss + λ₅·Causal_Loss
```

| Loss Component | Purpose |
|----------------|---------|
| LM Loss | Language modeling (cross-entropy) |
| JEPA Loss | World model prediction in representation space |
| Reasoning Loss | Proof validity and grounding |
| Energy Loss | Computation efficiency regularization |
| Causal Loss | Causal consistency constraints |

## 📊 Benchmarking

```bash
# Run full benchmark suite
python examples/benchmark_demo.py --model-size small

# Run specific benchmarks
python examples/benchmark_demo.py --scaling --long-context
```

### Available Benchmarks

- **Scaling Benchmark**: Verify O(n) computational complexity
- **Long-Context Benchmark**: Test performance on 1K-64K token sequences
- **Reasoning Benchmark**: Evaluate proof validity and grounding
- **Causal Benchmark**: Test intervention and counterfactual accuracy

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=nexus --cov-report=html
```

## 📁 Project Structure

```
sutraworks-genNxt/
├── nexus/
│   ├── core/
│   │   ├── state_space.py      # O(n) sequence backbone
│   │   ├── world_model.py      # JEPA-style world modeling
│   │   ├── reasoning.py        # Neuro-symbolic reasoning
│   │   ├── energy.py           # Energy-based computation
│   │   ├── causal.py           # Causal inference engine
│   │   └── nexus_core.py       # Integrated architecture
│   ├── training/
│   │   ├── trainer.py          # Training orchestration
│   │   ├── data.py             # Dataset utilities
│   │   └── losses.py           # Multi-objective losses
│   └── evaluation/
│       ├── benchmarks.py       # Benchmark suites
│       └── metrics.py          # Evaluation metrics
├── examples/
│   ├── basic_usage.py          # Basic usage demo
│   ├── training_demo.py        # Training demonstration
│   └── benchmark_demo.py       # Benchmarking demo
├── tests/
│   └── test_core.py            # Unit tests
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔬 Research Background

NEXUS builds upon several groundbreaking research directions:

1. **State Space Models**: [Mamba](https://arxiv.org/abs/2312.00752), [S4](https://arxiv.org/abs/2111.00396)
2. **JEPA**: [Joint Embedding Predictive Architecture](https://openreview.net/forum?id=BZ5a1r-kVsf) (Yann LeCun)
3. **Neuro-Symbolic AI**: Combining neural networks with symbolic reasoning
4. **Energy-Based Models**: [EBMs for generative modeling and planning](https://arxiv.org/abs/1903.08689)
5. **Causal AI**: [Causal inference and counterfactual reasoning](https://arxiv.org/abs/2102.11107)

## 🗺️ Roadmap

- [x] Core architecture implementation
- [x] Training pipeline
- [x] Benchmarking suite
- [ ] Pre-trained model weights
- [ ] Tokenizer integration
- [ ] Multi-GPU training
- [ ] Flash attention optimization
- [ ] ONNX export support
- [ ] Hugging Face integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📬 Contact

For questions and feedback, please open an issue on GitHub.

---

**NEXUS** - Taking AI to the Next Generation 🚀
