# Literature Review

## Research Foundations of NEXUS

This document surveys the key research papers and intellectual foundations that inform NEXUS's design. Understanding this literature provides context for our architectural choices.

---

## 1. State Space Models

### 1.1 Foundational Work

#### HiPPO: Recurrent Memory with Optimal Polynomial Projections
**Gu et al., NeurIPS 2020**

- **Key Contribution**: Framework for continuous-time memorization using optimal polynomial basis
- **Core Insight**: Certain matrix structures (HiPPO matrices) enable optimal compression of history
- **Relevance to NEXUS**: Foundation for our state space initialization

```
Key Equation: dh/dt = A(t)h(t) + B(t)u(t)
Where A is the HiPPO matrix providing optimal memory compression
```

#### S4: Efficiently Modeling Long Sequences with Structured State Spaces
**Gu, Goel, Ré, ICLR 2022**

- **Key Contribution**: Made SSMs practical for deep learning via structured parameterization
- **Core Innovation**: DPLR (Diagonal Plus Low-Rank) parameterization enabling efficient computation
- **Relevance to NEXUS**: Direct inspiration for our sequence backbone

**Key Results**:
| Task | Previous SOTA | S4 |
|------|---------------|-----|
| Long Range Arena | 60.5% | 86.1% |
| Raw Speech (SC10) | 93.5% | 98.3% |
| Path-X (16K length) | 50% | 88.0% |

#### Mamba: Linear-Time Sequence Modeling with Selective State Spaces
**Gu & Dao, 2023**

- **Key Contribution**: Input-dependent selection mechanism for SSMs
- **Core Innovation**: Content-aware filtering via learned Δ, B, C
- **Hardware Optimization**: Parallel scan algorithm for GPU efficiency
- **Relevance to NEXUS**: Primary inspiration for SelectiveStateSpace module

**The Selection Mechanism**:
```python
# Mamba's key insight: make parameters input-dependent
B = Linear(x)  # Input projection varies with content
C = Linear(x)  # Output projection varies with content
Δ = softplus(Linear(x))  # Discretization step varies with content
```

### 1.2 Comparison: Attention vs. State Space

| Aspect | Attention | State Space |
|--------|-----------|-------------|
| Complexity | O(n²) | O(n) |
| Memory | O(n²) | O(1) per step |
| Parallelization | Fully parallel | Sequential or parallel scan |
| Long-range | Explicit | Implicit via state |
| Content-aware | Yes (Q-K matching) | Yes (Mamba selection) |

---

## 2. World Models and JEPA

### 2.1 World Models in AI

#### World Models
**Ha & Schmidhuber, NeurIPS 2018**

- **Key Contribution**: Learn compact world model, plan in "dream" space
- **Architecture**: VAE encoder + MDN-RNN dynamics + controller
- **Relevance to NEXUS**: Inspiration for imagination-based planning

#### Dreamer: Dream to Control
**Hafner et al., ICLR 2020**

- **Key Contribution**: Learning behaviors from world model imagination
- **Core Innovation**: Latent imagination for efficient planning
- **Relevance to NEXUS**: Validates representation-space prediction

### 2.2 Joint Embedding Predictive Architectures

#### A Path Towards Autonomous Machine Intelligence
**LeCun, 2022**

- **Key Contribution**: Blueprint for human-level AI through world models
- **JEPA Principle**: Predict in representation space, not observation space
- **Relevance to NEXUS**: Foundational philosophy for world model

**JEPA vs. Generative Models**:
```
Generative: p(x|z) - predict observations from latents
JEPA: d(f(x), g(y)) - compare representations directly

Why JEPA is better:
1. Avoids modeling irrelevant details
2. More sample efficient
3. Robust to observation noise
```

#### I-JEPA: Self-Supervised Learning from Images
**Assran et al., CVPR 2023**

- **Key Contribution**: Practical JEPA implementation for vision
- **Innovation**: Masking in representation space
- **Results**: State-of-the-art with less compute
- **Relevance to NEXUS**: Implementation guidance for our world model

---

## 3. Neuro-Symbolic AI

### 3.1 Neural Theorem Proving

#### End-to-End Differentiable Proving
**Rocktäschel & Riedel, NeurIPS 2017**

- **Key Contribution**: Differentiable backward chaining prover
- **Innovation**: Soft unification with learned embeddings
- **Relevance to NEXUS**: Foundation for reasoning module

**Soft Unification**:
```
Traditional: match(A, B) ∈ {True, False}
Neural: match(A, B) = σ(embed(A)ᵀ embed(B))

Benefits:
- Gradients flow through matching
- Handles near-matches gracefully
- Learns similarity from data
```

#### Neural Logic Machines
**Dong et al., ICLR 2019**

- **Key Contribution**: Differentiable inductive logic programming
- **Architecture**: Tensorized logic operations
- **Relevance to NEXUS**: Inspiration for rule representation

### 3.2 Knowledge Grounding

#### ERNIE: Enhanced Representation through Knowledge Integration
**Sun et al., ACL 2019**

- **Key Contribution**: Integrate knowledge graphs into language models
- **Relevance to NEXUS**: Knowledge grounding strategies

#### RAG: Retrieval-Augmented Generation
**Lewis et al., NeurIPS 2020**

- **Key Contribution**: Ground generation in retrieved documents
- **Relevance to NEXUS**: Retrieval-based grounding techniques

---

## 4. Energy-Based Models

### 4.1 Foundations

#### A Tutorial on Energy-Based Learning
**LeCun et al., 2006**

- **Key Contribution**: Unified framework for discriminative learning
- **Core Idea**: Model compatibility, not probability
- **Relevance to NEXUS**: Theoretical foundation for energy module

**Energy vs. Probability**:
```
Probabilistic: P(y|x) = exp(-E(x,y)) / Z(x)
Energy-based: Just use E(x,y) directly

Advantages of energy:
- No normalization constant Z
- Can model unnormalized densities
- Natural for optimization-based inference
```

### 4.2 Modern EBMs

#### Your Classifier is Secretly an Energy-Based Model
**Grathwohl et al., ICLR 2020**

- **Key Contribution**: Reinterpret classifiers as EBMs
- **Innovation**: Joint training of discriminative and generative objectives
- **Relevance to NEXUS**: Multi-task energy formulation

#### Energy-Based Models for Continual Learning
**Li et al., NeurIPS 2020**

- **Key Contribution**: EBMs for detecting distribution shift
- **Relevance to NEXUS**: Uncertainty quantification via energy

### 4.3 Adaptive Computation

#### Adaptive Computation Time for Recurrent Neural Networks
**Graves, 2016**

- **Key Contribution**: Input-dependent computation depth
- **Mechanism**: Halting probability at each step
- **Relevance to NEXUS**: Inspiration for adaptive energy module

#### Universal Transformers
**Dehghani et al., ICLR 2019**

- **Key Contribution**: Recurrent Transformer with adaptive depth
- **Relevance to NEXUS**: Integration with modern architectures

---

## 5. Causal Inference

### 5.1 Foundational Theory

#### Causality: Models, Reasoning, and Inference
**Pearl, 2009**

- **Key Contribution**: Mathematical framework for causation
- **The Three Levels**: Association, Intervention, Counterfactual
- **Relevance to NEXUS**: Theoretical foundation for causal engine

**Pearl's Causal Hierarchy**:
```
Level 1 - Association: P(y|x) - "What if I observe X=x?"
Level 2 - Intervention: P(y|do(X=x)) - "What if I set X=x?"
Level 3 - Counterfactual: P(y_x|x', y') - "What if X had been x?"

Each level requires more causal knowledge.
Most ML operates at Level 1.
NEXUS operates at all three.
```

#### Elements of Causal Inference
**Peters, Janzing, Schölkopf, 2017**

- **Key Contribution**: Algorithmic causal inference
- **Relevance to NEXUS**: Practical algorithms for causal discovery

### 5.2 Causal Discovery

#### DAGs with NO TEARS
**Zheng et al., NeurIPS 2018**

- **Key Contribution**: Continuous optimization for causal discovery
- **Innovation**: Differentiable acyclicity constraint
- **Relevance to NEXUS**: Learnable causal structure

**The Acyclicity Constraint**:
```
h(A) = tr(e^{A ⊙ A}) - d = 0

Where A is adjacency matrix.
h(A) = 0 iff A is acyclic.
This makes causal discovery a continuous optimization problem.
```

#### Causal Discovery with Reinforcement Learning
**Zhu et al., ICLR 2020**

- **Key Contribution**: RL for causal structure search
- **Relevance to NEXUS**: Alternative discovery approaches

### 5.3 Causal Representation Learning

#### Towards Causal Representation Learning
**Schölkopf et al., 2021**

- **Key Contribution**: Unified view of causality and representation learning
- **Core Thesis**: Good representations should be causal
- **Relevance to NEXUS**: Philosophical foundation for causal integration

---

## 6. Synthesis: NEXUS Innovation

### 6.1 What's New in NEXUS

| Existing Work | NEXUS Innovation |
|---------------|------------------|
| Mamba (SSM) | + Multi-head selective state space |
| JEPA (world model) | + Hierarchical temporal abstraction |
| Neural provers | + Soft unification with attention |
| EBMs | + Adaptive depth in sequence models |
| Causal discovery | + End-to-end differentiable integration |

### 6.2 Novel Contributions

1. **Unified Architecture**: First to combine all five paradigms coherently
2. **O(n) Reasoning**: Symbolic reasoning with linear-time backbone
3. **Causal Generation**: Generation grounded in causal models
4. **Energy-Guided Refinement**: Adaptive computation for language
5. **Multi-Objective Training**: Joint optimization of diverse objectives

---

## 7. Reading List by Topic

### Essential Reading
1. Mamba paper (Gu & Dao, 2023)
2. JEPA manuscript (LeCun, 2022)
3. Causality book (Pearl, 2009)

### Deep Dives
- **State Space**: S4 → S4D → H3 → Mamba
- **World Models**: Ha & Schmidhuber → Dreamer → JEPA
- **Neuro-Symbolic**: NTP → Neural Logic Machines → NSIL
- **EBMs**: LeCun Tutorial → Modern EBMs → Adaptive Computation
- **Causal**: Pearl → NOTEARS → Causal Representation Learning

### Implementation References
- Mamba official implementation (GitHub)
- I-JEPA official implementation (Meta Research)
- Neural theorem prover implementations

---

## Citation Index

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{lecun2022path,
  title={A Path Towards Autonomous Machine Intelligence},
  author={LeCun, Yann},
  journal={OpenReview},
  year={2022}
}

@book{pearl2009causality,
  title={Causality: Models, Reasoning, and Inference},
  author={Pearl, Judea},
  year={2009},
  publisher={Cambridge University Press}
}

@inproceedings{rocktaschel2017end,
  title={End-to-End Differentiable Proving},
  author={Rockt{\"a}schel, Tim and Riedel, Sebastian},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{zheng2018dags,
  title={DAGs with NO TEARS},
  author={Zheng, Xun and others},
  booktitle={NeurIPS},
  year={2018}
}
```

---

*"If I have seen further, it is by standing on the shoulders of giants." — Newton*

*NEXUS stands on many shoulders.*
