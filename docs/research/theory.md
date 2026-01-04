# Theoretical Foundations

## Mathematical Framework for NEXUS

This document presents the rigorous mathematical foundations underlying each NEXUS component, establishing the theoretical basis for our architectural choices.

---

## 1. State Space Models: From Control Theory to Sequence Modeling

### 1.1 Continuous-Time State Space Models

The foundation of NEXUS's sequence processing comes from linear dynamical systems theory:

**Continuous-Time Definition**:
$$
\begin{aligned}
\frac{dx(t)}{dt} &= \mathbf{A}x(t) + \mathbf{B}u(t) \\
y(t) &= \mathbf{C}x(t) + \mathbf{D}u(t)
\end{aligned}
$$

Where:
- $x(t) \in \mathbb{R}^N$ is the hidden state
- $u(t) \in \mathbb{R}^1$ is the input
- $y(t) \in \mathbb{R}^1$ is the output
- $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix
- $\mathbf{B} \in \mathbb{R}^{N \times 1}$ is the input projection
- $\mathbf{C} \in \mathbb{R}^{1 \times N}$ is the output projection

### 1.2 Discretization

For sequence modeling, we discretize using the Zero-Order Hold (ZOH) method:

$$
\begin{aligned}
\bar{\mathbf{A}} &= \exp(\Delta \mathbf{A}) \\
\bar{\mathbf{B}} &= (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}
\end{aligned}
$$

For numerical stability, we use the approximation:
$$
\begin{aligned}
\bar{\mathbf{A}} &= \left(\mathbf{I} - \frac{\Delta}{2}\mathbf{A}\right)^{-1}\left(\mathbf{I} + \frac{\Delta}{2}\mathbf{A}\right) \\
\bar{\mathbf{B}} &= \left(\mathbf{I} - \frac{\Delta}{2}\mathbf{A}\right)^{-1} \Delta \mathbf{B}
\end{aligned}
$$

### 1.3 Recurrent Computation (O(n) Complexity)

The discretized system enables efficient sequential computation:

$$
\begin{aligned}
x_k &= \bar{\mathbf{A}} x_{k-1} + \bar{\mathbf{B}} u_k \\
y_k &= \mathbf{C} x_k
\end{aligned}
$$

**Complexity Analysis**:
- Each step: O(N²) for state update, O(N) for output
- Total for sequence of length L: O(L · N²)
- With structured A (diagonal): O(L · N) = **O(n)**

### 1.4 Selective State Space (Mamba Innovation)

Traditional SSMs use fixed parameters. NEXUS employs **input-dependent** selection:

$$
\begin{aligned}
\mathbf{B}_t &= \text{Linear}_B(x_t) \\
\mathbf{C}_t &= \text{Linear}_C(x_t) \\
\Delta_t &= \text{softplus}(\text{Linear}_\Delta(x_t))
\end{aligned}
$$

This allows the model to selectively propagate or forget information based on content.

### 1.5 The Selectivity Mechanism

The key insight is that $\Delta_t$ controls the balance between:
- **Small Δ**: Ignore current input, preserve state (forget gate ≈ 1)
- **Large Δ**: Incorporate current input, reset state (forget gate ≈ 0)

This provides content-aware filtering crucial for language modeling.

---

## 2. JEPA: Joint Embedding Predictive Architecture

### 2.1 Core Principle

Instead of predicting raw observations (pixels, tokens), JEPA predicts in **abstract representation space**:

$$
\mathcal{L}_{\text{JEPA}} = \|\text{Predictor}(\text{Encode}_\theta(x)) - \text{sg}[\text{Encode}_\xi(y)]\|^2
$$

Where:
- $x$ is context
- $y$ is target
- $\text{sg}[\cdot]$ is stop-gradient (prevents collapse)
- $\theta$ is online encoder parameters
- $\xi$ is target encoder parameters (EMA of $\theta$)

### 2.2 Why Representation-Space Prediction?

**Theorem (Representation Compression)**: Predicting in representation space is more sample-efficient than predicting raw observations when:
$$
\dim(\text{representation}) \ll \dim(\text{observation})
$$

**Intuition**: A 256-dim embedding captures the "essence" of an image better than predicting 786,432 pixels.

### 2.3 Exponential Moving Average Target

To prevent representation collapse, the target encoder uses EMA:

$$
\xi \leftarrow \tau \xi + (1 - \tau) \theta
$$

Where $\tau \in [0.99, 0.999]$ provides a slowly-moving target.

### 2.4 Hierarchical Abstraction

NEXUS extends JEPA with multi-scale temporal abstraction:

$$
z_t^{(l)} = \text{Abstract}^{(l)}(z_t^{(l-1)}, z_{t-1}^{(l)})
$$

Where layer $l$ operates at progressively coarser timescales, enabling:
- Short-term: Word-level patterns
- Medium-term: Sentence-level structure
- Long-term: Document-level themes

---

## 3. Neuro-Symbolic Reasoning

### 3.1 The Integration Challenge

Neural networks excel at pattern recognition; symbolic systems excel at logical reasoning. NEXUS bridges both through **differentiable symbolic operations**.

### 3.2 Soft Unification

Classical unification asks: "Do these terms match exactly?"
Soft unification asks: "How similar are these terms?"

$$
\text{Unify}_\text{soft}(t_1, t_2) = \sigma\left(\frac{\langle f(t_1), f(t_2) \rangle}{\tau}\right)
$$

Where:
- $f(\cdot)$ embeds terms into vector space
- $\langle \cdot, \cdot \rangle$ is dot product
- $\tau$ is temperature
- $\sigma$ is sigmoid

### 3.3 Neural Rule Base

Rules are represented as neural embeddings:

$$
\text{Rule}_i = (\mathbf{h}_i, \mathbf{b}_i, \mathbf{c}_i)
$$

Where:
- $\mathbf{h}_i$ is the head (conclusion) embedding
- $\mathbf{b}_i$ is the body (premises) embedding
- $\mathbf{c}_i$ is the confidence embedding

### 3.4 Differentiable Forward Chaining

Given facts $F$ and rules $R$, we compute derived facts:

$$
F' = F \cup \{r.\text{head} : r \in R, \text{Unify}_\text{soft}(r.\text{body}, F) > \theta\}
$$

This is made differentiable through soft attention over rules:

$$
\alpha_i = \text{softmax}_i(\text{Unify}_\text{soft}(r_i.\text{body}, F))
$$

### 3.5 Proof Trace Generation

Each inference step is logged:

$$
\text{Proof} = [(r_1, p_1, c_1), (r_2, p_2, c_2), \ldots]
$$

Where $(r_i, p_i, c_i)$ indicates rule $r_i$ applied to premises $p_i$ yielding conclusion $c_i$.

---

## 4. Energy-Based Models

### 4.1 Energy Function Formulation

An energy function assigns scalar "energy" to configurations:

$$
E_\theta(x, y) : \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}
$$

Lower energy = more compatible configuration.

### 4.2 Inference as Energy Minimization

Given input $x$, find output:

$$
\hat{y} = \arg\min_y E_\theta(x, y)
$$

### 4.3 Iterative Refinement (Langevin Dynamics)

NEXUS uses gradient-based refinement:

$$
y_{t+1} = y_t - \eta \nabla_y E_\theta(x, y_t) + \sqrt{2\eta} \epsilon_t
$$

Where:
- $\eta$ is step size
- $\epsilon_t \sim \mathcal{N}(0, \mathbf{I})$ is noise (optional, for exploration)

### 4.4 Adaptive Computation Depth

The number of refinement steps adapts to input complexity:

$$
T^*(x) = \min\{t : E_\theta(x, y_t) < \epsilon \text{ or } \|\nabla_y E_\theta\| < \delta\}
$$

This provides:
- **Efficiency**: Simple inputs exit early
- **Capability**: Hard inputs get more computation

### 4.5 Contrastive Energy Learning

Training pushes down energy of correct pairs, up for incorrect:

$$
\mathcal{L}_{\text{contrastive}} = \mathbb{E}_{(x,y^+)}[E_\theta(x, y^+)] - \mathbb{E}_{(x,y^-)}[E_\theta(x, y^-)] + \text{margin}
$$

---

## 5. Causal Inference

### 5.1 Structural Causal Models (SCMs)

An SCM consists of:
- Variables $V = \{V_1, \ldots, V_n\}$
- Structural equations $V_i := f_i(\text{Pa}_i, U_i)$
- Noise variables $U = \{U_1, \ldots, U_n\}$

### 5.2 The Three Levels of Causation (Pearl's Ladder)

1. **Association** (seeing): $P(Y|X)$
2. **Intervention** (doing): $P(Y|do(X))$
3. **Counterfactual** (imagining): $P(Y_{X=x'}|X=x, Y=y)$

Transformers operate at Level 1. NEXUS operates at all three levels.

### 5.3 The do-Calculus

Intervention differs from conditioning:

$$
P(Y|do(X=x)) \neq P(Y|X=x)
$$

The intervention $do(X=x)$:
1. Sets $X$ to $x$
2. Removes all arrows into $X$ (breaks confounding)

### 5.4 Causal Discovery

NEXUS learns causal structure from observational data using:

**Score-based approach**:
$$
\hat{G} = \arg\max_G \text{Score}(G; D) - \lambda \|G\|_0
$$

Where the score measures how well graph $G$ explains data $D$.

**Constraint-based approach**:
Using conditional independence tests:
$$
X \perp\!\!\!\perp Y | Z \Rightarrow \text{no edge } X \rightarrow Y \text{ unless through } Z
$$

### 5.5 Counterfactual Computation

To compute "What would Y have been if X had been x'?":

1. **Abduction**: Infer noise $U$ from observations
2. **Action**: Intervene $do(X=x')$  
3. **Prediction**: Compute $Y$ under new $X$ with same $U$

$$
Y_{X=x'} = f_Y(\text{Pa}_Y|_{X=x'}, U_Y)
$$

---

## 6. Integration: The NEXUS Synthesis

### 6.1 Information Flow

```
Input → Embedding → [State-Space Backbone] → Hidden States
                            ↓
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
        [World Model] [Reasoner]    [Causal Engine]
              ↓             ↓             ↓
              └─────────────┴─────────────┘
                            ↓
                    [Energy Module]
                    (Iterative Refinement)
                            ↓
                        Output
```

### 6.2 Loss Function

Total loss combines all components:

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{LM}}}_{\text{language}} + \lambda_1 \underbrace{\mathcal{L}_{\text{JEPA}}}_{\text{world model}} + \lambda_2 \underbrace{\mathcal{L}_{\text{reason}}}_{\text{reasoning}} + \lambda_3 \underbrace{\mathcal{L}_{\text{energy}}}_{\text{efficiency}} + \lambda_4 \underbrace{\mathcal{L}_{\text{causal}}}_{\text{causality}}
$$

### 6.3 Theoretical Guarantees

**Theorem (Linear Complexity)**: NEXUS forward pass is O(n) in sequence length.

*Proof*: The state-space backbone dominates computation. With diagonal state matrix:
- State update: O(N) per step
- Total: O(L × N) = O(n) where n = L × N □

**Theorem (Reasoning Soundness)**: If the neural rule base approximates a sound logical system, then high-confidence conclusions are approximately sound.

**Theorem (Causal Identifiability)**: Under standard causal assumptions (faithfulness, causal sufficiency), the learned causal graph is identifiable up to Markov equivalence.

---

## 7. Connections to Cognitive Science

### 7.1 Dual Process Theory

NEXUS mirrors human cognition's dual-process structure:
- **System 1** (fast, automatic): State-space backbone
- **System 2** (slow, deliberate): Energy-based refinement

### 7.2 Mental Simulation

The world model enables "mental simulation"—imagining consequences before acting—a core human cognitive capability.

### 7.3 Causal Reasoning

Human intelligence is fundamentally causal. NEXUS's causal engine provides similar capabilities:
- "What caused this?"
- "What if I do X?"
- "Why did Y happen?"

---

## Further Reading

- [S4: Efficiently Modeling Long Sequences](https://arxiv.org/abs/2111.00396)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [JEPA: A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [Neural Theorem Provers](https://arxiv.org/abs/1705.11040)
- [Elements of Causal Inference](https://mitpress.mit.edu/books/elements-causal-inference)

---

*"Mathematics is the language in which God has written the universe." — Galileo*

*NEXUS speaks this language fluently.*
