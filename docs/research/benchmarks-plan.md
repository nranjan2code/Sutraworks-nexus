# NEXUS Benchmark & Evaluation Plan

## Overview

This document defines the comprehensive evaluation strategy for NEXUS, covering benchmark selection, evaluation metrics, experimental design, and validation protocols.

---

## Evaluation Philosophy

### Principles

1. **Capability-Focused**: Test actual capabilities, not benchmark gaming
2. **Fair Comparison**: Use standardized settings across architectures
3. **Multi-Dimensional**: Evaluate efficiency, accuracy, AND explainability
4. **Reproducible**: All experiments must be reproducible

### Evaluation Dimensions

| Dimension | What We Measure | Why It Matters |
|-----------|-----------------|----------------|
| **Efficiency** | FLOPs, latency, memory | Practical deployment |
| **Quality** | Accuracy, perplexity | Core capability |
| **Reasoning** | Proof validity, logical consistency | Explainability |
| **Causality** | Intervention accuracy, counterfactuals | True understanding |
| **Robustness** | Out-of-distribution, adversarial | Real-world reliability |
| **Scaling** | Performance vs. compute/data | Future potential |

---

## Benchmark Categories

### Category 1: Computational Efficiency

#### 1.1 Scaling Benchmark

**Objective**: Verify O(n) computational complexity claim

**Protocol**:
```python
sequence_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
for seq_len in sequence_lengths:
    measure:
        - Forward pass time
        - Backward pass time
        - Peak memory usage
        - FLOPs count
```

**Expected Results**:
| Metric | Transformer Scaling | NEXUS Scaling |
|--------|--------------------:|---------------:|
| Time | O(n²) | O(n) |
| Memory | O(n²) | O(n) |
| FLOPs | O(n²) | O(n) |

**Success Criteria**: Linear fit R² > 0.99 for NEXUS

---

#### 1.2 Throughput Benchmark

**Objective**: Measure practical tokens/second throughput

**Protocol**:
- Batch sizes: [1, 4, 16, 64]
- Sequence lengths: [512, 2048, 8192]
- Hardware: A100 40GB, H100 80GB
- Precision: FP16, BF16

**Metrics**:
- Tokens/second (inference)
- Tokens/second (training)
- Memory efficiency (tokens/GB)

---

#### 1.3 Latency Benchmark

**Objective**: Measure time-to-first-token and generation speed

**Protocol**:
- Prompt lengths: [100, 1000, 10000, 50000]
- Generation lengths: [50, 100, 500]
- Measure P50, P95, P99 latencies

**Success Criteria**: NEXUS latency < Transformer latency at all scales

---

### Category 2: Language Modeling Quality

#### 2.1 Perplexity Benchmarks

**Datasets**:
| Dataset | Domain | Size | Sequence Length |
|---------|--------|------|-----------------|
| WikiText-103 | General | 103M tokens | Standard |
| PG-19 | Books | 1B tokens | Long-form |
| ArXiv | Scientific | 500M tokens | Technical |
| GitHub Code | Programming | 1B tokens | Code |

**Metrics**: Perplexity (PPL), bits-per-byte (BPB)

**Success Criteria**: Within 5% of Transformer baseline

---

#### 2.2 Long-Context Language Modeling

**Objective**: Evaluate quality on very long sequences

**Datasets**:
- Long Range Arena (LRA)
- PG-19 (book-level)
- Scrolls benchmark

**Protocol**:
- Evaluate at 4K, 16K, 64K, 128K token contexts
- Measure perplexity degradation with length

**Success Criteria**: Less perplexity degradation than Transformer at >32K

---

### Category 3: Reasoning Benchmarks

#### 3.1 Mathematical Reasoning

**Benchmarks**:
| Benchmark | Task | Metric |
|-----------|------|--------|
| GSM8K | Grade school math | Accuracy |
| MATH | Competition math | Accuracy |
| MMLU-Math | Multiple choice math | Accuracy |
| MathQA | Math word problems | Accuracy |

**NEXUS-Specific Metrics**:
- Proof trace validity rate
- Reasoning step accuracy
- Error localization precision

---

#### 3.2 Logical Reasoning

**Benchmarks**:
| Benchmark | Task | Metric |
|-----------|------|--------|
| LogiQA | Logical inference | Accuracy |
| ReClor | Reading comprehension + logic | Accuracy |
| FOLIO | First-order logic | Accuracy |
| ProofWriter | Proof generation | Proof validity |

**Protocol**:
- Compare final answer accuracy
- Evaluate proof/reasoning trace quality
- Test logical consistency

---

#### 3.3 Commonsense Reasoning

**Benchmarks**:
- HellaSwag
- WinoGrande
- ARC (AI2 Reasoning Challenge)
- PIQA (Physical Intuition QA)

---

### Category 4: Causal Reasoning

#### 4.1 Causal Discovery Benchmarks

**Objective**: Evaluate causal graph learning accuracy

**Datasets**:
| Dataset | Variables | Samples | Ground Truth |
|---------|-----------|---------|--------------|
| Sachs | 11 | 7,466 | Protein network |
| Asia | 8 | Synthetic | Known DAG |
| Alarm | 37 | Synthetic | Medical |
| Child | 20 | Synthetic | Known DAG |

**Metrics**:
- Structural Hamming Distance (SHD)
- Structural Intervention Distance (SID)
- F1 score for edge detection

---

#### 4.2 Intervention Prediction

**Objective**: Predict effects of interventions

**Protocol**:
```
For each causal benchmark:
1. Train on observational data
2. Test intervention predictions: P(Y | do(X))
3. Compare to ground truth interventional data
```

**Metrics**:
- Intervention effect estimation error
- Ranking correlation for intervention effects

---

#### 4.3 Counterfactual Reasoning

**Objective**: Evaluate counterfactual accuracy

**Benchmarks**:
- CLADDER (Causal Ladder benchmark)
- Counterfactual reasoning datasets
- Custom counterfactual evaluation

**Metrics**:
- Counterfactual accuracy
- Consistency checks (same abduction → same counterfactual)

---

#### 4.4 Causal NLP Benchmarks

**Benchmarks**:
| Benchmark | Task | NEXUS Advantage |
|-----------|------|-----------------|
| COPA | Causal reasoning | Native causal engine |
| XCOPA | Cross-lingual causal | Causal transfer |
| e-CARE | Causal explanation | Proof traces |
| Causal QA | Causal questions | Intervention reasoning |

---

### Category 5: World Modeling

#### 5.1 Prediction Quality

**Objective**: Evaluate world model prediction accuracy

**Protocol**:
```
1. Encode sequence: z = encode(x[1:t])
2. Predict future: z_pred = world_model.imagine(z, n_steps=k)
3. Compare to actual: z_actual = encode(x[t+1:t+k])
4. Measure: MSE(z_pred, z_actual)
```

**Metrics**:
- Prediction MSE at various horizons
- Correlation between predicted and actual states
- Uncertainty calibration

---

#### 5.2 Planning Benchmarks

**Objective**: Evaluate planning through world model simulation

**Benchmarks**:
- MiniHack planning tasks
- ALFWorld (text-based planning)
- WebShop (web navigation)

**Metrics**:
- Task completion rate
- Planning efficiency (steps to goal)
- Success vs. computation tradeoff

---

#### 5.3 Video/Sequence Prediction

**Objective**: Predict future states in structured domains

**Benchmarks**:
- Moving MNIST
- KTH Actions
- RoboNet

**Metrics**:
- Prediction accuracy
- Structural similarity (SSIM)
- FVD (Fréchet Video Distance)

---

### Category 6: Long-Context Benchmarks

#### 6.1 Needle-in-a-Haystack

**Objective**: Retrieve specific information from very long contexts

**Protocol**:
- Context lengths: [4K, 16K, 64K, 128K, 256K]
- Needle positions: [start, 25%, 50%, 75%, end]
- Measure retrieval accuracy

**Success Criteria**: >95% accuracy at all positions up to 128K

---

#### 6.2 Multi-Document QA

**Benchmarks**:
- HotpotQA
- MuSiQue
- Qasper
- QuALITY

**Protocol**:
- Provide multiple documents as context
- Test reasoning across documents
- Measure accuracy and source attribution

---

#### 6.3 Book-Level Understanding

**Benchmarks**:
- NarrativeQA (book summaries)
- BookSum (summarization)
- Custom book comprehension tasks

**Metrics**:
- Question answering accuracy
- Summarization quality (ROUGE, BERTScore)
- Character/plot tracking accuracy

---

### Category 7: Compositional Generalization

#### 7.1 Systematic Generalization

**Benchmarks**:
| Benchmark | Task | Challenge |
|-----------|------|-----------|
| SCAN | Command → action sequence | Novel compositions |
| COGS | Semantic parsing | Structural generalization |
| CFQ | SPARQL generation | Compositional complexity |
| gSCAN | Grounded commands | Spatial + compositional |

**NEXUS Advantage**: Symbolic reasoning aids composition

---

#### 7.2 Algorithmic Tasks

**Benchmarks**:
- Sorting (various input sizes)
- Copying (various lengths)
- Addition (multi-digit)
- Parity (long sequences)

**Protocol**:
- Train on small instances
- Test generalization to larger instances
- Measure length generalization

---

### Category 8: Robustness & Safety

#### 8.1 Distribution Shift

**Protocol**:
- Train on domain A
- Test on related domain B
- Measure performance degradation

**Datasets**:
- WILDS collection
- Domain adaptation benchmarks

---

#### 8.2 Adversarial Robustness

**Attacks**:
- TextFooler
- BERT-Attack
- Semantic adversaries

**Metrics**:
- Attack success rate
- Performance under attack
- Reasoning consistency under perturbation

---

#### 8.3 Calibration

**Objective**: Test if confidence matches accuracy

**Metrics**:
- Expected Calibration Error (ECE)
- Brier score
- Reliability diagrams

---

## NEXUS-Specific Evaluations

### Proof Trace Evaluation

**Metrics**:
| Metric | Description |
|--------|-------------|
| Validity Rate | % of proofs that are logically valid |
| Completeness | % of relevant rules used |
| Groundedness | % of conclusions grounded in facts |
| Consistency | % of non-contradictory proofs |

**Protocol**:
```python
for sample in test_set:
    output, proof_trace = model.reason(sample)
    metrics['validity'] += verify_proof(proof_trace)
    metrics['groundedness'] += check_grounding(proof_trace, sample)
    metrics['consistency'] += check_consistency(proof_trace)
```

---

### Adaptive Computation Evaluation

**Metrics**:
| Metric | Description |
|--------|-------------|
| Compute vs. Difficulty | Correlation between energy iterations and task difficulty |
| Early Exit Rate | % of easy samples that exit early |
| Accuracy vs. Compute | Pareto frontier |

**Protocol**:
```python
for sample in test_set:
    output, energy_trace = model(sample)
    record:
        - num_iterations
        - final_energy
        - task_difficulty (ground truth)
        - accuracy
```

---

### World Model Evaluation

**Metrics**:
| Metric | Description |
|--------|-------------|
| Imagination Quality | MSE in representation space |
| Planning Success | Task completion via imagination |
| Consistency | Repeated imaginations converge |

---

## Experimental Design

### Baseline Models

| Model | Type | Purpose |
|-------|------|---------|
| GPT-2/GPT-3 | Transformer | Quality baseline |
| Mamba | SSM | Efficiency baseline |
| LLaMA | Transformer | Open-source baseline |
| Mistral | Transformer | Efficient Transformer |

### Ablation Studies

| Ablation | Purpose |
|----------|---------|
| NEXUS w/o World Model | Importance of world modeling |
| NEXUS w/o Reasoner | Importance of symbolic reasoning |
| NEXUS w/o Causal | Importance of causal engine |
| NEXUS w/o Energy | Importance of adaptive compute |
| NEXUS SSM only | Pure efficiency baseline |

### Hyperparameter Sensitivity

Test sensitivity to:
- Model scale (d_model, n_layers)
- World model parameters
- Reasoning module capacity
- Energy iteration limits
- Loss weight balancing

---

## Implementation Plan

### Phase 1: Core Benchmarks (Weeks 1-4)

- [ ] Scaling benchmark implementation
- [ ] Perplexity evaluation pipeline
- [ ] Basic reasoning benchmarks

### Phase 2: Advanced Benchmarks (Weeks 5-8)

- [ ] Causal benchmark suite
- [ ] Long-context evaluations
- [ ] World model benchmarks

### Phase 3: NEXUS-Specific (Weeks 9-12)

- [ ] Proof trace evaluation
- [ ] Adaptive computation analysis
- [ ] Ablation studies

### Phase 4: Comprehensive Evaluation (Weeks 13-16)

- [ ] Full benchmark suite execution
- [ ] Baseline comparisons
- [ ] Paper-ready results

---

## Reporting Standards

### Required for Each Experiment

1. **Configuration**: Full hyperparameters
2. **Hardware**: GPU type, count, memory
3. **Randomness**: Seeds, number of runs
4. **Metrics**: Mean ± std across runs
5. **Compute**: Total GPU hours
6. **Code**: Reproducibility artifacts

### Statistical Rigor

- Minimum 3 runs with different seeds
- Report confidence intervals
- Use appropriate statistical tests
- Document any data snooping

---

## Success Criteria Summary

| Category | Minimum Success | Target Success |
|----------|-----------------|----------------|
| Efficiency | O(n) verified | 10x faster than Transformer at 32K |
| Quality | Within 10% of Transformer | Match Transformer |
| Reasoning | 70% proof validity | 90% proof validity |
| Causality | Beat correlational baseline | Match causal ground truth |
| Long-Context | Work at 64K | Work at 256K |
| Robustness | Not worse than Transformer | Better calibration |

---

## Appendix: Benchmark Resources

### Datasets

| Dataset | Source | License |
|---------|--------|---------|
| WikiText-103 | HuggingFace | CC BY-SA |
| PG-19 | DeepMind | Apache 2.0 |
| GSM8K | OpenAI | MIT |
| CLADDER | Papers with Code | Research |

### Compute Estimates

| Benchmark Suite | GPU Hours (A100) |
|-----------------|----------------:|
| Scaling | 50 |
| Quality | 200 |
| Reasoning | 100 |
| Causal | 150 |
| Long-Context | 300 |
| **Total** | **800** |

---

*"In God we trust. All others must bring data." — W. Edwards Deming*
