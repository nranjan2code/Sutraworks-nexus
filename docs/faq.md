# FAQ & Troubleshooting

## Frequently Asked Questions and Solutions

---

## General Questions

### What is NEXUS?

NEXUS (Neural EXploratory Unified Synthesis) is a **living AI system** that:
- **Never hallucinates** - refuses politely when uncertain
- **Learns continuously** - every interaction is a learning opportunity
- **Evolves organically** - no stages or labels, just smooth continuous growth
- Achieves O(n) efficiency vs Transformer's O(n²)

It combines five paradigms:
- Selective State Space Models (O(n) sequence processing)
- Hierarchical World Models (predictive representations)
- Neuro-Symbolic Reasoning (hybrid reasoning)
- Energy-Based Adaptive Computation (dynamic compute)
- Causal Inference Engine (causal understanding)

### How is NEXUS different from Transformers?

| Aspect | Transformer | NEXUS |
|--------|-------------|-------|
| Complexity | O(n²) | O(n) |
| Memory | O(n²) per layer | O(n) per layer |
| Context length | Limited (128K max) | Theoretically unlimited |
| Reasoning | Implicit | Explicit symbolic |
| Computation | Fixed | Adaptive |
| Causality | Correlational | Causal |
| Hallucination | Prone to fabricate | Refuses when uncertain |
| Learning | Static after training | Continuous while serving |

### What tasks is NEXUS good for?

- Long-context understanding
- Complex reasoning tasks
- Causal reasoning and planning
- Tasks requiring variable computation
- Scenarios needing interpretability

### What are the hardware requirements?

**Minimum (for small models)**:
- 4GB GPU memory
- 16GB RAM
- PyTorch 2.0+

**Recommended (for training)**:
- 24GB+ GPU memory
- 64GB RAM
- Multiple GPUs for large models

---

## Installation

### Q: Installation fails with "CUDA not found"

**Solution**: Install CUDA toolkit or use CPU-only version:
```bash
# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or install CUDA toolkit
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Q: ImportError: No module named 'nexus'

**Solution**: Install NEXUS in development mode:
```bash
cd /path/to/nexus
pip install -e .
```

### Q: Dependency conflicts

**Solution**: Use a fresh virtual environment:
```bash
python -m venv nexus_env
source nexus_env/bin/activate  # Linux/Mac
# or: nexus_env\Scripts\activate  # Windows

pip install -r requirements.txt
pip install -e .
```

---

## Training

### Q: Training loss is NaN

**Possible causes and solutions**:

1. **Learning rate too high**:
```python
# Reduce learning rate
config.learning_rate = 1e-5  # Try lower
```

2. **Missing gradient clipping**:
```python
config.max_grad_norm = 0.5  # Enable/reduce
```

3. **Numerical instability**:
```python
# Use BF16 instead of FP16
config.dtype = 'bfloat16'
```

4. **Bad initialization**:
```python
# Re-initialize model
model = NEXUSCore(...)
```

### Q: Out of Memory (OOM) during training

**Solutions**:

1. **Reduce batch size**:
```python
config.batch_size = 8  # Reduce
config.gradient_accumulation_steps = 4  # Compensate
```

2. **Enable gradient checkpointing**:
```python
model.state_space.gradient_checkpointing = True
```

3. **Reduce sequence length**:
```python
config.max_seq_length = 1024  # Reduce
```

4. **Use mixed precision**:
```python
config.fp16 = True
```

### Q: Training is very slow

**Solutions**:

1. **Enable torch.compile**:
```python
model = torch.compile(model)
```

2. **Optimize data loading**:
```python
dataloader = DataLoader(
    dataset,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4
)
```

3. **Use larger batches with accumulation**:
```python
config.batch_size = 64
config.gradient_accumulation_steps = 1
```

### Q: Loss not decreasing

**Possible causes**:

1. **Learning rate too low**: Try 1e-3 or 1e-4
2. **Data issue**: Check data quality and tokenization
3. **Model too small**: Increase d_model or ssm_n_layers
4. **Bug in loss function**: Verify loss computation

---

## Inference

### Q: Generation is slow

**Solutions**:

1. **Use KV-cache** (for attention if used):
```python
model.enable_kv_cache()
```

2. **Batch multiple prompts**:
```python
# Process multiple prompts together
outputs = model.generate(batch_of_prompts)
```

3. **Quantize model**:
```python
from torch.quantization import quantize_dynamic
model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

### Q: Poor generation quality

**Solutions**:

1. **Adjust temperature**:
```python
# Lower for more focused
output = model.generate(prompt, temperature=0.7)
```

2. **Use top-k/top-p sampling**:
```python
output = model.generate(prompt, top_k=40, top_p=0.95)
```

3. **Increase reasoning steps**:
```python
model.reasoner.max_steps = 15
```

### Q: Inconsistent outputs

**Cause**: Randomness in sampling and modules

**Solutions**:
```python
# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Use deterministic generation
output = model.generate(prompt, do_sample=False)  # Greedy
```

---

## Module-Specific

### State Space

**Q: State space layer is slow**

Ensure efficient implementation:
```python
# Use parallel scan (if available)
model.state_space.use_parallel_scan = True
```

**Q: Long sequences cause issues**

Check state accumulation:
```python
# Reset state periodically for very long sequences
model.state_space.reset_state()
```

### World Model

**Q: World model predictions are poor**

1. Check EMA update frequency:
```python
# Call after each step
model.world_model.update_target()
```

2. Adjust prediction horizon:
```python
model.world_model.prediction_horizon = 5  # Shorter for stability
```

### Reasoner

**Q: Reasoning doesn't improve outputs**

1. Increase reasoning steps:
```python
model.reasoner.max_steps = 10
```

2. Check proof generation:
```python
output, proof = model.reasoner(hidden, return_proof=True)
print(f"Proof length: {len(proof['steps'])}")
```

### Energy Module

**Q: Always using max iterations**

Check convergence threshold:
```python
# Lower threshold for earlier convergence
model.energy.convergence_threshold = 0.1

# Or check input complexity
_, info = model(input_ids, return_all_outputs=True)
print(f"Iterations: {info['energy_info']['iterations']}")
```

**Q: Early exit too aggressive**

Adjust confidence threshold:
```python
model.energy.confidence_threshold = 0.99  # Higher = less early exit
```

### Causal Engine

**Q: Causal graph has cycles**

DAG constraint may not be enforced strongly:
```python
# Increase DAG constraint weight
model.causal.dag_weight = 10.0
```

**Q: Poor causal discovery**

1. More training data needed
2. Increase sparsity:
```python
model.causal.sparsity_weight = 0.1
```

---

## Performance

### Memory Usage

**Estimate memory usage**:
```python
def estimate_memory(model, batch_size, seq_length):
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    activations = batch_size * seq_length * model.config.d_model * 4  # Rough estimate
    
    print(f"Parameters: {params / 1024**2:.1f} MB")
    print(f"Activations (estimate): {activations / 1024**2:.1f} MB")
    print(f"Total (estimate): {(params + activations) / 1024**2:.1f} MB")
```

### Latency Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
) as prof:
    output = model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## Common Errors

### `RuntimeError: CUDA out of memory`

See OOM solutions above.

### `ValueError: Expected tensor with X dimensions, got Y`

Shape mismatch - check input shapes:
```python
print(f"Input shape: {input_ids.shape}")
# Should be [batch_size, seq_length]
```

### `AttributeError: 'NEXUSCore' object has no attribute 'X'`

1. Check version compatibility
2. Ensure proper import:
```python
from nexus.core import NEXUSCore
```

### `AssertionError: Causal graph is not DAG`

Graph has cycles:
```python
# Re-initialize causal module
from nexus.core.causal import CausalInferenceEngine
model.causal_engine = CausalInferenceEngine(model.config.get_causal_config())
# Or increase DAG constraint during training
```

---

## Getting Help

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or specific to NEXUS
logging.getLogger('nexus').setLevel(logging.DEBUG)
```

### Diagnostic Script

```python
def diagnose_nexus(model, sample_input):
    """Run diagnostic checks on NEXUS model."""
    
    print("=== NEXUS Diagnostics ===\n")
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
    
    # Forward pass test
    try:
        output = model(sample_input)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output['logits'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    # Module outputs
    try:
        output, info = model(sample_input, return_all_outputs=True)
        print(f"✓ Module outputs available")
        print(f"  Energy iterations: {info['energy_info']['iterations']}")
    except Exception as e:
        print(f"✗ Module outputs failed: {e}")
    
    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = model(sample_input)
        memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"✓ Peak GPU memory: {memory:.1f} MB")
    
    print("\n=== End Diagnostics ===")

# Usage
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
config = NEXUSConfig(vocab_size=32000, d_model=256)
model = NEXUSCore(config).cuda()
sample = torch.randn(1, 100, config.d_model).cuda()
diagnose_nexus(model, sample)
```

### Report Issues

When reporting issues, include:
1. NEXUS version
2. PyTorch version
3. Hardware info (GPU, memory)
4. Minimal reproduction code
5. Full error traceback
6. Diagnostic script output

---

## Additional Resources

- [Architecture Documentation](architecture/overview.md)
- [API Reference](api/reference.md)
- [Tutorials](tutorials/getting-started.md)
- [GitHub Issues](https://github.com/sutraworks/nexus/issues)

---

*Can't find your answer? Open an issue on GitHub.*
