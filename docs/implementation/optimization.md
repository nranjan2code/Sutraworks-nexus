# Performance Optimization

## Optimizing NEXUS for Production

This guide covers techniques to maximize NEXUS performance in both training and inference.

---

## Optimization Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Optimization Strategies                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │   Memory    │  │   Compute   │  │    I/O      │  │   Model     │      │
│   │   Optimiz.  │  │   Optimiz.  │  │   Optimiz.  │  │   Optimiz.  │      │
│   ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤      │
│   │ • Gradient  │  │ • Mixed     │  │ • Data      │  │ • Pruning   │      │
│   │   checkpt   │  │   precision │  │   loading   │  │ • Quantiz.  │      │
│   │ • Activation│  │ • Fusion    │  │ • Prefetch  │  │ • Distill   │      │
│   │   offload   │  │ • Compile   │  │ • Caching   │  │ • Sparsity  │      │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Optimization

### 1. Gradient Checkpointing

Trade compute for memory by recomputing activations during backward pass.

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientNEXUS(NEXUSCore):
    """NEXUS with gradient checkpointing."""
    
    def __init__(self, *args, checkpoint_layers=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_layers = checkpoint_layers
    
    def forward(self, input_ids, return_all_outputs=False):
        x = self.embedding(input_ids)
        
        # Checkpoint each state space layer
        for layer in self.state_space.layers:
            if self.checkpoint_layers and self.training:
                x = checkpoint(
                    layer,
                    x,
                    use_reentrant=False
                )
            else:
                x = layer(x)
        
        # Continue with rest of forward pass
        # ...
        
        return x
```

**Memory savings**: ~40-60% reduction in activation memory.

### 2. Mixed Precision Training

Use FP16/BF16 for forward pass, FP32 for gradients.

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, use_bf16=True):
        self.model = model
        self.scaler = GradScaler()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    def train_step(self, batch, optimizer):
        optimizer.zero_grad()
        
        with autocast(dtype=self.dtype):
            output = self.model(batch['input_ids'])
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                batch['labels'].view(-1)
            )
        
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()
```

**Benefits**:
- 2x memory reduction
- 2-3x speedup on modern GPUs

### 3. Activation Offloading

Offload activations to CPU during forward pass.

```python
class OffloadingWrapper:
    """Offload tensors to CPU during forward."""
    
    def __init__(self, fraction=0.5):
        self.fraction = fraction
        self.offloaded = {}
    
    def offload(self, name, tensor):
        if np.random.random() < self.fraction:
            self.offloaded[name] = tensor.cpu()
            return None
        return tensor
    
    def restore(self, name):
        if name in self.offloaded:
            tensor = self.offloaded.pop(name).cuda()
            return tensor
        return None
```

### 4. Flash Attention

Use memory-efficient attention implementations.

```python
# With PyTorch 2.0+
from torch.nn.functional import scaled_dot_product_attention

class FlashAttention(nn.Module):
    """Memory-efficient attention using Flash Attention."""
    
    def forward(self, q, k, v, mask=None):
        # Automatically uses Flash Attention kernel
        return scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=True
        )
```

---

## Compute Optimization

### 1. torch.compile (PyTorch 2.0+)

```python
import torch

# Compile model for optimized kernels
model = torch.compile(model, mode='reduce-overhead')

# For maximum performance (longer compile time)
model = torch.compile(model, mode='max-autotune')

# Fullgraph mode for best optimization
model = torch.compile(model, fullgraph=True)
```

**Speedup**: 20-50% on typical workloads.

### 2. Kernel Fusion

Fuse operations to reduce memory bandwidth.

```python
class FusedNorm(nn.Module):
    """Fused LayerNorm + activation."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
    
    @torch.jit.script_method
    def forward(self, x):
        # JIT will fuse these operations
        return F.gelu(self.norm(x))


class FusedLinearGELU(nn.Module):
    """Fused Linear + GELU."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        # Use fused kernel
        return F.gelu(self.linear(x), approximate='tanh')
```

### 3. Custom CUDA Kernels

For critical operations, use custom kernels.

```python
# Example: Using Triton for custom kernels
import triton
import triton.language as tl

@triton.jit
def fused_selective_scan_kernel(
    x_ptr, A_ptr, B_ptr, C_ptr, out_ptr,
    seq_len, state_dim,
    BLOCK_SIZE: tl.constexpr
):
    """Fused selective scan kernel."""
    pid = tl.program_id(0)
    # ... kernel implementation
    pass
```

### 4. Parallel Scan Implementation

Efficient parallel implementation of state space scan.

```python
class ParallelSelectiveScan(nn.Module):
    """Parallel-friendly selective scan."""
    
    def forward(self, x, A, B, C, delta):
        # Reshape for parallel processing
        batch, seq_len, hidden_dim = x.shape
        
        # Use associative scan
        # (can be parallelized across sequence)
        
        # Compute in log(N) steps using tree reduction
        return self._parallel_scan(x, A, B, C, delta)
    
    def _parallel_scan(self, x, A, B, C, delta):
        """Parallel scan using tree reduction."""
        
        seq_len = x.size(1)
        
        # Work up the tree
        for d in range(int(np.log2(seq_len))):
            step = 2 ** (d + 1)
            half_step = 2 ** d
            
            # Parallel over all positions
            indices = torch.arange(half_step - 1, seq_len, step)
            # Combine pairs
            # ...
        
        # Work down the tree
        # ...
        
        return x
```

---

## I/O Optimization

### 1. Efficient Data Loading

```python
class OptimizedDataLoader:
    """Optimized data loading pipeline."""
    
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers=8,
        prefetch_factor=4
    ):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            drop_last=True
        )
    
    def __iter__(self):
        for batch in self.loader:
            # Non-blocking transfer to GPU
            yield {
                k: v.cuda(non_blocking=True)
                for k, v in batch.items()
            }
```

### 2. Memory-Mapped Data

```python
import numpy as np

class MemmapDataset:
    """Memory-mapped dataset for large data."""
    
    def __init__(self, data_path, seq_length):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) // self.seq_length
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        
        tokens = self.data[start:end].astype(np.int64)
        return torch.from_numpy(tokens)
```

### 3. Async Prefetching

```python
import threading
import queue

class AsyncPrefetcher:
    """Prefetch batches asynchronously."""
    
    def __init__(self, dataloader, device, num_prefetch=2):
        self.dataloader = dataloader
        self.device = device
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        
        self.thread = threading.Thread(target=self._prefetch)
        self.thread.start()
    
    def _prefetch(self):
        stream = torch.cuda.Stream()
        
        for batch in self.dataloader:
            if self.stop_event.is_set():
                break
            
            with torch.cuda.stream(stream):
                batch = {k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()}
            
            stream.synchronize()
            self.queue.put(batch)
    
    def __iter__(self):
        while True:
            try:
                batch = self.queue.get(timeout=60)
                yield batch
            except queue.Empty:
                break
    
    def stop(self):
        self.stop_event.set()
        self.thread.join()
```

---

## Model Optimization

### 1. Quantization

```python
import torch.quantization as quant

class QuantizedNEXUS:
    """Post-training quantization for NEXUS."""
    
    @staticmethod
    def quantize_dynamic(model):
        """Dynamic quantization (weights only)."""
        return quant.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
    
    @staticmethod
    def quantize_static(model, calibration_data):
        """Static quantization (weights + activations)."""
        model.eval()
        
        # Prepare for quantization
        model.qconfig = quant.get_default_qconfig('fbgemm')
        quant.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)
        
        # Convert
        quant.convert(model, inplace=True)
        
        return model


# INT8 inference
def int8_inference(model, input_ids):
    """Run inference with INT8 model."""
    quantized = QuantizedNEXUS.quantize_dynamic(model)
    return quantized(input_ids)
```

### 2. Pruning

```python
import torch.nn.utils.prune as prune

class PrunedNEXUS:
    """Structured and unstructured pruning."""
    
    @staticmethod
    def prune_unstructured(model, amount=0.3):
        """Remove individual weights."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=amount)
        return model
    
    @staticmethod
    def prune_structured(model, amount=0.2):
        """Remove entire neurons/channels."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(
                    module, 'weight',
                    amount=amount,
                    n=2,
                    dim=0
                )
        return model
    
    @staticmethod
    def make_permanent(model):
        """Make pruning permanent (remove masks)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass
        return model
```

### 3. Knowledge Distillation

```python
class DistillationTrainer:
    """Train smaller model from larger teacher."""
    
    def __init__(
        self,
        teacher,
        student,
        temperature=2.0,
        alpha=0.5
    ):
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
    
    def train_step(self, batch):
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(batch['input_ids'])
        
        # Get student predictions
        student_logits = self.student(batch['input_ids'])
        
        # Soft targets (distillation loss)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        distill_loss *= self.temperature ** 2
        
        # Hard targets (standard loss)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            batch['labels'].view(-1)
        )
        
        # Combined loss
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return loss
```

---

## Inference Optimization

### 1. KV-Cache

```python
class KVCache:
    """Key-Value cache for efficient generation."""
    
    def __init__(self, max_length, batch_size, num_heads, head_dim):
        self.k_cache = torch.zeros(batch_size, num_heads, max_length, head_dim)
        self.v_cache = torch.zeros(batch_size, num_heads, max_length, head_dim)
        self.position = 0
    
    def update(self, k, v):
        seq_len = k.size(2)
        self.k_cache[:, :, self.position:self.position + seq_len] = k
        self.v_cache[:, :, self.position:self.position + seq_len] = v
        self.position += seq_len
    
    def get(self):
        return self.k_cache[:, :, :self.position], self.v_cache[:, :, :self.position]
```

### 2. Speculative Decoding

```python
class SpeculativeDecoder:
    """Speed up generation with draft model."""
    
    def __init__(self, target_model, draft_model, k=4):
        self.target = target_model
        self.draft = draft_model
        self.k = k  # Speculation length
    
    def generate(self, prompt, max_tokens):
        generated = prompt.clone()
        
        while generated.size(1) < max_tokens:
            # Draft k tokens
            draft_tokens = []
            for _ in range(self.k):
                draft_logits = self.draft(generated)
                next_token = draft_logits[:, -1].argmax(dim=-1)
                draft_tokens.append(next_token)
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            # Verify with target model
            target_logits = self.target(generated)
            
            # Accept/reject draft tokens
            # (compare probabilities)
            # ...
        
        return generated
```

### 3. Batched Generation

```python
class BatchedGenerator:
    """Efficient batched generation."""
    
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
    
    def generate_batch(self, prompts, max_new_tokens):
        # Pad prompts to same length
        padded, lengths = self._pad_prompts(prompts)
        
        # Generate
        for _ in range(max_new_tokens):
            logits = self.model(padded)
            next_tokens = logits[:, -1].argmax(dim=-1)
            padded = torch.cat([padded, next_tokens.unsqueeze(1)], dim=1)
        
        # Unpad and return
        return self._unpad(padded, lengths)
```

---

## Profiling and Benchmarking

### Profile Memory

```python
def profile_memory(model, input_shape):
    """Profile GPU memory usage."""
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Forward pass
    x = torch.randn(*input_shape).cuda()
    output = model(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"Current memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

### Profile Compute

```python
def profile_compute(model, input_shape, num_warmup=10, num_runs=100):
    """Profile compute time."""
    
    import time
    
    x = torch.randn(*input_shape).cuda()
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(x)
    
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    print(f"Mean latency: {np.mean(times) * 1000:.2f} ms")
    print(f"Std latency: {np.std(times) * 1000:.2f} ms")
    print(f"Throughput: {input_shape[0] * input_shape[1] / np.mean(times):.0f} tokens/sec")
```

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

def detailed_profile(model, input_ids):
    """Detailed profiling with PyTorch profiler."""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_forward"):
            output = model(input_ids)
    
    # Print summary
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    # Export for visualization
    prof.export_chrome_trace("trace.json")
```

---

## Optimization Checklist

### Training
- [ ] Enable mixed precision (FP16/BF16)
- [ ] Use gradient checkpointing for large models
- [ ] Optimize batch size for GPU memory
- [ ] Enable torch.compile
- [ ] Use efficient data loading (num_workers, pin_memory)
- [ ] Gradient accumulation for effective larger batches

### Inference
- [ ] Quantize model (INT8/FP16)
- [ ] Enable KV-cache for generation
- [ ] Use batched inference when possible
- [ ] Consider speculative decoding
- [ ] Profile and optimize bottlenecks

### Model
- [ ] Consider pruning for deployment
- [ ] Distill to smaller model if needed
- [ ] Remove unused modules
- [ ] Optimize module-specific compute

---

## Next Steps

- [Deployment Guide](deployment.md) - Deploy optimized models
- [API Reference](../api/optimization.md) - Optimization API
- [Benchmarks](../implementation/evaluation.md) - Measure improvements

---

*Performance is a feature. Optimize relentlessly.*
