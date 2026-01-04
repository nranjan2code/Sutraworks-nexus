# Training Pipeline

## Comprehensive Guide to Training NEXUS Models

This document covers the complete training pipeline for NEXUS, including data preparation, loss functions, optimization, and best practices.

---

## Training Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NEXUS Training Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │    Data     │ ─► │   Model     │ ─► │   Loss      │ ─► │ Optimizer   │ │
│  │  Pipeline   │    │   Forward   │    │  Compute    │    │   Step      │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│        │                  │                  │                  │         │
│        │                  │                  │                  │         │
│        ▼                  ▼                  ▼                  ▼         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  • Tokenize │    │  • SSM      │    │  • LM Loss  │    │  • AdamW    │ │
│  │  • Batch    │    │  • World    │    │  • World    │    │  • LR Sched │ │
│  │  • Collate  │    │  • Reason   │    │  • Reason   │    │  • Clip     │ │
│  │             │    │  • Energy   │    │  • Energy   │    │  • Update   │ │
│  │             │    │  • Causal   │    │  • Causal   │    │             │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Dataset Format

NEXUS supports multiple data formats:

#### 1. Raw Text Format
```python
# Simple text file, one document per line
data/
├── train.txt
├── valid.txt
└── test.txt
```

#### 2. JSONL Format
```jsonl
{"text": "This is a training example.", "metadata": {"source": "wiki"}}
{"text": "Another example with more context.", "metadata": {"source": "books"}}
```

#### 3. Pre-tokenized Format
```python
# NumPy memmap for large datasets
data/
├── train.bin      # Tokenized IDs as np.uint16
├── train.idx      # Document boundaries
├── valid.bin
└── valid.idx
```

### NEXUSDataset

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class NEXUSDataset(Dataset):
    """Dataset for NEXUS training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        mode: str = 'train'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # Load data based on format
        if data_path.endswith('.txt'):
            self.data = self._load_text(data_path)
        elif data_path.endswith('.jsonl'):
            self.data = self._load_jsonl(data_path)
        elif data_path.endswith('.bin'):
            self.data = self._load_memmap(data_path)
        else:
            raise ValueError(f"Unknown format: {data_path}")
    
    def _load_text(self, path):
        """Load plain text file."""
        with open(path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]
    
    def _load_jsonl(self, path):
        """Load JSONL file."""
        import json
        data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item['text'])
        return data
    
    def _load_memmap(self, path):
        """Load pre-tokenized memmap."""
        return np.memmap(path, dtype=np.uint16, mode='r')
    
    def __len__(self):
        if isinstance(self.data, np.memmap):
            return len(self.data) // self.max_length
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, np.memmap):
            # Pre-tokenized: slice directly
            start = idx * self.max_length
            end = start + self.max_length
            tokens = self.data[start:end].astype(np.int64)
            return {
                'input_ids': torch.tensor(tokens[:-1]),
                'labels': torch.tensor(tokens[1:])
            }
        else:
            # Raw text: tokenize on-the-fly
            text = self.data[idx]
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True
            )
            
            # Pad if needed
            if len(tokens) < self.max_length:
                tokens = tokens + [self.tokenizer.pad_token_id] * (
                    self.max_length - len(tokens)
                )
            
            return {
                'input_ids': torch.tensor(tokens[:-1]),
                'labels': torch.tensor(tokens[1:])
            }


class NEXUSCollator:
    """Collate function for batching."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
```

### DataLoader Creation

```python
from torch.utils.data import DataLoader

def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """Create optimized DataLoader."""
    
    collator = NEXUSCollator()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
```

---

## Loss Functions

### Multi-Objective NEXUS Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NEXUSLoss(nn.Module):
    """Combined multi-objective loss for NEXUS training."""
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        world_weight: float = 0.1,
        reason_weight: float = 0.05,
        energy_weight: float = 0.05,
        causal_weight: float = 0.05,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.weights = {
            'lm': lm_weight,
            'world': world_weight,
            'reason': reason_weight,
            'energy': energy_weight,
            'causal': causal_weight
        }
        
        # Language modeling loss
        self.lm_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing
        )
        
        # Module-specific losses
        self.world_loss = WorldModelLoss()
        self.reason_loss = ReasoningLoss()
        self.energy_loss = EnergyLoss()
        self.causal_loss = CausalLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        module_outputs: dict = None
    ):
        """
        Compute combined loss.
        
        Args:
            logits: [batch, seq_len, vocab_size] model output
            labels: [batch, seq_len] target token IDs
            module_outputs: dict with module-specific outputs
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values
        """
        losses = {}
        
        # Primary language modeling loss
        losses['lm'] = self.lm_loss(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Module losses (if available)
        if module_outputs is not None:
            if 'world' in module_outputs:
                losses['world'] = self.world_loss(
                    module_outputs['world'],
                    module_outputs.get('world_targets')
                )
            
            if 'reason' in module_outputs:
                losses['reason'] = self.reason_loss(
                    module_outputs['reason']
                )
            
            if 'energy' in module_outputs:
                losses['energy'] = self.energy_loss(
                    module_outputs['energy']
                )
            
            if 'causal' in module_outputs:
                losses['causal'] = self.causal_loss(
                    module_outputs['causal']
                )
        
        # Weighted combination
        total_loss = sum(
            self.weights.get(name, 0.0) * loss
            for name, loss in losses.items()
        )
        
        return total_loss, losses


class WorldModelLoss(nn.Module):
    """Loss for world model predictions."""
    
    def __init__(self, variance_weight: float = 0.1):
        super().__init__()
        self.variance_weight = variance_weight
    
    def forward(self, predictions, targets=None):
        """
        Prediction loss for world model.
        
        Uses self-supervised target from EMA encoder.
        """
        if targets is None:
            # Self-supervised: use variance regularization
            return self.variance_weight * predictions.var(dim=-1).mean()
        
        # Prediction error
        pred_loss = F.mse_loss(predictions, targets)
        
        # Variance regularization (prevent collapse)
        var_loss = F.relu(0.1 - predictions.var(dim=-1)).mean()
        
        return pred_loss + self.variance_weight * var_loss


class ReasoningLoss(nn.Module):
    """Loss for reasoning module."""
    
    def forward(self, reasoning_output):
        """
        Encourage consistent reasoning.
        
        - Proof consistency
        - Rule coherence
        """
        # Encourage confident predictions
        entropy = -torch.sum(
            reasoning_output * torch.log(reasoning_output + 1e-10),
            dim=-1
        ).mean()
        
        # Low entropy = confident = good
        return entropy * 0.1


class EnergyLoss(nn.Module):
    """Loss for energy module."""
    
    def __init__(self, target_iterations: int = 5):
        super().__init__()
        self.target = target_iterations
    
    def forward(self, energy_info):
        """
        Encourage appropriate iteration count.
        """
        if isinstance(energy_info, dict):
            iterations = energy_info.get('iterations', self.target)
            final_energy = energy_info.get('final_energy', 0.0)
            
            # Penalize not converging
            convergence_loss = final_energy
            
            # Soft iteration regularization
            iteration_loss = 0.01 * abs(iterations - self.target)
            
            return convergence_loss + iteration_loss
        
        return torch.tensor(0.0)


class CausalLoss(nn.Module):
    """Loss for causal module."""
    
    def __init__(self, sparsity_weight: float = 0.01, dag_weight: float = 1.0):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.dag_weight = dag_weight
    
    def forward(self, causal_output):
        """
        Encourage valid causal structure.
        
        - DAG constraint
        - Sparsity
        """
        if hasattr(causal_output, 'adjacency'):
            A = causal_output.adjacency
            
            # DAG constraint: tr(e^A) - n = 0
            expm_A = torch.matrix_exp(A * A)
            dag_loss = torch.trace(expm_A) - A.size(0)
            
            # Sparsity
            sparsity_loss = A.abs().sum()
            
            return self.dag_weight * dag_loss + self.sparsity_weight * sparsity_loss
        
        return torch.tensor(0.0)
```

---

## Training Configuration

```python
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # === Model (these are passed to NEXUSConfig) ===
    vocab_size: int = 32000
    d_model: int = 256           # Hidden dimension
    ssm_n_layers: int = 6        # Number of state space layers
    
    # === Optimization ===
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    
    # === Schedule ===
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'
    
    # === Batching ===
    batch_size: int = 32
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1
    
    # === Regularization ===
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # === Loss Weights ===
    lm_loss_weight: float = 1.0
    world_loss_weight: float = 0.1
    reason_loss_weight: float = 0.05
    energy_loss_weight: float = 0.05
    causal_loss_weight: float = 0.05
    
    # === Checkpointing ===
    save_steps: int = 1000
    save_total_limit: int = 5
    output_dir: str = './outputs'
    
    # === Evaluation ===
    eval_steps: int = 500
    eval_batch_size: int = 16
    
    # === Logging ===
    log_steps: int = 10
    wandb_project: Optional[str] = None
    
    # === Hardware ===
    device: str = 'cuda'
    fp16: bool = True
    compile: bool = False
    num_workers: int = 4
```

---

## Training Loop

### NEXUSTrainer

```python
import os
import math
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class NEXUSTrainer:
    """Main trainer for NEXUS models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: NEXUSLoss = None
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or NEXUSLoss()
        
        # Move model to device
        self.model = self.model.to(config.device)
        
        # Compile if requested (PyTorch 2.0+)
        if config.compile:
            self.model = torch.compile(self.model)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None
        
        # State tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self):
        """Create optimizer with weight decay handling."""
        
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        
        def lr_lambda(step):
            # Warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            
            # Decay
            progress = (step - self.config.warmup_steps) / (
                self.config.max_steps - self.config.warmup_steps
            )
            
            if self.config.lr_decay == 'cosine':
                return 0.5 * (1 + math.cos(math.pi * progress))
            elif self.config.lr_decay == 'linear':
                return 1 - progress
            else:
                return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(
        self,
        train_loader,
        val_loader=None,
        resume_from: str = None
    ):
        """Main training loop."""
        
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Training loop
        self.model.train()
        accumulation_counter = 0
        
        progress = tqdm(total=self.config.max_steps, desc="Training")
        progress.update(self.global_step)
        
        train_iter = iter(train_loader)
        
        while self.global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Move to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            # Forward pass
            loss, loss_dict = self._training_step(batch)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
            else:
                (loss / self.config.gradient_accumulation_steps).backward()
            
            accumulation_counter += 1
            
            # Optimizer step
            if accumulation_counter >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                accumulation_counter = 0
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_steps == 0:
                    self._log_metrics(loss_dict)
                
                # Evaluation
                if val_loader and self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate(val_loader)
                    self.model.train()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best')
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
                
                progress.update(1)
                progress.set_postfix(loss=loss.item())
        
        progress.close()
        self.save_checkpoint('final')
    
    def _training_step(self, batch):
        """Single training step."""
        
        with autocast(enabled=self.config.fp16):
            # Forward
            output, info = self.model(
                batch['input_ids'],
                return_all_outputs=True
            )
            
            # Loss
            loss, loss_dict = self.loss_fn(
                output,
                batch['labels'],
                module_outputs=info.get('module_outputs')
            )
        
        return loss, loss_dict
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            with autocast(enabled=self.config.fp16):
                output = self.model(batch['input_ids'])
                loss, _ = self.loss_fn(output, batch['labels'])
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _log_metrics(self, loss_dict):
        """Log training metrics."""
        
        metrics = {
            'train/total_loss': sum(loss_dict.values()).item(),
            'train/lr': self.scheduler.get_last_lr()[0],
            'train/step': self.global_step
        }
        
        for name, loss in loss_dict.items():
            metrics[f'train/{name}_loss'] = loss.item()
        
        # Print to console
        print(f"Step {self.global_step}: loss={metrics['train/total_loss']:.4f}")
        
        # Log to wandb if configured
        if self.config.wandb_project:
            import wandb
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, name):
        """Save training checkpoint."""
        
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, 'model.pt')
        )
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        print(f"Saved checkpoint: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir):
        """Load training checkpoint."""
        
        # Load model
        model_path = os.path.join(checkpoint_dir, 'model.pt')
        self.model.load_state_dict(torch.load(model_path))
        
        # Load training state
        state_path = os.path.join(checkpoint_dir, 'training_state.pt')
        state = torch.load(state_path)
        
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.global_step = state['global_step']
        self.best_val_loss = state['best_val_loss']
        
        print(f"Loaded checkpoint from step {self.global_step}")
```

---

## Advanced Training Techniques

### Curriculum Learning

```python
class CurriculumScheduler:
    """Gradually increase sequence length during training."""
    
    def __init__(
        self,
        start_length: int = 256,
        end_length: int = 2048,
        warmup_steps: int = 10000
    ):
        self.start_length = start_length
        self.end_length = end_length
        self.warmup_steps = warmup_steps
    
    def get_seq_length(self, step: int) -> int:
        if step >= self.warmup_steps:
            return self.end_length
        
        progress = step / self.warmup_steps
        length = self.start_length + progress * (self.end_length - self.start_length)
        
        # Round to nearest 64 for efficiency
        return int(length // 64) * 64
```

### Dynamic Loss Weighting

```python
class DynamicLossWeights:
    """Automatically balance loss weights based on magnitudes."""
    
    def __init__(self, num_losses: int, momentum: float = 0.99):
        self.weights = torch.ones(num_losses)
        self.running_avg = torch.zeros(num_losses)
        self.momentum = momentum
    
    def update(self, losses: torch.Tensor):
        """Update weights based on loss magnitudes."""
        # Update running average
        self.running_avg = self.momentum * self.running_avg + (1 - self.momentum) * losses
        
        # Inverse weighting: higher loss → lower weight
        self.weights = 1 / (self.running_avg + 1e-8)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def __call__(self, losses: torch.Tensor) -> torch.Tensor:
        self.update(losses.detach())
        return (losses * self.weights).sum()
```

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedNEXUS(NEXUSCore):
    """NEXUS with gradient checkpointing for memory efficiency."""
    
    def forward(self, input_ids, return_all_outputs=False):
        # Checkpoint each layer
        x = self.embedding(input_ids)
        
        for layer in self.state_space.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        
        # Continue without checkpointing for smaller modules
        # ... rest of forward pass
```

---

## Monitoring and Debugging

### Training Metrics to Track

```python
metrics_to_track = [
    # Loss
    'train/total_loss',
    'train/lm_loss',
    'train/world_loss',
    'train/reason_loss',
    'train/energy_loss',
    'train/causal_loss',
    'val/loss',
    
    # Learning dynamics
    'train/learning_rate',
    'train/grad_norm',
    
    # Module-specific
    'energy/avg_iterations',
    'causal/dag_constraint',
    'reasoning/proof_length',
    
    # System
    'system/gpu_memory',
    'system/throughput_tokens_per_sec'
]
```

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Loss explosion | NaN or very large loss | Lower learning rate, enable gradient clipping |
| Slow convergence | Loss plateaus early | Increase learning rate warmup, check data |
| Memory OOM | CUDA out of memory | Reduce batch size, enable checkpointing |
| Module collapse | One loss dominates | Adjust loss weights, check gradients |

---

## Example Training Script

```python
#!/usr/bin/env python
"""Train NEXUS model."""

import torch
from nexus.core import NEXUSCore
from nexus.training import NEXUSTrainer, TrainingConfig, NEXUSDataset

def main():
    # Configuration
    train_config = TrainingConfig(
        vocab_size=32000,
        d_model=256,
        ssm_n_layers=6,
        learning_rate=1e-4,
        batch_size=32,
        max_steps=100000,
        output_dir='./outputs/nexus_run1'
    )
    
    # Model configuration
    from nexus.core.nexus_core import NEXUSConfig
    model_config = NEXUSConfig(
        vocab_size=train_config.vocab_size,
        d_model=train_config.d_model,
        ssm_n_layers=train_config.ssm_n_layers
    )
    model = NEXUSCore(model_config)
    
    # Data
    train_dataset = NEXUSDataset('data/train.bin', max_length=2048)
    val_dataset = NEXUSDataset('data/valid.bin', max_length=2048)
    
    train_loader = create_dataloader(train_dataset, config.batch_size)
    val_loader = create_dataloader(val_dataset, config.eval_batch_size, shuffle=False)
    
    # Train
    trainer = NEXUSTrainer(model, config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
```

---

## Next Steps

- [Evaluation Guide](evaluation.md) - How to evaluate trained models
- [Optimization Guide](optimization.md) - Performance optimization
- [API Reference](../api/training.md) - Training API documentation

---

*Training is teaching the model to think. Do it well.*
