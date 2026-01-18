"""
Continual / Online Learning Loop for NEXUS.

This wraps a ``NEXUSCore`` or ``FlowingNEXUS`` so it can keep serving answers
while performing small, guarded online updates. It:
- ingests new samples as they arrive
- mixes replayed samples to avoid catastrophic forgetting
- limits work per cycle to bound cost and drift
- exposes the same model for answering during continual learning

Quickstart - Traditional Architecture
-------------------------------------
```python
from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
from nexus.training import TrainingConfig, ContinualConfig, ContinualLearner

model = NEXUSCore(NEXUSConfig())
train_cfg = TrainingConfig()
cont_cfg = ContinualConfig(buffer_size=2048, replay_ratio=0.5)
learner = ContinualLearner(model, train_cfg, cont_cfg)

# Answer
answers = learner.respond(batch)

# Learn from new data while continuing to answer
metrics = learner.observe_and_learn([batch])
```

Quickstart - Flowing Architecture
---------------------------------
```python
from nexus.core import create_flowing_nexus
from nexus.training import TrainingConfig, FlowingContinualConfig, FlowingContinualLearner

model = create_flowing_nexus(size="base")
train_cfg = TrainingConfig()
cont_cfg = FlowingContinualConfig(buffer_size=2048, replay_ratio=0.5)
learner = FlowingContinualLearner(model, train_cfg, cont_cfg)

# Answer with confidence tracking
answers = learner.respond(batch)
print(f"Confidence: {answers['confidence'].mean():.3f}")
print(f"Flow depth: {answers['flow_steps']}")

# Learn from new data
metrics = learner.observe_and_learn([batch])
print(f"Average flow depth: {metrics.get('avg_flow_steps', 0):.1f}")
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from nexus.core.nexus_core import NEXUSCore
from nexus.training.trainer import TrainingConfig
from nexus.training.losses import NEXUSLoss

if TYPE_CHECKING:
    from nexus.core.flowing import FlowingNEXUS, FlowingConfig


@dataclass
class ContinualConfig:
    """Settings for continual / online learning."""

    buffer_size: int = 2048  # max samples kept in replay buffer
    replay_ratio: float = 0.5  # fraction of each microbatch drawn from buffer
    microbatch_size: int = 4
    max_updates_per_cycle: int = 2  # guardrail: limit work per observe() call
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    log_every: int = 10
    # Optimizer overrides (defaults from TrainingConfig if None)
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    eps: Optional[float] = None


@dataclass
class FlowingContinualConfig:
    """
    Settings for continual learning with FlowingNEXUS.
    
    Extends ContinualConfig with flow-specific parameters for
    adaptive computation and convergence-aware learning.
    """
    
    # Buffer settings
    buffer_size: int = 2048
    replay_ratio: float = 0.5
    microbatch_size: int = 4
    max_updates_per_cycle: int = 2
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Logging
    log_every: int = 10
    
    # Optimizer (defaults from TrainingConfig if None)
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    eps: Optional[float] = None
    
    # Flow-specific settings
    convergence_bonus_weight: float = 0.1  # Reward for faster convergence
    min_confidence_for_replay: float = 0.3  # Only replay confident samples
    adaptive_lr_by_flow_depth: bool = True  # Scale LR by flow complexity
    max_flow_steps_for_learning: int = 30  # Limit flow steps during learning
    
    # Jacobian regularization (for stable dynamics)
    jac_reg_weight: float = 0.01
    
    # Curriculum learning by complexity
    curriculum_enabled: bool = True
    curriculum_warmup_steps: int = 1000


class ContinualLearner:
    """Self-evolving wrapper that learns while serving answers."""

    def __init__(
        self,
        model: NEXUSCore,
        train_config: TrainingConfig,
        continual_config: ContinualConfig,
    ) -> None:
        self.model = model.to(continual_config.device)
        self.train_config = train_config
        self.config = continual_config

        self.loss_fn = NEXUSLoss(
            lm_weight=train_config.lm_loss_weight,
            world_model_weight=train_config.world_model_loss_weight,
            reasoning_weight=train_config.reasoning_loss_weight,
            energy_weight=train_config.energy_loss_weight,
            causal_weight=train_config.causal_loss_weight,
        )

        lr = continual_config.learning_rate or train_config.learning_rate
        wd = continual_config.weight_decay or train_config.weight_decay
        b1 = continual_config.beta1 or train_config.beta1
        b2 = continual_config.beta2 or train_config.beta2
        eps = continual_config.eps or train_config.eps

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(b1, b2),
            eps=eps,
            weight_decay=wd,
        )
        self.scaler = (
            torch.cuda.amp.GradScaler() if continual_config.mixed_precision else None
        )

        self.replay_buffer: List[Dict[str, torch.Tensor]] = []
        self.update_step = 0

    @torch.no_grad()
    def respond(self, batch: Dict[str, torch.Tensor], modality: str = "token") -> Dict[str, torch.Tensor]:
        """Serve answers/inference without blocking learning."""
        self.model.eval()
        batch = self._move_to_device(batch)
        input_tensor = batch.get("input_ids")
        if input_tensor is None:
            input_tensor = batch.get("features")
        outputs = self.model(
            input_tensor,
            modality=modality,
            context_mask=batch.get("context_mask"),
            target_mask=batch.get("target_mask"),
            return_all=True,
        )
        return outputs

    def observe_and_learn(self, new_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Ingest new samples, mix with replay, and apply a few safe updates."""
        # Store new samples on CPU to keep GPU memory bounded
        for sample in new_samples:
            self._add_to_buffer(sample)

        metrics: Dict[str, float] = {}
        max_updates = min(self.config.max_updates_per_cycle, len(new_samples))
        if max_updates == 0:
            return metrics

        self.model.train()
        for _ in range(max_updates):
            batch = self._sample_microbatch()
            if batch is None:
                break

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(
                    batch["input_ids"],
                    modality="token",
                    context_mask=batch.get("context_mask"),
                    target_mask=batch.get("target_mask"),
                    return_all=True,
                )
                targets = {"labels": batch.get("labels", batch["input_ids"])}
                loss, loss_dict = self.loss_fn(outputs, targets)

            self._backward_step(loss)
            self.update_step += 1

            if self.update_step % self.config.log_every == 0:
                metrics = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
                metrics["total_loss"] = loss_dict.get("total_loss", loss).item()

        return metrics

    def _backward_step(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _add_to_buffer(self, sample: Dict[str, torch.Tensor]) -> None:
        """Store a sample on CPU, trimming oldest if buffer exceeds capacity."""
        cpu_sample: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                cpu_sample[k] = v.detach().cpu()
            else:
                cpu_sample[k] = v
        self.replay_buffer.append(cpu_sample)
        if len(self.replay_buffer) > self.config.buffer_size:
            overflow = len(self.replay_buffer) - self.config.buffer_size
            del self.replay_buffer[:overflow]

    def _sample_microbatch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Create a microbatch mixing recent samples with replayed history."""
        if not self.replay_buffer:
            return None

        batch_size = self.config.microbatch_size
        replay_count = int(batch_size * self.config.replay_ratio)
        new_count = batch_size - replay_count

        # Always sample newest first for new data
        new_samples = self.replay_buffer[-new_count:] if new_count > 0 else []

        # Replay: random older samples
        if replay_count > 0 and len(self.replay_buffer) > new_count:
            replay_candidates = self.replay_buffer[:-new_count] if new_count > 0 else self.replay_buffer
            replay_indices = torch.randperm(len(replay_candidates))[:replay_count]
            replay_samples = [replay_candidates[i] for i in replay_indices]
        else:
            replay_samples = []

        batch_samples = replay_samples + new_samples
        if not batch_samples:
            return None

        collated: Dict[str, List[torch.Tensor]] = {}
        for sample in batch_samples:
            for k, v in sample.items():
                if torch.is_tensor(v):
                    collated.setdefault(k, []).append(v)

        batch: Dict[str, torch.Tensor] = {}
        for k, tensors in collated.items():
            batch[k] = torch.stack(tensors, dim=0).to(self.config.device)
        return batch

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to the configured device."""
        return {k: v.to(self.config.device) if torch.is_tensor(v) else v for k, v in batch.items()}


class FlowingContinualLearner:
    """
    Continual learning wrapper for FlowingNEXUS (layer-free architecture).
    
    This learner is specifically designed for the equilibrium-based architecture:
    - Tracks flow depth (emergent computation) during learning
    - Uses convergence quality as a learning signal
    - Implements curriculum learning based on input complexity
    - Applies Jacobian regularization for stable dynamics
    
    The key insight: in FlowingNEXUS, "harder" inputs require more flow steps.
    We can use this as a natural curriculum signal.
    """
    
    def __init__(
        self,
        model: "FlowingNEXUS",
        train_config: TrainingConfig,
        continual_config: FlowingContinualConfig,
    ) -> None:
        from nexus.core.flowing import FlowingNEXUS
        
        if not isinstance(model, FlowingNEXUS):
            raise TypeError(
                f"FlowingContinualLearner requires FlowingNEXUS model, "
                f"got {type(model).__name__}. Use ContinualLearner for NEXUSCore."
            )
        
        self.model = model.to(continual_config.device)
        self.train_config = train_config
        self.config = continual_config
        
        # Flow-specific loss function
        self.loss_fn = FlowingLoss(
            lm_weight=train_config.lm_loss_weight,
            convergence_bonus_weight=continual_config.convergence_bonus_weight,
            jac_reg_weight=continual_config.jac_reg_weight,
        )
        
        # Optimizer setup
        lr = continual_config.learning_rate or train_config.learning_rate
        wd = continual_config.weight_decay or train_config.weight_decay
        b1 = continual_config.beta1 or train_config.beta1
        b2 = continual_config.beta2 or train_config.beta2
        eps = continual_config.eps or train_config.eps
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(b1, b2),
            eps=eps,
            weight_decay=wd,
        )
        
        # Cosine annealing with warm restarts for continual learning
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,  # Restart every 100 steps
            T_mult=2,  # Double period after each restart
            eta_min=lr * 0.01,
        )
        
        self.scaler = (
            torch.cuda.amp.GradScaler() if continual_config.mixed_precision else None
        )
        
        # Replay buffer with confidence metadata
        self.replay_buffer: List[Dict[str, torch.Tensor]] = []
        self.confidence_scores: List[float] = []
        self.flow_depths: List[int] = []
        
        # Statistics
        self.update_step = 0
        self.total_flow_steps = 0
        self.total_samples = 0
    
    @torch.no_grad()
    def respond(
        self,
        batch: Dict[str, torch.Tensor],
        modality: str = "token",
    ) -> Dict[str, torch.Tensor]:
        """
        Serve answers with confidence and flow depth tracking.
        
        Returns:
            Dictionary containing:
            - logits: Model predictions
            - confidence: Per-sample confidence scores
            - flow_steps: Number of flow iterations (emergent depth)
            - converged: Whether equilibrium was reached
        """
        self.model.eval()
        batch = self._move_to_device(batch)
        
        input_tensor = batch.get("input_ids")
        if input_tensor is None:
            input_tensor = batch.get("features")
            modality = "continuous"
        
        outputs = self.model(input_tensor, modality=modality)
        
        # Track statistics
        self.total_samples += input_tensor.shape[0]
        self.total_flow_steps += outputs["flow_steps"] * input_tensor.shape[0]
        
        return outputs
    
    def observe_and_learn(
        self,
        new_samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Learn from new samples with flow-aware training.
        
        Key differences from standard ContinualLearner:
        1. Tracks flow depth for curriculum learning
        2. Filters replay by confidence threshold
        3. Applies adaptive learning rate based on complexity
        4. Regularizes Jacobian for stable dynamics
        """
        # First, assess complexity of new samples
        for sample in new_samples:
            self._assess_and_add_to_buffer(sample)
        
        metrics: Dict[str, float] = {}
        max_updates = min(self.config.max_updates_per_cycle, len(new_samples))
        if max_updates == 0:
            return metrics
        
        self.model.train()
        
        # Temporarily limit flow steps during training for efficiency
        original_max_steps = self.model.config.max_flow_steps
        self.model.config.max_flow_steps = min(
            original_max_steps,
            self.config.max_flow_steps_for_learning,
        )
        
        total_loss = 0.0
        total_flow_steps = 0
        
        try:
            for _ in range(max_updates):
                batch = self._sample_microbatch_by_complexity()
                if batch is None:
                    break
                
                # Adaptive learning rate based on batch complexity
                if self.config.adaptive_lr_by_flow_depth:
                    self._adjust_lr_for_complexity(batch)
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(batch["input_ids"], modality="token")
                    targets = {"labels": batch.get("labels", batch["input_ids"])}
                    loss, loss_dict = self.loss_fn(outputs, targets, self.model)
                
                self._backward_step(loss)
                self.scheduler.step()
                self.update_step += 1
                
                total_loss += loss.item()
                total_flow_steps += outputs["flow_steps"]
                
                if self.update_step % self.config.log_every == 0:
                    metrics = {
                        k: v.item() if isinstance(v, torch.Tensor) else float(v)
                        for k, v in loss_dict.items()
                    }
                    metrics["total_loss"] = total_loss / max_updates
                    metrics["avg_flow_steps"] = total_flow_steps / max_updates
                    metrics["buffer_size"] = len(self.replay_buffer)
        finally:
            # Restore original max steps
            self.model.config.max_flow_steps = original_max_steps
        
        return metrics
    
    def _assess_and_add_to_buffer(self, sample: Dict[str, torch.Tensor]) -> None:
        """Assess sample complexity and add to buffer with metadata."""
        cpu_sample: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                cpu_sample[k] = v.detach().cpu()
            else:
                cpu_sample[k] = v
        
        # Quick forward pass to assess complexity (no grad)
        with torch.no_grad():
            self.model.eval()
            input_t = cpu_sample.get("input_ids", cpu_sample.get("features"))
            if input_t is not None:
                input_t = input_t.unsqueeze(0).to(self.config.device)
                modality = "token" if "input_ids" in cpu_sample else "continuous"
                result = self.model(input_t, modality=modality)
                
                confidence = result["confidence"].mean().item()
                flow_depth = result["flow_steps"]
            else:
                confidence = 0.5
                flow_depth = 25  # Default middle value
        
        self.replay_buffer.append(cpu_sample)
        self.confidence_scores.append(confidence)
        self.flow_depths.append(flow_depth)
        
        # Trim buffer if needed
        if len(self.replay_buffer) > self.config.buffer_size:
            # Remove lowest confidence samples first
            if self.config.min_confidence_for_replay > 0:
                # Find indices to remove
                indices_to_remove = [
                    i for i, conf in enumerate(self.confidence_scores)
                    if conf < self.config.min_confidence_for_replay
                ][:len(self.replay_buffer) - self.config.buffer_size]
                
                if indices_to_remove:
                    for idx in sorted(indices_to_remove, reverse=True):
                        del self.replay_buffer[idx]
                        del self.confidence_scores[idx]
                        del self.flow_depths[idx]
            
            # If still over capacity, remove oldest
            while len(self.replay_buffer) > self.config.buffer_size:
                self.replay_buffer.pop(0)
                self.confidence_scores.pop(0)
                self.flow_depths.pop(0)
    
    def _sample_microbatch_by_complexity(self) -> Optional[Dict[str, torch.Tensor]]:
        """Sample microbatch with curriculum learning based on flow depth."""
        if not self.replay_buffer:
            return None
        
        batch_size = self.config.microbatch_size
        
        if self.config.curriculum_enabled and self.update_step < self.config.curriculum_warmup_steps:
            # During warmup: prefer simpler samples (lower flow depth)
            progress = self.update_step / self.config.curriculum_warmup_steps
            
            # Sort by flow depth and sample from easier portion
            sorted_indices = sorted(
                range(len(self.flow_depths)),
                key=lambda i: self.flow_depths[i]
            )
            
            # Sample from the easier portion based on progress
            max_idx = int(len(sorted_indices) * (0.3 + 0.7 * progress))
            max_idx = max(batch_size, max_idx)
            candidate_indices = sorted_indices[:max_idx]
            
            selected_indices = torch.randperm(len(candidate_indices))[:batch_size].tolist()
            selected = [candidate_indices[i] for i in selected_indices]
        else:
            # After warmup: uniform sampling with confidence filter
            valid_indices = [
                i for i, conf in enumerate(self.confidence_scores)
                if conf >= self.config.min_confidence_for_replay
            ]
            
            if not valid_indices:
                valid_indices = list(range(len(self.replay_buffer)))
            
            selected_indices = torch.randperm(len(valid_indices))[:batch_size].tolist()
            selected = [valid_indices[i] for i in selected_indices]
        
        batch_samples = [self.replay_buffer[i] for i in selected]
        
        if not batch_samples:
            return None
        
        collated: Dict[str, List[torch.Tensor]] = {}
        for sample in batch_samples:
            for k, v in sample.items():
                if torch.is_tensor(v):
                    collated.setdefault(k, []).append(v)
        
        batch: Dict[str, torch.Tensor] = {}
        for k, tensors in collated.items():
            batch[k] = torch.stack(tensors, dim=0).to(self.config.device)
        
        return batch
    
    def _adjust_lr_for_complexity(self, batch: Dict[str, torch.Tensor]) -> None:
        """Adjust learning rate based on batch complexity estimate."""
        # Use a simple heuristic: shorter sequences are simpler
        seq_len = batch["input_ids"].shape[1]
        max_seq = self.model.config.max_seq_len
        
        complexity_factor = seq_len / max_seq
        # Scale LR down for complex samples (avoid large updates)
        lr_scale = 1.0 - 0.5 * complexity_factor
        
        base_lr = self.config.learning_rate or self.train_config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = base_lr * lr_scale
    
    def _backward_step(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient scaling."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to the configured device."""
        return {
            k: v.to(self.config.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get learning statistics."""
        avg_flow = self.total_flow_steps / max(1, self.total_samples)
        avg_confidence = sum(self.confidence_scores) / max(1, len(self.confidence_scores))
        
        return {
            "total_updates": self.update_step,
            "total_samples": self.total_samples,
            "avg_flow_depth": avg_flow,
            "avg_confidence": avg_confidence,
            "buffer_size": len(self.replay_buffer),
            "buffer_avg_flow": sum(self.flow_depths) / max(1, len(self.flow_depths)),
        }


class FlowingLoss(nn.Module):
    """
    Loss function for FlowingNEXUS with convergence-aware terms.
    
    Components:
    1. Language modeling loss (standard cross-entropy)
    2. Convergence bonus (reward for faster equilibrium)
    3. Jacobian regularization (stable dynamics)
    """
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        convergence_bonus_weight: float = 0.1,
        jac_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.lm_weight = lm_weight
        self.convergence_bonus_weight = convergence_bonus_weight
        self.jac_reg_weight = jac_reg_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: Optional["FlowingNEXUS"] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute flow-aware loss.
        
        Args:
            outputs: Model outputs (logits, flow_steps, converged, etc.)
            targets: Target labels
            model: FlowingNEXUS model for Jacobian regularization
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        device = outputs["logits"].device
        total_loss = torch.tensor(0.0, device=device)
        
        # Language modeling loss
        if "logits" in outputs and "labels" in targets:
            logits = outputs["logits"]
            labels = targets["labels"]
            
            lm_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss_dict["lm_loss"] = lm_loss
            total_loss = total_loss + self.lm_weight * lm_loss
        
        # Convergence bonus: reward faster convergence
        if "flow_steps" in outputs and "converged" in outputs:
            max_steps = 50  # Default, could be passed
            flow_steps = outputs["flow_steps"]
            converged = outputs["converged"]
            
            # Normalized depth (0 = instant, 1 = max steps)
            normalized_depth = flow_steps / max_steps
            
            # Bonus for convergence + early stopping
            if converged:
                convergence_bonus = 1.0 - normalized_depth  # Higher bonus for faster
            else:
                convergence_bonus = -0.1  # Small penalty for non-convergence
            
            convergence_loss = -convergence_bonus  # Negative because we minimize
            loss_dict["convergence_loss"] = torch.tensor(convergence_loss, device=device)
            total_loss = total_loss + self.convergence_bonus_weight * convergence_loss
        
        # Jacobian regularization (if model provided)
        if model is not None and self.jac_reg_weight > 0 and "hidden_states" in outputs:
            # Only compute occasionally (expensive)
            if torch.rand(1).item() < 0.1:  # 10% of batches
                try:
                    # Estimate spectral norm of dynamics Jacobian
                    z = outputs["hidden_states"]
                    x = targets.get("input_features", z)  # Use hidden states if no input features
                    jac_penalty = self._estimate_jacobian_penalty(model, z, x)
                    loss_dict["jac_reg"] = jac_penalty
                    total_loss = total_loss + self.jac_reg_weight * jac_penalty
                except Exception:
                    pass  # Skip if estimation fails
        
        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict
    
    def _estimate_jacobian_penalty(
        self,
        model: "FlowingNEXUS",
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate penalty for Jacobian spectral norm > 1.
        
        Uses power iteration for efficient estimation.
        """
        z = z.detach().requires_grad_(True)
        
        # Compute dynamics output
        delta = model.dynamics(z, x)
        
        # Power iteration to estimate spectral norm
        v = torch.randn_like(z)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        
        for _ in range(3):  # Few iterations
            Jv = torch.autograd.grad(
                delta, z, v,
                retain_graph=True,
                create_graph=True,
            )[0]
            v = Jv / (Jv.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Final estimate
        Jv = torch.autograd.grad(
            delta, z, v,
            retain_graph=False,
            create_graph=True,
        )[0]
        
        spectral_norm = Jv.norm(dim=-1).mean()
        
        # Penalize if > 0.99 (want contraction for stable fixed point)
        penalty = torch.nn.functional.relu(spectral_norm - 0.99)
        
        return penalty
