"""
Continual / Online Learning Loop for NEXUS.

This wraps a ``NEXUSCore`` so it can keep serving answers while performing
small, guarded online updates. It:
- ingests new samples as they arrive
- mixes replayed samples to avoid catastrophic forgetting
- limits work per cycle to bound cost and drift
- exposes the same model for answering during continual learning

Quickstart
----------
```
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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW

from nexus.core.nexus_core import NEXUSCore
from nexus.training.trainer import TrainingConfig
from nexus.training.losses import NEXUSLoss


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
