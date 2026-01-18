"""
NEXUS Training Pipeline
========================

Comprehensive training framework for the NEXUS architecture.

Key features:
- Multi-objective training (prediction + reasoning + energy + causal)
- Distributed training support
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpointing and resumption
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from nexus.core.nexus_core import NEXUSCore, NEXUSConfig


@dataclass
class TrainingConfig:
    """Configuration for NEXUS training."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training schedule
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Loss weights
    lm_loss_weight: float = 1.0
    world_model_loss_weight: float = 0.5
    reasoning_loss_weight: float = 0.3
    energy_loss_weight: float = 0.2
    causal_loss_weight: float = 0.2
    
    # EMA settings for world model target encoder
    ema_enabled: bool = True
    ema_decay: float = 0.996
    ema_update_every_n_steps: int = 1
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4
    
    # Logging
    log_every_n_steps: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


class NEXUSLoss(nn.Module):
    """
    Multi-objective loss function for NEXUS training.
    
    Combines:
    - Language modeling loss (next token prediction)
    - World model loss (JEPA-style representation prediction)
    - Reasoning loss (logical consistency)
    - Energy loss (contrastive learning)
    - Causal loss (structure learning)
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Language modeling loss
        if "logits" in outputs and "labels" in targets:
            lm_loss = F.cross_entropy(
                outputs["logits"].view(-1, outputs["logits"].size(-1)),
                targets["labels"].view(-1),
                ignore_index=-100,
            )
            loss_dict["lm_loss"] = lm_loss
            total_loss = total_loss + self.config.lm_loss_weight * lm_loss
            
        # World model loss (if world model outputs available)
        if "world_model" in outputs and "predicted" in outputs["world_model"]:
            wm_output = outputs["world_model"]
            wm_loss = F.smooth_l1_loss(
                wm_output["predicted"],
                wm_output["target"],
                reduction="none",
            )
            # Apply target mask
            if "target_mask" in wm_output:
                mask = wm_output["target_mask"].unsqueeze(-1).float()
                wm_loss = (wm_loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                wm_loss = wm_loss.mean()
                
            loss_dict["world_model_loss"] = wm_loss
            total_loss = total_loss + self.config.world_model_loss_weight * wm_loss
            
        # Energy loss (contrastive)
        if "energy_output" in outputs and "contrastive_loss" in outputs["energy_output"]:
            energy_loss = outputs["energy_output"]["contrastive_loss"]
            loss_dict["energy_loss"] = energy_loss
            total_loss = total_loss + self.config.energy_loss_weight * energy_loss
            
        # Causal loss (structure regularization)
        if "causal_output" in outputs:
            causal_out = outputs["causal_output"]
            causal_loss = (
                causal_out.get("acyclicity_loss", 0.0) +
                0.01 * causal_out.get("sparsity_loss", 0.0)
            )
            if isinstance(causal_loss, torch.Tensor):
                loss_dict["causal_loss"] = causal_loss
                total_loss = total_loss + self.config.causal_loss_weight * causal_loss
                
        # Reasoning loss (if supervised reasoning targets available)
        if "reasoning" in outputs and "reasoning_target" in targets:
            reasoning_loss = F.mse_loss(
                outputs["reasoning"]["answer"],
                targets["reasoning_target"],
            )
            loss_dict["reasoning_loss"] = reasoning_loss
            total_loss = total_loss + self.config.reasoning_loss_weight * reasoning_loss
            
        loss_dict["total_loss"] = total_loss
        
        return total_loss, loss_dict


class NEXUSTrainer:
    """
    Training orchestrator for NEXUS.
    
    Handles:
    - Training loop with gradient accumulation
    - Evaluation and metrics
    - Checkpointing
    - Logging
    - Learning rate scheduling
    - EMA updates for world model target encoder
    """
    
    def __init__(
        self,
        model: NEXUSCore,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = NEXUSLoss(config)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        
        # EMA tracking
        self._ema_enabled = config.ema_enabled and self._has_world_model()
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def _has_world_model(self) -> bool:
        """Check if model has a world model with target encoder."""
        return (
            hasattr(self.model, "world_model") and
            self.model.world_model is not None and
            hasattr(self.model.world_model, "update_target_encoder")
        )
    
    def _update_ema(self) -> None:
        """Update EMA for world model target encoder."""
        if not self._ema_enabled:
            return
        
        if self.global_step % self.config.ema_update_every_n_steps == 0:
            self.model.world_model.update_target_encoder()
        
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay handling."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
        
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        
        # Cosine decay
        total_steps = self.config.max_steps or (
            len(self.train_dataloader) * self.config.num_epochs
        )
        decay_steps = total_steps - self.config.warmup_steps
        
        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=decay_steps,
            eta_min=self.config.learning_rate * 0.01,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.config.warmup_steps],
        )
        
    def train(self) -> Dict[str, float]:
        """
        Run training loop.
        
        Returns:
            Final training metrics
        """
        self.model.train()
        
        total_steps = self.config.max_steps or (
            len(self.train_dataloader) * self.config.num_epochs
        )
        
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        running_loss = 0.0
        step_losses = {}
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(
                        batch["input_ids"],
                        modality="token",
                        return_all=True,
                    )
                    
                    targets = {"labels": batch.get("labels", batch["input_ids"])}
                    loss, loss_dict = self.loss_fn(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Accumulate losses for logging
                running_loss += loss.item() * self.config.gradient_accumulation_steps
                for k, v in loss_dict.items():
                    if k not in step_losses:
                        step_losses[k] = 0.0
                    step_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
                    
                # Gradient accumulation step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # EMA update for world model target encoder
                    self._update_ema()
                    
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        avg_loss = running_loss / self.config.log_every_n_steps
                        avg_losses = {
                            k: v / self.config.log_every_n_steps 
                            for k, v in step_losses.items()
                        }
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        })
                        
                        running_loss = 0.0
                        step_losses = {}
                        
                    # Evaluation
                    if (
                        self.eval_dataloader is not None and
                        self.global_step % self.config.eval_every_n_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        self.model.train()
                        
                        if eval_metrics["total_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["total_loss"]
                            self.save_checkpoint("best")
                            
                    # Checkpointing
                    if self.global_step % self.config.save_every_n_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                        
                    # Check max steps
                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        break
                        
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
                
        progress_bar.close()
        
        # Save final checkpoint
        self.save_checkpoint("final")
        
        return {"final_loss": running_loss}
        
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation loop."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_losses = {}
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            batch = self._move_to_device(batch)
            
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(
                    batch["input_ids"],
                    modality="token",
                    return_all=True,
                )
                
                targets = {"labels": batch.get("labels", batch["input_ids"])}
                loss, loss_dict = self.loss_fn(outputs, targets)
                
            batch_size = batch["input_ids"].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for k, v in loss_dict.items():
                if k not in all_losses:
                    all_losses[k] = 0.0
                val = v.item() if isinstance(v, torch.Tensor) else v
                all_losses[k] += val * batch_size
                
        # Average losses
        avg_loss = total_loss / total_samples
        avg_losses = {k: v / total_samples for k, v in all_losses.items()}
        avg_losses["total_loss"] = avg_loss
        
        return avg_losses
        
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}.pt"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
