"""
NEXUS Loss Functions
=====================

Multi-objective loss functions for training NEXUS components.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NEXUSLoss(nn.Module):
    """
    Combined loss for NEXUS multi-objective training.
    
    Re-exported from trainer for convenience.
    """
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        world_model_weight: float = 0.5,
        reasoning_weight: float = 0.3,
        energy_weight: float = 0.2,
        causal_weight: float = 0.2,
    ):
        super().__init__()
        self.lm_weight = lm_weight
        self.world_model_weight = world_model_weight
        self.reasoning_weight = reasoning_weight
        self.energy_weight = energy_weight
        self.causal_weight = causal_weight
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss."""
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=outputs.get("logits", outputs.get("hidden_states")).device)
        
        # Language modeling loss
        if "logits" in outputs and "labels" in targets:
            lm_loss = F.cross_entropy(
                outputs["logits"].view(-1, outputs["logits"].size(-1)),
                targets["labels"].view(-1),
                ignore_index=-100,
            )
            loss_dict["lm_loss"] = lm_loss
            total_loss = total_loss + self.lm_weight * lm_loss
            
        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict


class JEPALoss(nn.Module):
    """
    JEPA-style representation prediction loss.
    
    Predicts target representations from context representations,
    learning abstract world models without pixel/token reconstruction.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute JEPA loss.
        
        Args:
            predicted: Predicted representations (batch, seq_len, d_model)
            target: Target representations (batch, seq_len, d_model)
            mask: Optional mask for target positions
            
        Returns:
            Loss value
        """
        # Normalize
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target, dim=-1)
        
        # Smooth L1 loss
        loss = F.smooth_l1_loss(predicted, target, reduction="none")
        
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
            
        return loss


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for representation learning.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor: Anchor representations (batch, d_model)
            positive: Positive representations (batch, d_model)
            negatives: Negative representations (batch, n_neg, d_model)
        """
        batch_size = anchor.shape[0]
        
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        
        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = torch.bmm(
                negatives, anchor.unsqueeze(-1)
            ).squeeze(-1) / self.temperature
            
            # Combine
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # In-batch negatives
            sim_matrix = torch.mm(anchor, positive.t()) / self.temperature
            labels = torch.arange(batch_size, device=anchor.device)
            loss = F.cross_entropy(sim_matrix, labels)
            
        return loss


class CausalLoss(nn.Module):
    """
    Loss for causal structure learning.
    
    Combines:
    - DAG constraint (acyclicity)
    - Sparsity regularization
    - Reconstruction loss
    """
    
    def __init__(
        self,
        acyclicity_weight: float = 1.0,
        sparsity_weight: float = 0.1,
    ):
        super().__init__()
        self.acyclicity_weight = acyclicity_weight
        self.sparsity_weight = sparsity_weight
        
    def forward(
        self,
        adjacency: torch.Tensor,
        reconstructed: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute causal structure loss.
        
        Args:
            adjacency: Adjacency matrix (n_vars, n_vars)
            reconstructed: Reconstructed values
            target: Target values
        """
        loss_dict = {}
        
        # Acyclicity constraint: h(A) = tr(e^A) - d = 0
        n = adjacency.shape[0]
        exp_adj = torch.matrix_exp(adjacency * adjacency)
        h = torch.trace(exp_adj) - n
        acyclicity_loss = h ** 2
        loss_dict["acyclicity"] = acyclicity_loss
        
        # Sparsity
        sparsity_loss = adjacency.abs().sum()
        loss_dict["sparsity"] = sparsity_loss
        
        total = (
            self.acyclicity_weight * acyclicity_loss +
            self.sparsity_weight * sparsity_loss
        )
        
        # Reconstruction
        if reconstructed is not None and target is not None:
            recon_loss = F.mse_loss(reconstructed, target)
            loss_dict["reconstruction"] = recon_loss
            total = total + recon_loss
            
        loss_dict["total"] = total
        return total, loss_dict
