"""
Adaptive Energy-Based Computation Module
==========================================

Implements energy-based models for adaptive computation and principled
uncertainty quantification.

Key innovations:
- Dynamic computation depth based on input complexity
- Energy landscape for representing belief states
- Contrastive learning with proper negative sampling
- Calibrated confidence estimation

Unlike fixed-depth networks, this module allocates more computation
to harder examples and provides meaningful uncertainty estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass  
class EnergyConfig:
    """Configuration for Adaptive Energy Module."""
    
    d_model: int = 256           # Model dimension
    d_energy: int = 128          # Energy function hidden dimension
    max_iterations: int = 10     # Maximum refinement iterations
    energy_threshold: float = 0.1  # Convergence threshold
    n_negatives: int = 16        # Number of negative samples
    temperature: float = 0.07    # Contrastive temperature
    step_size: float = 0.1       # Gradient descent step size
    noise_scale: float = 0.01    # Langevin dynamics noise
    dropout: float = 0.1


class EnergyFunction(nn.Module):
    """
    Neural energy function E(x, y) for scoring state compatibility.
    
    Lower energy = more compatible/likely state.
    Used for:
    - Representation refinement
    - Uncertainty estimation  
    - Contrastive learning
    """
    
    def __init__(self, config: EnergyConfig):
        super().__init__()
        self.config = config
        
        # Joint encoding of input and state
        self.joint_encoder = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_energy * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_energy * 2, config.d_energy),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Energy output
        self.energy_head = nn.Sequential(
            nn.Linear(config.d_energy, config.d_energy // 2),
            nn.GELU(),
            nn.Linear(config.d_energy // 2, 1),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy E(x, y).
        
        Args:
            x: Input representation (batch, d_model)
            y: State/output representation (batch, d_model)
            
        Returns:
            Energy values (batch,) - lower is more compatible
        """
        # Joint encoding
        joint = torch.cat([x, y], dim=-1)
        hidden = self.joint_encoder(joint)
        
        # Compute energy
        energy = self.energy_head(hidden).squeeze(-1)
        
        return energy


class IterativeRefinement(nn.Module):
    """
    Iterative refinement via energy-based gradient descent.
    
    Refines representations by descending the energy landscape,
    allocating more computation to harder/ambiguous inputs.
    """
    
    def __init__(self, config: EnergyConfig):
        super().__init__()
        self.config = config
        self.energy_fn = EnergyFunction(config)
        
        # Initial state generator
        self.init_state = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        
        # Refinement network (predicts update direction)
        self.refine_net = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        n_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Iteratively refine state given input.
        
        Args:
            x: Input representation (batch, seq_len, d_model) or (batch, d_model)
            n_iterations: Number of refinement steps (None = adaptive)
            
        Returns:
            refined: Refined representation
            info: Dictionary with energies and iteration counts
        """
        original_shape = x.shape
        is_3d = x.dim() == 3
        
        # Handle sequence input
        if is_3d:
            batch, seq_len, d_model = x.shape
            x_flat = x.view(batch * seq_len, d_model)
            refined, info = self._refine(x_flat, n_iterations)
            refined = refined.view(batch, seq_len, d_model)
            # Reshape energies back to (batch,) by averaging over seq_len
            info["final_energy"] = info["final_energy"].view(batch, seq_len).mean(dim=1)
            info["iterations"] = info["iterations"].view(batch, seq_len).mean(dim=1)
            return refined, info
        else:
            return self._refine(x, n_iterations)
            
    def _refine(
        self,
        x: torch.Tensor,
        n_iterations: Optional[int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Core refinement loop."""
        batch = x.shape[0]
        device = x.device
        
        # Initialize state
        y = self.init_state(x)
        
        # Track energies and iterations
        energies = []
        
        max_iter = n_iterations or self.config.max_iterations
        actual_iterations = torch.zeros(batch, device=device)
        converged = torch.zeros(batch, dtype=torch.bool, device=device)
        
        for i in range(max_iter):
            # Compute current energy
            energy = self.energy_fn(x, y)
            energies.append(energy)
            
            # Check convergence
            if i > 0:
                energy_change = torch.abs(energies[-1] - energies[-2])
                newly_converged = energy_change < self.config.energy_threshold
                converged = converged | newly_converged
                
            # Update iteration count for non-converged samples
            actual_iterations = actual_iterations + (~converged).float()
            
            # Early exit if all converged
            if converged.all():
                break
                
            # Compute update direction
            combined = torch.cat([x, y], dim=-1)
            update = self.refine_net(combined)
            
            # Apply update with step size (only to non-converged)
            step = self.config.step_size * (~converged).float().unsqueeze(-1)
            y = y - step * update
            
            # Optional: Add Langevin dynamics noise for exploration
            if self.training:
                noise = torch.randn_like(y) * self.config.noise_scale
                y = y + noise * (~converged).float().unsqueeze(-1)
                
        info = {
            "energies": torch.stack(energies, dim=1),
            "final_energy": energies[-1],
            "iterations": actual_iterations,
            "converged": converged,
        }
        
        return y, info


class ContrastiveEnergyLearning(nn.Module):
    """
    Contrastive learning with energy-based negative sampling.
    
    Trains the energy function by:
    - Lowering energy for positive (compatible) pairs
    - Raising energy for negative (incompatible) pairs
    """
    
    def __init__(self, config: EnergyConfig):
        super().__init__()
        self.config = config
        self.energy_fn = EnergyFunction(config)
        
        # Negative sample generator
        self.neg_generator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model * config.n_negatives),
        )
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive energy loss.
        
        Args:
            anchor: Anchor representations (batch, d_model)
            positive: Positive (compatible) representations (batch, d_model)
            negatives: Optional negative samples (batch, n_neg, d_model)
            
        Returns:
            Dictionary with loss and metrics
        """
        batch = anchor.shape[0]
        
        # Generate negatives if not provided
        if negatives is None:
            neg_flat = self.neg_generator(anchor)
            negatives = neg_flat.view(batch, self.config.n_negatives, -1)
            
        # Compute energies
        pos_energy = self.energy_fn(anchor, positive)  # (batch,)
        
        # Negative energies
        neg_energies = []
        for i in range(negatives.shape[1]):
            neg_e = self.energy_fn(anchor, negatives[:, i])
            neg_energies.append(neg_e)
        neg_energy = torch.stack(neg_energies, dim=1)  # (batch, n_neg)
        
        # InfoNCE-style contrastive loss
        # Treat as softmax over positive + negatives
        all_energies = torch.cat([
            pos_energy.unsqueeze(1),
            neg_energy
        ], dim=1)  # (batch, 1 + n_neg)
        
        # Lower energy = higher probability
        logits = -all_energies / self.config.temperature
        labels = torch.zeros(batch, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy (positive should have lowest energy)
        predictions = all_energies.argmin(dim=1)
        accuracy = (predictions == 0).float().mean()
        
        return {
            "loss": loss,
            "pos_energy": pos_energy.mean(),
            "neg_energy": neg_energy.mean(),
            "accuracy": accuracy,
        }


class AdaptiveEnergyModule(nn.Module):
    """
    Adaptive Energy-Based Computation Module - Core NEXUS component.
    
    Provides:
    1. Adaptive computation depth based on input complexity
    2. Energy-based uncertainty quantification
    3. Contrastive representation learning
    4. Principled confidence estimation
    
    Key advantages over fixed-depth networks:
    - More computation for harder examples
    - Meaningful uncertainty estimates
    - Energy landscape interpretation
    - Proper negative sampling
    """
    
    def __init__(self, config: EnergyConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.energy_fn = EnergyFunction(config)
        self.refinement = IterativeRefinement(config)
        self.contrastive = ContrastiveEnergyLearning(config)
        
        # Input projection
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        
        # Complexity estimator (predicts required iterations)
        self.complexity_estimator = nn.Sequential(
            nn.Linear(config.d_model, config.d_energy),
            nn.GELU(),
            nn.Linear(config.d_energy, config.max_iterations),
            nn.Softmax(dim=-1),
        )
        
        # Confidence head (energy-based uncertainty)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_energy),
            nn.GELU(),
            nn.Linear(config.d_energy, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive computation.
        
        Args:
            x: Input tensor (batch, seq_len, d_model) or (batch, d_model)
            target: Optional target for contrastive learning
            
        Returns:
            Dictionary with refined output, energy, confidence, and metrics
        """
        # Project input
        x = self.input_proj(x)
        
        # Estimate complexity for adaptive computation
        if x.dim() == 3:
            complexity = self.complexity_estimator(x.mean(dim=1))
        else:
            complexity = self.complexity_estimator(x)
            
        expected_iterations = torch.sum(
            complexity * torch.arange(
                self.config.max_iterations, 
                device=x.device,
                dtype=x.dtype
            ),
            dim=-1
        )
        
        # Iterative refinement
        refined, refine_info = self.refinement(x)
        
        # Compute confidence from energy
        final_energy = refine_info["final_energy"]
        
        if refined.dim() == 3:
            refined_mean = refined.mean(dim=1)
        else:
            refined_mean = refined
            
        conf_input = torch.cat([
            refined_mean,
            final_energy.unsqueeze(-1)
        ], dim=-1)
        confidence = self.confidence_head(conf_input).squeeze(-1)
        
        result = {
            "output": refined,
            "energy": final_energy,
            "confidence": confidence,
            "iterations": refine_info["iterations"],
            "expected_iterations": expected_iterations,
            "complexity": complexity,
        }
        
        # Contrastive loss if target provided
        if target is not None:
            if target.dim() == 3:
                target_flat = target.view(-1, target.shape[-1])
                refined_flat = refined.view(-1, refined.shape[-1])
            else:
                target_flat = target
                refined_flat = refined
                
            contrast_result = self.contrastive(
                refined_flat,
                target_flat,
            )
            result["contrastive_loss"] = contrast_result["loss"]
            result["contrastive_accuracy"] = contrast_result["accuracy"]
            
        return result
        
    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty via energy landscape sampling.
        
        Samples multiple refinement trajectories to estimate
        epistemic uncertainty.
        """
        samples = []
        energies = []
        
        for _ in range(n_samples):
            refined, info = self.refinement(x)
            samples.append(refined)
            energies.append(info["final_energy"])
            
        samples = torch.stack(samples, dim=0)
        energies = torch.stack(energies, dim=0)
        
        # Compute statistics
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        energy_mean = energies.mean(dim=0)
        energy_std = energies.std(dim=0)
        
        return {
            "mean": mean,
            "std": std,
            "samples": samples,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
        }


class EnergyBasedPlanningModule(nn.Module):
    """
    Energy-based planning for goal-directed behavior.
    
    Uses energy function to evaluate and rank action sequences,
    enabling planning without explicit policy learning.
    """
    
    def __init__(self, config: EnergyConfig):
        super().__init__()
        self.config = config
        self.energy_fn = EnergyFunction(config)
        
        # Action proposal network
        self.action_proposer = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model * 8),
        )
        
        # Action evaluator
        self.action_energy = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_energy),
            nn.GELU(),
            nn.Linear(config.d_energy, 1),
        )
        
    def plan(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        n_actions: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Plan actions to reach goal state.
        
        Args:
            current_state: Current state (batch, d_model)
            goal_state: Goal state (batch, d_model)
            n_actions: Number of action candidates
            
        Returns:
            Best action and planning metrics
        """
        batch = current_state.shape[0]
        
        # Propose candidate actions
        context = torch.cat([current_state, goal_state], dim=-1)
        actions_flat = self.action_proposer(context)
        actions = actions_flat.view(batch, n_actions, -1)
        
        # Evaluate each action
        energies = []
        for i in range(n_actions):
            action = actions[:, i]
            action_context = torch.cat([
                current_state,
                goal_state,
                action
            ], dim=-1)
            energy = self.action_energy(action_context).squeeze(-1)
            energies.append(energy)
            
        energies = torch.stack(energies, dim=1)
        
        # Select best action (lowest energy)
        best_idx = energies.argmin(dim=1)
        best_action = actions[torch.arange(batch), best_idx]
        best_energy = energies[torch.arange(batch), best_idx]
        
        return {
            "best_action": best_action,
            "best_energy": best_energy,
            "all_actions": actions,
            "all_energies": energies,
        }
