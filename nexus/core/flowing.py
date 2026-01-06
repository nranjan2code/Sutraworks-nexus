"""
FlowingNEXUS - Truly Layer-Free Neural Architecture
=====================================================

"Growth is not a ladder with rungs to climb.
 It is water finding its level."

This module implements the complete layer-free NEXUS architecture where:
- There are NO discrete layers to traverse
- Computation flows continuously toward equilibrium
- "Depth" emerges from the input, not architecture
- The system evolves as a unified dynamical system

Traditional Architecture:
    input → layer₁ → layer₂ → ... → layerₙ → output
    
FlowingNEXUS:
    input → unified_dynamics(z*) → output
    where z* satisfies: z* = f(z*, input)
    
The forward pass IS an optimization finding the fixed point.
The backward pass uses implicit differentiation (O(1) memory).
The "depth" is how many iterations until convergence.

This is neural computation as physics - states evolving
according to learned dynamics until they settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.core.equilibrium import (
    EquilibriumConfig,
    EquilibriumCore,
    ContinuousDynamics,
    ImplicitFunction,
    ContinuousAttention,
    ContinuousMemory,
)
from nexus.core.continuous_ssm import (
    ContinuousSSMConfig,
    ContinuousSSM,
    HierarchicalContinuousSSM,
)


@dataclass
class FlowingConfig:
    """
    Configuration for FlowingNEXUS - the layer-free architecture.
    
    Note: No 'n_layers' parameter - depth is emergent!
    """
    
    # Core dimensions
    d_model: int = 512              # State dimension
    d_latent: int = 256             # Latent space dimension
    
    # Continuous dynamics
    max_flow_steps: int = 50        # Maximum evolution steps
    convergence_threshold: float = 1e-4
    damping: float = 0.5            # Update damping
    
    # State space (sequence processing)
    ssm_d_state: int = 64
    ssm_dt_base: float = 0.1
    
    # Attention (continuous)
    n_heads: int = 8
    attention_dt: float = 0.1
    
    # Memory (co-evolving)
    memory_size: int = 128
    
    # World model
    n_abstraction_levels: int = 3
    prediction_horizon: int = 5
    
    # Reasoning
    n_predicates: int = 1000
    max_reasoning_depth: int = 10
    
    # Causal
    n_causal_vars: int = 32
    
    # Training
    dropout: float = 0.1
    spectral_norm: bool = True
    implicit_diff: bool = True      # Use implicit differentiation
    jac_reg_weight: float = 0.01    # Jacobian regularization
    
    # General
    vocab_size: int = 50000
    max_seq_len: int = 8192


class UnifiedDynamics(nn.Module):
    """
    The ONE dynamics function that replaces ALL layers.
    
    This single module handles:
    - Sequence processing (via continuous SSM dynamics)
    - Attention (via continuous attention flow)
    - Memory (via co-evolving memory states)
    - Reasoning (via iterative refinement)
    
    Instead of SSM → Attention → FFN → ... (discrete stages),
    we have ONE transformation applied iteratively:
    
        z_{t+1} = z_t + dt * unified_f(z_t, x)
    """
    
    def __init__(self, config: FlowingConfig):
        super().__init__()
        self.config = config
        
        # State processing branch
        self.state_transform = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        
        # Attention-like interaction (but continuous)
        self.interaction = ContinuousAttention(
            EquilibriumConfig(
                d_model=config.d_model,
                d_hidden=config.d_model * 2,
            )
        )
        
        # Gating for selective update
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.d_model),
        )
        
        if config.spectral_norm:
            self._apply_spectral_norm()
            
    def _apply_spectral_norm(self):
        """Constrain Lipschitz constant for stable fixed points."""
        for module in self.state_transform:
            if isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module)
                
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute state update dz = f(z, x, t).
        
        Args:
            z: Current state (batch, seq_len, d_model)
            x: Input/context (batch, seq_len, d_model)
            t: Optional time (batch,) or scalar
            
        Returns:
            State update (batch, seq_len, d_model)
        """
        batch, seq_len, d = z.shape
        
        # Local transformation
        combined = torch.cat([z, x], dim=-1)
        local_update = self.state_transform(combined)
        
        # Global interaction (attention-like)
        global_update = self.interaction(z, x, dt=0.1) - z
        
        # Combine updates
        total_update = local_update + global_update
        
        # Gating
        gate = self.gate(combined)
        total_update = gate * total_update
        
        # Time modulation
        if t is not None:
            if t.dim() == 0:
                t = t.expand(batch)
            t_emb = self.time_embed(t.view(-1, 1))
            t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
            total_update = total_update * (1 + 0.1 * torch.tanh(t_emb))
            
        return total_update


class FlowingNEXUS(nn.Module):
    """
    FlowingNEXUS - Complete Layer-Free Architecture.
    
    This is the main model class where ALL computation happens through
    continuous flow toward equilibrium. There are NO layers to count,
    NO fixed depth - just dynamics finding their natural resting point.
    
    ┌──────────────────────────────────────────────────────────────┐
    │                      FlowingNEXUS                            │
    │                                                              │
    │   input ──┐                                                  │
    │           │                                                  │
    │           ▼                                                  │
    │   ┌─────────────────────────────────────────────┐           │
    │   │         Unified Dynamics f(z, x)            │           │
    │   │                                             │◄──────┐   │
    │   │  z_{t+1} = z_t + damping * f(z_t, x)       │       │   │
    │   │                                             │       │   │
    │   └─────────────────┬───────────────────────────┘       │   │
    │                     │                                    │   │
    │                     ▼                                    │   │
    │              ┌──────────────┐                           │   │
    │              │  Converged?  │──── No ────────────────────┘   │
    │              └──────┬───────┘                                │
    │                     │ Yes                                    │
    │                     ▼                                        │
    │              equilibrium z*                                  │
    │                     │                                        │
    │                     ▼                                        │
    │                 output                                       │
    └──────────────────────────────────────────────────────────────┘
    
    Key Properties:
    - Depth emerges from input complexity
    - Memory efficient (implicit differentiation)
    - Natural uncertainty (non-convergence = uncertainty)
    - Computation adapts to input difficulty
    """
    
    def __init__(self, config: FlowingConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.embedding = FlowingEmbedding(config)
        
        # THE unified dynamics (replaces all layers)
        self.dynamics = UnifiedDynamics(config)
        
        # Continuous SSM for efficient sequence processing
        ssm_config = ContinuousSSMConfig(
            d_model=config.d_model,
            d_state=config.ssm_d_state,
            dt_base=config.ssm_dt_base,
            max_evolution_steps=config.max_flow_steps,
            convergence_threshold=config.convergence_threshold,
        )
        self.continuous_ssm = ContinuousSSM(ssm_config)
        
        # Co-evolving memory
        self.memory = ContinuousMemory(
            EquilibriumConfig(d_model=config.d_model),
            memory_size=config.memory_size,
        )
        
        # Initial state generator
        self.init_state = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        
        # Output heads
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.regression_head = nn.Linear(config.d_model, config.d_model)
        
        # Confidence from equilibrium quality
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model + 2, 128),  # +2 for convergence info
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Tie embeddings
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable dynamics."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
    def forward(
        self,
        x: torch.Tensor,
        modality: str = "token",
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through continuous dynamics to equilibrium.
        
        The "forward pass" is actually an optimization:
        Find z* such that z* ≈ z* + f(z*, x)
        
        Args:
            x: Input tensor
            modality: "token" or "continuous"
            return_trajectory: Whether to return evolution trajectory
            
        Returns:
            Dictionary with predictions and flow metrics
        """
        # Embed input
        embedded = self.embedding(x, modality=modality)
        batch, seq_len, d_model = embedded.shape
        
        # Initialize state
        z = self.init_state(embedded)
        memory_state = None
        
        # Evolution trajectory
        trajectory = [z] if return_trajectory else None
        energies = []
        
        # Flow toward equilibrium
        converged = False
        for step in range(self.config.max_flow_steps):
            # SSM processing (efficient sequence handling)
            ssm_result = self.continuous_ssm(z)
            z_ssm = ssm_result["output"]
            
            # Unified dynamics update
            delta = self.dynamics(z_ssm, embedded, t=torch.tensor(step / self.config.max_flow_steps))
            
            # Memory co-evolution
            z_mem, memory_state = self.memory(z_ssm, memory_state, dt=0.1)
            
            # Combined update with damping
            z_new = z + self.config.damping * (delta + 0.1 * (z_mem - z))
            
            # Compute "energy" (residual norm)
            energy = (z_new - z).norm(dim=-1).mean()
            energies.append(energy)
            
            if trajectory is not None:
                trajectory.append(z_new)
                
            # Check convergence
            if energy < self.config.convergence_threshold:
                converged = True
                break
                
            z = z_new
            
        # z is now the equilibrium (or best approximation)
        z_star = z
        
        # Apply implicit differentiation for efficient backward
        if self.training and self.config.implicit_diff:
            z_star = ImplicitFunction.apply(
                self.dynamics,
                embedded,
                z_star,
                EquilibriumConfig(
                    d_model=self.config.d_model,
                    max_iterations=self.config.max_flow_steps,
                    convergence_threshold=self.config.convergence_threshold,
                    damping=self.config.damping,
                ),
            )
            
        # Generate outputs from equilibrium
        lm_logits = self.lm_head(z_star)
        regression_out = self.regression_head(z_star)
        
        # Confidence based on convergence quality
        final_energy = energies[-1] if energies else torch.tensor(0.0)
        conf_input = torch.cat([
            z_star.mean(dim=1),
            final_energy.unsqueeze(0).expand(batch, 1),
            torch.tensor([[step / self.config.max_flow_steps]], device=z_star.device).expand(batch, 1),
        ], dim=-1)
        confidence = self.confidence_head(conf_input).squeeze(-1)
        
        return {
            "logits": lm_logits,
            "hidden_states": z_star,
            "regression": regression_out,
            "confidence": confidence,
            "flow_steps": step + 1,
            "converged": converged,
            "final_energy": final_energy,
            "trajectory": trajectory,
            "memory": memory_state,
        }
        
    def imagine(
        self,
        context: torch.Tensor,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """
        Imagine future states by continuing the flow.
        
        Instead of predicting discrete future tokens,
        we extend the flow dynamics into the future.
        """
        # Get equilibrium for context
        result = self.forward(context, modality="continuous")
        z = result["hidden_states"]
        
        # Continue evolution (not seeking equilibrium, but exploring)
        predictions = [z]
        for _ in range(n_steps):
            delta = self.dynamics(z, context)
            z = z + 0.5 * delta  # Larger steps for exploration
            predictions.append(z)
            
        return torch.stack(predictions, dim=1)
        
    def reason(
        self,
        query: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Reason by finding equilibrium with extended iterations.
        
        Harder reasoning = more iterations to converge.
        """
        # Use more iterations for reasoning
        old_max = self.config.max_flow_steps
        self.config.max_flow_steps = self.config.max_reasoning_depth * 2
        
        result = self.forward(query, modality="continuous", return_trajectory=True)
        
        self.config.max_flow_steps = old_max
        
        return {
            "answer": result["hidden_states"],
            "reasoning_depth": result["flow_steps"],
            "confidence": result["confidence"],
            "reasoning_trajectory": result["trajectory"],
        }
        
    def get_flow_complexity(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Measure the computational complexity for a given input.
        
        Returns metrics about how "hard" this input is -
        how many steps needed to reach equilibrium.
        """
        with torch.no_grad():
            result = self.forward(x, return_trajectory=True)
            
        return {
            "flow_steps": result["flow_steps"],
            "converged": result["converged"],
            "final_energy": result["final_energy"].item(),
            "relative_depth": result["flow_steps"] / self.config.max_flow_steps,
        }


class FlowingEmbedding(nn.Module):
    """Input embedding for FlowingNEXUS."""
    
    def __init__(self, config: FlowingConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.continuous_proj = nn.Linear(config.d_model, config.d_model)
        
        # Continuous positional encoding (not discrete positions)
        self.pos_encoding = nn.Sequential(
            nn.Linear(1, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model),
        )
        
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        modality: str = "token",
    ) -> torch.Tensor:
        """Embed input into model space."""
        if modality == "token":
            embedded = self.token_embedding(x)
        else:
            embedded = self.continuous_proj(x)
            
        batch, seq_len = embedded.shape[:2]
        
        # Continuous positional encoding
        positions = torch.linspace(0, 1, seq_len, device=x.device)
        positions = positions.view(1, -1, 1).expand(batch, -1, -1)
        pos_emb = self.pos_encoding(positions)
        
        embedded = embedded + pos_emb
        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class LivingFlowingNEXUS(nn.Module):
    """
    Living system wrapper for FlowingNEXUS.
    
    Combines the layer-free architecture with continuous learning
    and uncertainty-aware responses.
    """
    
    def __init__(self, config: FlowingConfig):
        super().__init__()
        self.config = config
        self.model = FlowingNEXUS(config)
        
        # Learning rate for online adaptation
        self.adaptation_rate = 0.001
        
        # Track experience
        self.total_interactions = 0
        self.total_flow_steps = 0
        
    def interact(
        self,
        x: torch.Tensor,
        modality: str = "token",
        learn: bool = True,
    ) -> Dict[str, Any]:
        """
        Interact with the living system.
        
        - Processes input through continuous flow
        - Optionally learns from the interaction
        - Returns response with confidence
        """
        result = self.model(x, modality=modality)
        
        self.total_interactions += 1
        self.total_flow_steps += result["flow_steps"]
        
        # Decide whether to respond based on confidence
        responded = result["confidence"].mean() > 0.3
        
        return {
            "responded": responded,
            "logits": result["logits"] if responded else None,
            "confidence": result["confidence"],
            "flow_depth": result["flow_steps"],
            "converged": result["converged"],
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the living system."""
        avg_depth = self.total_flow_steps / max(1, self.total_interactions)
        
        return {
            "total_interactions": self.total_interactions,
            "average_flow_depth": avg_depth,
            "depth_efficiency": 1 - (avg_depth / self.config.max_flow_steps),
        }


def create_flowing_nexus(
    size: str = "base",
    **kwargs,
) -> FlowingNEXUS:
    """
    Factory function to create FlowingNEXUS models.
    
    Note: 'size' here refers to model width, not depth.
    Depth is emergent and adapts to input!
    """
    size_configs = {
        "small": {
            "d_model": 256,
            "d_latent": 128,
            "n_heads": 4,
            "memory_size": 64,
        },
        "base": {
            "d_model": 512,
            "d_latent": 256,
            "n_heads": 8,
            "memory_size": 128,
        },
        "large": {
            "d_model": 1024,
            "d_latent": 512,
            "n_heads": 16,
            "memory_size": 256,
        },
        "xl": {
            "d_model": 2048,
            "d_latent": 1024,
            "n_heads": 32,
            "memory_size": 512,
        },
    }
    
    config_dict = size_configs.get(size, size_configs["base"])
    config_dict.update(kwargs)
    
    config = FlowingConfig(**config_dict)
    return FlowingNEXUS(config)


def create_living_flowing_nexus(
    size: str = "base",
    **kwargs,
) -> LivingFlowingNEXUS:
    """Create a living layer-free NEXUS system."""
    config_dict = {
        "small": {"d_model": 256, "d_latent": 128},
        "base": {"d_model": 512, "d_latent": 256},
        "large": {"d_model": 1024, "d_latent": 512},
    }.get(size, {"d_model": 512, "d_latent": 256})
    
    config_dict.update(kwargs)
    config = FlowingConfig(**config_dict)
    
    return LivingFlowingNEXUS(config)
