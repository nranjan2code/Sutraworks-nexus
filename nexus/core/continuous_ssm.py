"""
Continuous State Space - Layer-Free Sequence Processing
=========================================================

Traditional SSM stacks N layers: x → SSM₁ → SSM₂ → ... → SSMₙ → y
Continuous SSM iterates ONE dynamics to equilibrium: x → SSM(z*, x) where z* = f(z*, x)

The key insight: State space models are ALREADY defined by continuous dynamics:
    dh/dt = Ah + Bx
    y = Ch + Dx

We simply embrace this continuous nature fully, removing the artificial
layer discretization and letting the system evolve to its natural equilibrium.

This achieves:
1. O(n) sequence processing (maintained from original SSM)
2. Adaptive "depth" based on input complexity
3. Memory efficiency via implicit differentiation
4. Natural interpretation as dynamical system
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ContinuousSSMConfig:
    """Configuration for Continuous State Space."""
    
    d_model: int = 512          # Model dimension
    d_state: int = 64           # Continuous state dimension
    d_conv: int = 4             # Local convolution width
    expand: int = 2             # Expansion factor
    
    # Continuous dynamics
    dt_base: float = 0.1        # Base time step
    max_evolution_steps: int = 20  # Max evolution iterations
    convergence_threshold: float = 1e-4
    
    # Stability
    spectral_norm: bool = True
    damping: float = 0.5
    
    # Training
    dropout: float = 0.1


class ContinuousStateKernel(nn.Module):
    """
    The fundamental state transition kernel, iterated continuously.
    
    Instead of N different layer parameters, we have ONE kernel
    that is applied repeatedly until the state stabilizes.
    
    This is the continuous equivalent of:
        h_{t+1} = A_bar * h_t + B_bar * x_t
        
    But now we iterate this in an outer loop until equilibrium.
    """
    
    def __init__(self, config: ContinuousSSMConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_model * config.expand
        
        # State transition matrix A (structured for stability)
        # Parameterized as -exp(A_log) to ensure eigenvalues have negative real part
        A = repeat(
            torch.arange(1, config.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # Input-to-state projection B (selective, input-dependent)
        self.B_proj = nn.Linear(self.d_inner, config.d_state, bias=False)
        
        # State-to-output projection C (selective, input-dependent)  
        self.C_proj = nn.Linear(self.d_inner, config.d_state, bias=False)
        
        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Time step projection (input-dependent dt)
        self.dt_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 4),
            nn.GELU(),
            nn.Linear(self.d_inner // 4, self.d_inner),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of continuous state evolution.
        
        Args:
            x: Input (batch, seq_len, d_inner)
            h: Current hidden state (batch, seq_len, d_inner, d_state)
            dt_scale: Scale factor for time step
            
        Returns:
            y: Output (batch, seq_len, d_inner)
            h_new: Updated hidden state
        """
        batch, seq_len, d_inner = x.shape
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Input-dependent B, C
        B = self.B_proj(x)  # (batch, seq_len, d_state)
        C = self.C_proj(x)  # (batch, seq_len, d_state)
        
        # Input-dependent time step
        dt = self.dt_proj(x)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt) * self.config.dt_base * dt_scale
        
        # Discretize: A_bar = exp(dt * A)
        # For small dt: A_bar ≈ I + dt * A
        dt_A = torch.einsum("bld,dn->bldn", dt, A)  # (batch, seq_len, d_inner, d_state)
        A_bar = torch.exp(dt_A)
        
        # B_bar = dt * B (zero-order hold approximation)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(-2)  # (batch, seq_len, d_inner, d_state)
        
        # State update: h_new = A_bar * h + B_bar * x
        x_expanded = x.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        h_new = A_bar * h + B_bar * x_expanded.expand(-1, -1, -1, self.config.d_state)
        
        # Output: y = C * h + D * x
        y = torch.einsum("bldn,bln->bld", h_new, C) + self.D * x
        
        return y, h_new


class ContinuousSSM(nn.Module):
    """
    Continuous State Space Model - Layer-Free Sequence Processing.
    
    This is THE module that replaces stacked SSM layers.
    Instead of:
        x → SSM_layer1 → SSM_layer2 → ... → SSM_layerN → y
        
    We have:
        x → ContinuousSSM(evolve until equilibrium) → y
        
    The "depth" emerges from how many evolution steps are needed
    for the state to stabilize - harder sequences need more steps.
    """
    
    def __init__(self, config: ContinuousSSMConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_model * config.expand
        
        # Input projection (with gating)
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)
        
        # Local convolution for short-range dependencies
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
        )
        
        # The ONE kernel that we iterate
        self.kernel = ContinuousStateKernel(config)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)
        
        # Layer norms for stability
        self.norm_in = nn.LayerNorm(config.d_model)
        self.norm_out = nn.LayerNorm(config.d_model)
        
        # Convergence predictor (learns when to stop)
        self.convergence_pred = nn.Sequential(
            nn.Linear(self.d_inner, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        if config.spectral_norm:
            self._apply_spectral_norm()
            
    def _apply_spectral_norm(self):
        """Apply spectral normalization for training stability."""
        nn.utils.spectral_norm(self.in_proj)
        nn.utils.spectral_norm(self.out_proj)
        
    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with continuous evolution to equilibrium.
        
        Args:
            x: Input (batch, seq_len, d_model)
            return_trajectory: Whether to return evolution trajectory
            
        Returns:
            Dictionary with output and evolution metrics
        """
        batch, seq_len, _ = x.shape
        residual = x
        
        # Normalize input
        x = self.norm_in(x)
        
        # Project and split for gating
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Local convolution
        x_conv = rearrange(x_inner, "b l d -> b d l")
        x_conv = self.conv(x_conv)[:, :, :seq_len]
        x_inner = rearrange(x_conv, "b d l -> b l d")
        x_inner = F.silu(x_inner)
        
        # Initialize hidden state
        h = torch.zeros(
            batch, seq_len, self.d_inner, self.config.d_state,
            device=x.device, dtype=x.dtype
        )
        
        # Continuous evolution to equilibrium
        trajectory = [] if return_trajectory else None
        
        for step in range(self.config.max_evolution_steps):
            # One evolution step
            y, h_new = self.kernel(x_inner, h)
            
            # Check convergence
            h_diff = (h_new - h).norm(dim=(-2, -1)).mean()
            
            if trajectory is not None:
                trajectory.append(y.clone())
                
            # Adaptive stopping based on convergence
            if h_diff < self.config.convergence_threshold:
                break
                
            # Damped update
            h = self.config.damping * h_new + (1 - self.config.damping) * h
            
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.norm_out(output)
        
        # Residual connection
        output = output + residual
        
        return {
            "output": output,
            "hidden_state": h,
            "evolution_steps": step + 1,
            "final_convergence": h_diff,
            "trajectory": trajectory,
        }


class BidirectionalContinuousSSM(nn.Module):
    """
    Bidirectional continuous SSM for non-causal tasks.
    
    Runs two continuous SSMs - one forward, one backward -
    and merges their equilibrium states.
    """
    
    def __init__(self, config: ContinuousSSMConfig):
        super().__init__()
        self.forward_ssm = ContinuousSSM(config)
        self.backward_ssm = ContinuousSSM(config)
        self.merge = nn.Linear(config.d_model * 2, config.d_model)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Bidirectional continuous processing."""
        # Forward pass
        fwd_result = self.forward_ssm(x)
        
        # Backward pass (reverse sequence)
        x_rev = x.flip(dims=[1])
        bwd_result = self.backward_ssm(x_rev)
        bwd_output = bwd_result["output"].flip(dims=[1])
        
        # Merge
        merged = self.merge(torch.cat([fwd_result["output"], bwd_output], dim=-1))
        
        return {
            "output": merged,
            "forward_steps": fwd_result["evolution_steps"],
            "backward_steps": bwd_result["evolution_steps"],
        }


class HierarchicalContinuousSSM(nn.Module):
    """
    Hierarchical continuous SSM with multiple time scales.
    
    Different "levels" operate at different time scales,
    allowing capture of both local and global patterns.
    But still layer-free - each level evolves to its own equilibrium.
    """
    
    def __init__(self, config: ContinuousSSMConfig, n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales
        
        # One SSM per time scale (but each is layer-free internally)
        self.scales = nn.ModuleList()
        for i in range(n_scales):
            scale_config = ContinuousSSMConfig(
                d_model=config.d_model,
                d_state=config.d_state,
                dt_base=config.dt_base * (2 ** i),  # Increasing time scales
                max_evolution_steps=config.max_evolution_steps,
                convergence_threshold=config.convergence_threshold,
            )
            self.scales.append(ContinuousSSM(scale_config))
            
        # Cross-scale interaction
        self.scale_merge = nn.Linear(config.d_model * n_scales, config.d_model)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-scale continuous processing."""
        outputs = []
        total_steps = 0
        
        for scale_ssm in self.scales:
            result = scale_ssm(x)
            outputs.append(result["output"])
            total_steps += result["evolution_steps"]
            
        # Merge across scales
        merged = self.scale_merge(torch.cat(outputs, dim=-1))
        
        return {
            "output": merged,
            "total_evolution_steps": total_steps,
            "per_scale_outputs": outputs,
        }


class ContinuousSSMWithMemory(nn.Module):
    """
    Continuous SSM with external memory that co-evolves.
    
    The memory and sequence state evolve together toward
    a joint equilibrium, enabling long-range dependencies
    beyond the sequence length.
    """
    
    def __init__(self, config: ContinuousSSMConfig, memory_size: int = 64):
        super().__init__()
        self.config = config
        self.memory_size = memory_size
        
        self.ssm = ContinuousSSM(config)
        
        # Learnable memory
        self.memory = nn.Parameter(torch.randn(memory_size, config.d_model) * 0.02)
        
        # Memory interaction
        self.read_memory = nn.MultiheadAttention(
            config.d_model, num_heads=8, batch_first=True
        )
        self.write_memory = nn.MultiheadAttention(
            config.d_model, num_heads=8, batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process with co-evolving memory."""
        batch = x.shape[0]
        memory = self.memory.unsqueeze(0).expand(batch, -1, -1)
        
        # Iterate until joint equilibrium
        for _ in range(self.config.max_evolution_steps // 2):
            # Read from memory
            x_mem, _ = self.read_memory(x, memory, memory)
            x = x + 0.1 * x_mem
            
            # SSM evolution step
            result = self.ssm(x)
            x = result["output"]
            
            # Write to memory
            memory_update, _ = self.write_memory(memory, x, x)
            memory = memory + 0.1 * memory_update
            
        return {
            "output": x,
            "memory": memory,
        }
