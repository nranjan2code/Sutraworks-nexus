"""
Selective State Space Module
=============================

Implements O(n) linear-time sequence modeling inspired by Mamba/S4 architectures.
Unlike Transformers' O(n²) attention, this achieves linear scaling while maintaining
the ability to selectively focus on relevant information.

Key innovations:
- Selective state space mechanism with input-dependent parameters
- Hardware-aware parallel scan algorithm
- Continuous-time, recurrent, and convolutional unified framework
- Recurrent mode for O(1) step inference
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class StateSpaceConfig:
    """Configuration for Selective State Space layers."""

    d_model: int = 256  # Model dimension
    d_state: int = 16  # SSM state expansion factor
    d_conv: int = 4  # Local convolution width
    expand: int = 2  # Block expansion factor
    dt_rank: str | int = "auto"  # Rank of Δ projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True  # Use parallel scan


class SelectiveStateSpace(nn.Module):
    """
    Selective State Space Module - Core building block of NEXUS.

    This module implements a selective state space model that processes sequences
    in O(n) time while maintaining the ability to selectively retain or discard
    information based on content.

    Mathematical formulation:
        h'(t) = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)

    Where A, B, C are input-dependent (selective), enabling content-aware filtering.
    """

    def __init__(self, config: StateSpaceConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)

        # Compute dt_rank
        self.dt_rank = math.ceil(self.d_model / 16) if config.dt_rank == "auto" else config.dt_rank

        # Input projection: project to 2 * d_inner (for gating)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)

        # Convolution layer for local context
        # In recurrent mode, we need to cache the conv state
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=config.conv_bias,
        )

        # SSM parameters projection
        # Projects input to dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt bias for stable training
        self._init_dt_proj()

        # A parameter (structured for stability)
        # A is parameterized as -exp(A_log) to ensure negativity
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def _init_dt_proj(self):
        """Initialize dt projection for stable dynamics."""
        config = self.config

        # Initialize dt_proj weight
        dt_init_std = self.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unknown dt_init: {config.dt_init}")

        # Initialize dt_proj bias to ensure dt is in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)

        # Inverse of softplus: log(exp(x) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through selective state space.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cache: (ssm_state, conv_state)
                   ssm_state: (batch, d_inner, d_state)
                   conv_state: (batch, d_inner, d_conv)

        Returns:
            y: Output tensor of shape (batch, seq_len, d_model)
            new_cache: Updated (ssm_state, conv_state)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split for gating
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Convolution with cache handling
        if cache is not None and seq_len == 1:
            # Step mode
            ssm_state, conv_state = cache

            # Update conv state (shift and append)
            # x is (B, 1, D) -> (B, D)
            x_step = x.squeeze(1)

            # conv_state is (B, D, K)
            # Shift: remove oldest, append new
            new_conv_state = torch.cat([conv_state[:, :, 1:], x_step.unsqueeze(-1)], dim=-1)

            # Compute convolution
            # Linear convolution matches Conv1d with padding
            # We treat conv_state as the window. Kernel is (D, 1, K) due to groups=D
            # This is effectively a dot product per channel

            # Manually apply convolution
            # weights: (D, 1, K) -> (D, K)
            weights = self.conv1d.weight.squeeze(1)
            # bias: (D,)
            bias = self.conv1d.bias

            # (B, D, K) * (D, K) -> sum over K -> (B, D)
            x_conv = torch.sum(new_conv_state * weights, dim=-1) + bias
            x_conv = x_conv.unsqueeze(1)  # (B, 1, D)

            # Update cache tuple
            conv_cache_next = new_conv_state

        else:
            # Sequence mode
            x_t = rearrange(x, "b l d -> b d l")

            # Create padding for causal conv if this is start of sequence
            if seq_len > self.d_conv:
                # Standard convolution
                x_conv = self.conv1d(x_t)[:, :, :seq_len]
            else:
                # Short sequence (e.g. init), just use standard conv with padding
                x_conv = self.conv1d(x_t)[:, :, :seq_len]

            x_conv = rearrange(x_conv, "b d l -> b l d")

            # Initialize cache if needed (zeros)
            # conv_state should hold last d_conv inputs
            # For simplicity in this implementation, we just grab the last d_conv elements
            if seq_len >= self.d_conv:
                conv_cache_next = x_t[:, :, -self.d_conv :]
            else:
                # Pad if too short
                pad = torch.zeros(batch, self.d_inner, self.d_conv - seq_len, device=x.device)
                conv_cache_next = torch.cat([pad, x_t], dim=-1)

        # Apply SiLU activation
        x_conv = F.silu(x_conv)

        # SSM computation
        if cache is not None and seq_len == 1:
            y, ssm_state_next = self._ssm_step(x_conv, ssm_state)
        else:
            y, ssm_state_next = self._ssm_forward(x_conv)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output, (ssm_state_next, conv_cache_next)

    def _ssm_step(self, x: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step SSM update.

        Args:
            x: Input (B, 1, D)
            h_prev: Previous state (B, D, N)

        Returns:
            y: Output (B, 1, D)
            h_next: Next state (B, D, N)
        """
        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # (D, N)
        D = self.D.float()  # (D,)

        # Project x to get dt, B, C
        x_dbl = self.x_proj(x)  # (B, 1, dt_rank + 2N)

        # Split projections
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Project dt
        dt = self.dt_proj(dt)  # (B, 1, D)
        dt = F.softplus(dt)

        # Squeeze sequence dim for simplified math
        # x: (B, D)
        x = x.squeeze(1)
        # dt: (B, D)
        dt = dt.squeeze(1)
        # B, C: (B, 1, N) -> (B, N)
        B = B.squeeze(1)
        C = C.squeeze(1)

        # Discretize
        # A_bar = exp(dt * A) -> (B, D, N)
        # dt: (B, D), A: (D, N) -> broadcast
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0)  # (B, D, N)
        A_bar = torch.exp(dt_A)

        # B_bar = dt * B
        # dt: (B, D), B: (B, N) -> (B, D, N)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(1)

        # State update: h = A_bar * h + B_bar * x
        # h: (B, D, N)
        # x: (B, D) -> (B, D, 1) to broadcast
        h_next = A_bar * h_prev + B_bar * x.unsqueeze(-1)

        # Output: y = C * h
        # C: (B, N) -> (B, 1, N) -> (B, D, N) broadcast?
        # No, standard SSM is contraction: sum_n (h_dn * C_bn) ?
        # Original Mamba paper: y = sum(h * C) ?
        # Actually usually it is channel-wise independent until mixing?
        # Typically C is (B, N) and we want y of shape (B, D).
        # Mamba uses specific parameterization.
        # We'll assume C is shared or projected such that: y_d = h_d^T C
        # Current implementation assumes C is (B, L, N).

        # Let's align with _parallel_scan logic:
        # y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
        y = torch.einsum("bdn,bn->bd", h_next, C)

        # Add D skip
        y = y + D * x

        return y.unsqueeze(1), h_next

    def _ssm_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core SSM computation with selective parameters.
        Returns y and final state.
        """
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        # Get A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()

        # Project x
        x_dbl = self.x_proj(x)

        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        if self.config.pscan:
            y, h_final = self._parallel_scan(x, dt, A, B, C, D)
        else:
            y, h_final = self._sequential_scan(x, dt, A, B, C, D)

        return y, h_final

    def _parallel_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        True parallel scan implementation using associative scan algorithm.
        
        This achieves O(n) work with O(log n) depth, enabling efficient
        parallelization on GPU. The key insight is that SSM recurrence
        can be expressed as an associative binary operation.
        
        For SSM: h_t = A_t * h_{t-1} + B_t * x_t
        
        We define associative operation ⊕ on pairs (A, Bu):
            (A2, Bu2) ⊕ (A1, Bu1) = (A2 * A1, A2 * Bu1 + Bu2)
        
        This allows parallel prefix sum computation.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        device = x.device
        dtype = x.dtype

        # Discretize
        dt_A = torch.einsum("bld,dn->bldn", dt, A)
        A_bar = torch.exp(dt_A)  # (batch, seq_len, d_inner, d_state)

        dt_B = dt.unsqueeze(-1) * B.unsqueeze(-2)  # (batch, seq_len, d_inner, d_state)
        B_bar = dt_B

        x_reshaped = x.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        Bu = B_bar * x_reshaped  # (batch, seq_len, d_inner, d_state)

        # Parallel associative scan
        # We compute the cumulative products and sums in log(n) parallel steps
        h_all = self._associative_scan(A_bar, Bu, device, dtype)

        # Compute outputs
        # y_t = sum_n(h_t[d,n] * C_t[b,n]) for each d
        y = torch.einsum("bldn,bln->bld", h_all, C)

        # Add skip connection
        y = y + D * x

        # Return final hidden state
        h_final = h_all[:, -1]  # (batch, d_inner, d_state)

        return y, h_final

    def _associative_scan(
        self,
        A_bar: torch.Tensor,
        Bu: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute parallel prefix scan using the associative property.
        
        The associative operation is:
            (A2, Bu2) ⊕ (A1, Bu1) = (A2 * A1, A2 * Bu1 + Bu2)
        
        This computes h[t] = A_bar[t] * h[t-1] + Bu[t] for all t in parallel.
        
        Uses Blelloch's parallel scan algorithm:
        1. Up-sweep (reduce) phase: compute partial products
        2. Down-sweep phase: compute final values
        
        Args:
            A_bar: Discretized A matrix (batch, seq_len, d_inner, d_state)
            Bu: B * x products (batch, seq_len, d_inner, d_state)
            
        Returns:
            h_all: Hidden states for all timesteps (batch, seq_len, d_inner, d_state)
        """
        batch, seq_len, d_inner, d_state = A_bar.shape
        
        # Pad to power of 2 for clean parallel scan
        log2_len = max(1, (seq_len - 1).bit_length())
        padded_len = 1 << log2_len
        
        if padded_len > seq_len:
            # Pad with identity elements: A=1, Bu=0
            pad_size = padded_len - seq_len
            A_pad = torch.ones(batch, pad_size, d_inner, d_state, device=device, dtype=dtype)
            Bu_pad = torch.zeros(batch, pad_size, d_inner, d_state, device=device, dtype=dtype)
            A_bar = torch.cat([A_bar, A_pad], dim=1)
            Bu = torch.cat([Bu, Bu_pad], dim=1)
        
        # Work arrays for scan
        A_work = A_bar.clone()
        Bu_work = Bu.clone()
        
        # Up-sweep (reduce) phase
        for d in range(log2_len):
            step = 1 << (d + 1)
            half_step = 1 << d
            
            # Indices for parallel combination
            # Combine elements at positions (k*step + half_step - 1) with (k*step + step - 1)
            idx_left = torch.arange(half_step - 1, padded_len, step, device=device)
            idx_right = torch.arange(step - 1, padded_len, step, device=device)
            
            # Apply associative operation: (A_right, Bu_right) ⊕ (A_left, Bu_left)
            # = (A_right * A_left, A_right * Bu_left + Bu_right)
            A_left = A_work[:, idx_left]
            Bu_left = Bu_work[:, idx_left]
            A_right = A_work[:, idx_right]
            Bu_right = Bu_work[:, idx_right]
            
            # Update right positions with combined values
            A_work[:, idx_right] = A_right * A_left
            Bu_work[:, idx_right] = A_right * Bu_left + Bu_right
        
        # Down-sweep phase
        # Set last element to identity for down-sweep
        A_work[:, -1] = 1.0
        Bu_work[:, -1] = 0.0
        
        for d in range(log2_len - 1, -1, -1):
            step = 1 << (d + 1)
            half_step = 1 << d
            
            idx_left = torch.arange(half_step - 1, padded_len, step, device=device)
            idx_right = torch.arange(step - 1, padded_len, step, device=device)
            
            # Save old left values
            A_left_old = A_work[:, idx_left].clone()
            Bu_left_old = Bu_work[:, idx_left].clone()
            
            # Left gets right's value
            A_work[:, idx_left] = A_work[:, idx_right]
            Bu_work[:, idx_left] = Bu_work[:, idx_right]
            
            # Right gets combination: (old_left) ⊕ (old_right)
            A_work[:, idx_right] = A_left_old * A_work[:, idx_right]
            Bu_work[:, idx_right] = A_left_old * Bu_work[:, idx_right] + Bu_left_old
        
        # Bu_work now contains the prefix sums (hidden states)
        # But we need to shift and add back the original contribution
        # h[t] = A_bar[0:t].prod() * h[0] + sum(A_bar[i+1:t].prod() * Bu[i])
        
        # Actually, for SSM starting from h[-1]=0:
        # We need to combine scan result with original A_bar and Bu
        h_all = A_bar * Bu_work + Bu  # Adjust for exclusive vs inclusive scan
        
        # Simpler approach: recompute with scan results as cumulative A products
        # Actually the cleanest is to use the scan directly
        h_all = Bu_work[:, :seq_len]  # Trim padding, this is exclusive scan
        
        # Convert exclusive to inclusive by shifting and adding current Bu
        h_all = torch.cat([
            Bu[:, :1],  # h[0] = Bu[0] (since h[-1] = 0)
            A_bar[:, 1:seq_len] * h_all[:, :-1] + Bu[:, 1:seq_len]
        ], dim=1) if seq_len > 1 else Bu[:, :1]
        
        return h_all

    def _sequential_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential scan implementation for debugging and small sequences.
        
        This is the reference implementation with O(n) sequential steps.
        Used when parallel scan overhead exceeds benefits (very short sequences)
        or for numerical verification.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        device = x.device
        dtype = x.dtype

        # Discretize
        dt_A = torch.einsum("bld,dn->bldn", dt, A)
        A_bar = torch.exp(dt_A)

        dt_B = dt.unsqueeze(-1) * B.unsqueeze(-2)
        B_bar = dt_B

        x_reshaped = x.unsqueeze(-1)
        Bu = B_bar * x_reshaped

        # Sequential scan with proper device placement
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
        ys = []

        for t in range(seq_len):
            h = A_bar[:, t] * h + Bu[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + D * x

        return y, h


class SelectiveStateSpaceBlock(nn.Module):
    """
    Full Selective State Space Block with normalization and residual.
    """

    def __init__(
        self,
        config: StateSpaceConfig,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.ssm = SelectiveStateSpace(config)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward with residual connection."""
        residual = x
        x = self.norm(x)
        x, new_cache = self.ssm(x, cache)
        x = self.dropout(x)
        return residual + x, new_cache


class SelectiveStateSpaceStack(nn.Module):
    """
    Stack of Selective State Space blocks forming the sequence backbone.
    """

    def __init__(
        self,
        config: StateSpaceConfig,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelectiveStateSpaceBlock(config, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward through all layers."""
        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x, nc = layer(x, layer_cache)
            new_cache.append(nc)

        x = self.norm(x)
        return x, new_cache
