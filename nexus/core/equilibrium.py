"""
Equilibrium-Based Computation - Layer-Free Architecture
=========================================================

The fundamental shift: Instead of stacking N discrete layers,
we define continuous dynamics and evolve to equilibrium.

Traditional Neural Net:
    output = f_N(f_{N-1}(...f_2(f_1(input))))
    
Equilibrium Model:
    Find z* such that z* = f_θ(z*, input)
    
Key insight: "Depth" becomes emergent from the input's complexity,
not a fixed architectural choice. Easy inputs converge fast,
hard inputs require more iterations - computation adapts naturally.

This module provides:
1. ContinuousDynamics - Single parametric function for state evolution
2. EquilibriumSolver - Find fixed points with convergence guarantees
3. ImplicitDifferentiation - Memory-efficient backprop through equilibrium
4. FlowField - Continuous vector field for state space evolution

Philosophy:
    "Growth is not a ladder with rungs to climb.
     It is water finding its level."
    
The network doesn't have layers to traverse - it has a landscape
to settle into. The fixed point IS the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


@dataclass
class EquilibriumConfig:
    """Configuration for equilibrium-based computation."""
    
    d_model: int = 512              # State dimension
    d_hidden: int = 1024            # Hidden dimension in dynamics
    
    # Convergence parameters
    max_iterations: int = 50        # Maximum iterations to equilibrium
    convergence_threshold: float = 1e-4  # Fixed point tolerance
    anderson_memory: int = 5        # Anderson acceleration memory
    
    # Stability parameters
    spectral_norm: bool = True      # Constrain Lipschitz constant
    damping: float = 0.5            # Update damping factor
    
    # Continuous-time parameters  
    dt_min: float = 0.1             # Minimum time step
    dt_max: float = 1.0             # Maximum time step
    adaptive_dt: bool = True        # Adapt time step to convergence
    
    # Training parameters
    dropout: float = 0.1
    phantom_grad: bool = True       # Use phantom gradients for stability
    jac_reg: float = 0.01           # Jacobian regularization weight


class ContinuousDynamics(nn.Module):
    """
    Continuous dynamics function for state evolution.
    
    This is THE core transformation - not one of many layers,
    but the single function that defines how states evolve.
    
    Mathematical form:
        dz/dt = f_θ(z, x, t)
        
    Or in discrete iteration form:
        z_{t+1} = z_t + dt * f_θ(z_t, x)
        
    The architecture ensures stability through:
    - Spectral normalization (Lipschitz bound)
    - Residual structure (identity + perturbation)
    - Damping (momentum-like smoothing)
    """
    
    def __init__(self, config: EquilibriumConfig):
        super().__init__()
        self.config = config
        
        # Core transformation - THIS is what replaces all layers
        self.transform = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_model),
        )
        
        # Time embedding for continuous-time dynamics
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.d_model),
        )
        
        # Gating mechanism for selective update
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )
        
        # Apply spectral normalization for stability
        if config.spectral_norm:
            self._apply_spectral_norm()
            
    def _apply_spectral_norm(self):
        """Apply spectral normalization to bound Lipschitz constant."""
        for module in self.transform.modules():
            if isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module)
                
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute dynamics dz/dt = f(z, x, t).
        
        Args:
            z: Current state (batch, seq_len, d_model)
            x: Input/context (batch, seq_len, d_model)
            t: Optional time parameter (batch,) or scalar
            
        Returns:
            State derivative/update (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = z.shape
        
        # Combine state and input
        combined = torch.cat([z, x], dim=-1)
        
        # Core transformation
        delta = self.transform(combined)
        
        # Add time embedding if provided
        if t is not None:
            if t.dim() == 0:
                t = t.expand(batch)
            t_embed = self.time_embed(t.view(-1, 1))
            t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)
            delta = delta + t_embed
            
        # Gated update - allows selective modification
        gate = self.gate(combined)
        delta = gate * delta
        
        return delta


class EquilibriumSolver(nn.Module):
    """
    Solver that finds fixed points of the dynamics.
    
    Instead of forward pass through layers, we SOLVE for equilibrium:
        Find z* such that f(z*, x) ≈ 0 (or z* = g(z*, x))
    
    Uses Anderson acceleration for faster convergence.
    """
    
    def __init__(self, config: EquilibriumConfig):
        super().__init__()
        self.config = config
        self.dynamics = ContinuousDynamics(config)
        
        # Initial state generator
        self.init_state = nn.Sequential(
            nn.Linear(config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, config.d_model),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        z_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Find equilibrium state given input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            z_init: Optional initial state
            
        Returns:
            Dictionary with equilibrium state and convergence info
        """
        if z_init is None:
            z = self.init_state(x)
        else:
            z = z_init
            
        # Use appropriate solver
        if self.config.anderson_memory > 0:
            z_star, info = self._anderson_solve(z, x)
        else:
            z_star, info = self._fixed_point_solve(z, x)
            
        return {
            "equilibrium": z_star,
            "iterations": info["iterations"],
            "converged": info["converged"],
            "residual": info["residual"],
            "trajectory": info.get("trajectory", None),
        }
        
    def _fixed_point_solve(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Simple fixed-point iteration with damping."""
        trajectory = [z] if self.training else None
        
        for i in range(self.config.max_iterations):
            # Compute update
            delta = self.dynamics(z, x)
            z_new = z + self.config.damping * delta
            
            # Check convergence
            residual = (z_new - z).norm(dim=-1).mean()
            
            if trajectory is not None:
                trajectory.append(z_new)
                
            if residual < self.config.convergence_threshold:
                return z_new, {
                    "iterations": i + 1,
                    "converged": True,
                    "residual": residual,
                    "trajectory": trajectory,
                }
                
            z = z_new
            
        return z, {
            "iterations": self.config.max_iterations,
            "converged": False,
            "residual": residual,
            "trajectory": trajectory,
        }
        
    def _anderson_solve(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Anderson acceleration for faster convergence.
        
        Maintains a history of iterates and finds optimal linear
        combination for the next iterate.
        """
        m = self.config.anderson_memory
        batch, seq_len, d_model = z.shape
        
        # Flatten for Anderson
        z_flat = z.view(batch, -1)
        x_flat_for_dynamics = x  # Keep original shape for dynamics
        
        # History buffers
        X_history = []  # Past iterates
        F_history = []  # Past function values
        
        trajectory = [z] if self.training else None
        
        for i in range(self.config.max_iterations):
            # Compute f(z) = z + dynamics(z, x)
            z_3d = z_flat.view(batch, seq_len, d_model)
            delta = self.dynamics(z_3d, x_flat_for_dynamics)
            f_z = z_flat + self.config.damping * delta.view(batch, -1)
            
            # Check convergence
            residual = (f_z - z_flat).norm(dim=-1).mean()
            
            if residual < self.config.convergence_threshold:
                z_out = f_z.view(batch, seq_len, d_model)
                if trajectory is not None:
                    trajectory.append(z_out)
                return z_out, {
                    "iterations": i + 1,
                    "converged": True,
                    "residual": residual,
                    "trajectory": trajectory,
                }
                
            # Anderson update
            X_history.append(z_flat)
            F_history.append(f_z)
            
            if len(X_history) > m:
                X_history.pop(0)
                F_history.pop(0)
                
            if len(X_history) > 1:
                # Solve least squares for optimal combination
                z_flat = self._anderson_step(X_history, F_history)
            else:
                z_flat = f_z
                
            if trajectory is not None:
                trajectory.append(z_flat.view(batch, seq_len, d_model))
                
        z_out = z_flat.view(batch, seq_len, d_model)
        return z_out, {
            "iterations": self.config.max_iterations,
            "converged": False,
            "residual": residual,
            "trajectory": trajectory,
        }
        
    def _anderson_step(
        self,
        X: list,
        F: list,
    ) -> torch.Tensor:
        """Compute Anderson acceleration step."""
        # Stack histories
        X_mat = torch.stack(X, dim=1)  # (batch, m, dim)
        F_mat = torch.stack(F, dim=1)  # (batch, m, dim)
        
        # Compute residuals
        G_mat = F_mat - X_mat  # (batch, m, dim)
        
        # Solve for optimal coefficients
        # min ||sum_i alpha_i * g_i||^2 s.t. sum_i alpha_i = 1
        
        # Gram matrix
        GTG = torch.bmm(G_mat, G_mat.transpose(1, 2))  # (batch, m, m)
        
        # Regularize for stability
        GTG = GTG + 1e-6 * torch.eye(GTG.shape[1], device=GTG.device)
        
        # Solve with constraint sum(alpha) = 1
        ones = torch.ones(GTG.shape[0], GTG.shape[1], 1, device=GTG.device)
        alpha = torch.linalg.solve(GTG, ones)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        
        # Weighted combination
        z_new = torch.bmm(alpha.transpose(1, 2), F_mat).squeeze(1)
        
        return z_new


class ImplicitFunction(Function):
    """
    Implicit differentiation through equilibrium.
    
    Key insight: We don't need to backprop through all iterations!
    At equilibrium z* = f(z*, x), we can use implicit function theorem:
    
        dz*/dx = (I - df/dz)^{-1} * df/dx
        
    This gives O(1) memory regardless of iteration count.
    """
    
    @staticmethod
    def forward(ctx, dynamics, x, z_star, config):
        """
        Forward pass just returns the equilibrium.
        We save what we need for backward.
        """
        ctx.save_for_backward(x, z_star)
        ctx.dynamics = dynamics
        ctx.config = config
        return z_star
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using implicit differentiation.
        
        Solves: (I - J^T) * v = grad_output
        Where J = d(dynamics)/dz at equilibrium
        """
        x, z_star = ctx.saved_tensors
        dynamics = ctx.dynamics
        config = ctx.config
        
        # Solve for v using fixed-point iteration
        v = grad_output.clone()
        
        with torch.enable_grad():
            z_star = z_star.detach().requires_grad_(True)
            
            for _ in range(config.max_iterations // 2):
                # Compute J^T * v via vector-Jacobian product
                delta = dynamics(z_star, x)
                Jv = torch.autograd.grad(
                    delta, z_star, v,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                
                # Fixed point: v = grad_output + damping * J^T * v
                v_new = grad_output + config.damping * Jv
                
                # Check convergence
                if (v_new - v).norm() < config.convergence_threshold:
                    break
                    
                v = v_new
                
        # Gradient w.r.t. x
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            z_star = z_star.detach().requires_grad_(True)
            delta = dynamics(z_star, x)
            
            grad_x = torch.autograd.grad(
                delta, x, v,
                retain_graph=False,
                create_graph=False,
            )[0]
            
        return None, grad_x, None, None


class FlowField(nn.Module):
    """
    Continuous vector field for state evolution.
    
    Instead of discrete layers, we define a continuous flow:
        dz/dt = v(z, x, t)
        
    States evolve along this flow until reaching equilibrium.
    This is the continuous-time analog of layer-free computation.
    """
    
    def __init__(self, config: EquilibriumConfig):
        super().__init__()
        self.config = config
        
        # Vector field parameterization
        self.velocity = nn.Sequential(
            nn.Linear(config.d_model * 2 + 1, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.GELU(), 
            nn.Linear(config.d_hidden, config.d_model),
        )
        
        # Divergence-free component (optional, for volume-preserving flow)
        self.curl = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_hidden),
            nn.Tanh(),
            nn.Linear(config.d_hidden, config.d_model),
        )
        
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity field v(z, x, t).
        
        Args:
            z: Current state
            x: Context/input
            t: Time (scalar or batch)
            
        Returns:
            Velocity vector at (z, t)
        """
        # Expand t to match z shape
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        t = t.view(-1, 1, 1).expand(-1, z.shape[1], 1)
        
        # Combine inputs
        combined = torch.cat([z, x, t], dim=-1)
        
        # Main velocity
        v = self.velocity(combined)
        
        # Add curl component for interesting dynamics
        curl_input = torch.cat([z, x], dim=-1)
        v = v + 0.1 * self.curl(curl_input)
        
        return v


class NeuralODE(nn.Module):
    """
    Neural ODE solver for continuous-depth networks.
    
    Integrates the flow field from t=0 to t=T:
        z(T) = z(0) + ∫₀ᵀ v(z(t), x, t) dt
        
    Uses adaptive step size for efficiency.
    """
    
    def __init__(self, config: EquilibriumConfig):
        super().__init__()
        self.config = config
        self.flow = FlowField(config)
        
    def forward(
        self,
        z0: torch.Tensor,
        x: torch.Tensor,
        T: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate flow from t=0 to t=T.
        
        Args:
            z0: Initial state
            x: Context/input
            T: Integration time (effective "depth")
            
        Returns:
            Final state and trajectory info
        """
        z = z0
        t = torch.tensor(0.0, device=z.device)
        dt = self.config.dt_min
        
        trajectory = [z] if self.training else None
        n_steps = 0
        
        while t < T:
            # Adaptive time step
            if self.config.adaptive_dt:
                dt = self._adaptive_step(z, x, t, dt)
                
            # Don't overshoot
            if t + dt > T:
                dt = T - t
                
            # RK4 integration step
            z = self._rk4_step(z, x, t, dt)
            
            t = t + dt
            n_steps += 1
            
            if trajectory is not None:
                trajectory.append(z)
                
        return {
            "output": z,
            "trajectory": trajectory,
            "n_steps": n_steps,
            "final_time": t,
        }
        
    def _rk4_step(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Fourth-order Runge-Kutta step."""
        k1 = self.flow(z, x, t)
        k2 = self.flow(z + 0.5 * dt * k1, x, t + 0.5 * dt)
        k3 = self.flow(z + 0.5 * dt * k2, x, t + 0.5 * dt)
        k4 = self.flow(z + dt * k3, x, t + dt)
        
        return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    def _adaptive_step(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> float:
        """Adapt step size based on local curvature."""
        v = self.flow(z, x, t)
        v_norm = v.norm(dim=-1).mean()
        
        # Smaller steps when velocity is large
        dt_new = self.config.dt_max / (1 + v_norm)
        dt_new = max(self.config.dt_min, min(self.config.dt_max, dt_new))
        
        return dt_new


class EquilibriumCore(nn.Module):
    """
    Layer-Free Core Module - The Heart of Continuous NEXUS.
    
    This replaces ALL stacked layers with:
    1. A single dynamics function iterated to equilibrium
    2. Implicit differentiation for memory-efficient training
    3. Adaptive computation depth based on input complexity
    
    Architecture Philosophy:
        Traditional: input → layer1 → layer2 → ... → layerN → output
        Equilibrium: input → evolve(state) → equilibrium → output
        
    The "depth" is not a hyperparameter but an emergent property
    of the input's complexity and the dynamics landscape.
    """
    
    def __init__(self, config: EquilibriumConfig):
        super().__init__()
        self.config = config
        
        # The ONE dynamics function (replaces N layers)
        self.solver = EquilibriumSolver(config)
        
        # Optional: Neural ODE for continuous-time interpretation
        self.neural_ode = NeuralODE(config)
        
        # Input processing
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        
        # Output processing
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        self.output_norm = nn.LayerNorm(config.d_model)
        
        # Mode selection
        self.use_ode = False  # Set True for continuous-time mode
        
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "equilibrium",
        T: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through layer-free dynamics.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mode: "equilibrium" (fixed-point) or "ode" (continuous-time)
            T: Integration time for ODE mode
            
        Returns:
            Dictionary with output and computation metrics
        """
        # Process input
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        if mode == "equilibrium":
            result = self.solver(x)
            output = result["equilibrium"]
        else:  # ODE mode
            z0 = x  # Start from input
            result = self.neural_ode(z0, x, T=T)
            output = result["output"]
            
        # Process output
        output = self.output_proj(output)
        output = self.output_norm(output)
        
        # Apply implicit differentiation if training
        if self.training and mode == "equilibrium":
            output = ImplicitFunction.apply(
                self.solver.dynamics,
                x,
                output,
                self.config,
            )
            
        return {
            "output": output,
            **result,
        }
        
    def compute_jacobian_penalty(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobian regularization for training stability.
        
        Encourages the dynamics to have spectral radius < 1
        at the equilibrium, ensuring stable fixed point.
        """
        z = z.detach().requires_grad_(True)
        
        # Compute dynamics
        delta = self.solver.dynamics(z, x)
        
        # Estimate spectral norm via power iteration
        v = torch.randn_like(z)
        v = v / v.norm(dim=-1, keepdim=True)
        
        for _ in range(3):  # Few power iterations
            Jv = torch.autograd.grad(
                delta, z, v,
                retain_graph=True,
                create_graph=True,
            )[0]
            v = Jv / (Jv.norm(dim=-1, keepdim=True) + 1e-8)
            
        # Final Jv gives estimate of largest singular value direction
        Jv = torch.autograd.grad(
            delta, z, v,
            retain_graph=False,
            create_graph=True,
        )[0]
        
        spectral_norm = Jv.norm(dim=-1).mean()
        
        # Penalize if spectral norm > 1 (unstable fixed point)
        penalty = F.relu(spectral_norm - 0.99)
        
        return penalty


class ContinuousAttention(nn.Module):
    """
    Attention as continuous flow, not discrete operation.
    
    Instead of Q, K, V projections and softmax in one shot,
    we define attention as a continuous process where states
    gradually align with each other.
    """
    
    def __init__(self, config: EquilibriumConfig):
        super().__init__()
        self.config = config
        
        # Continuous attention dynamics
        self.query_flow = nn.Linear(config.d_model, config.d_model)
        self.key_flow = nn.Linear(config.d_model, config.d_model)
        self.value_flow = nn.Linear(config.d_model, config.d_model)
        
        # Scale
        self.scale = config.d_model ** -0.5
        
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        One step of continuous attention evolution.
        
        Args:
            z: Current state
            x: Context (keys/values source)
            dt: Time step
            
        Returns:
            Updated state after attention step
        """
        Q = self.query_flow(z)
        K = self.key_flow(x)
        V = self.value_flow(x)
        
        # Soft attention (continuous, not hard selection)
        attn_logits = torch.bmm(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Attended value
        attended = torch.bmm(attn_weights, V)
        
        # Continuous update (not replacement)
        z_new = z + dt * (attended - z)
        
        return z_new


class ContinuousMemory(nn.Module):
    """
    Memory as a continuous evolving state, not discrete slots.
    
    The memory doesn't have fixed "read" and "write" operations -
    it continuously co-evolves with the processing state.
    """
    
    def __init__(self, config: EquilibriumConfig, memory_size: int = 128):
        super().__init__()
        self.config = config
        self.memory_size = memory_size
        
        # Learnable initial memory
        self.memory_init = nn.Parameter(
            torch.randn(memory_size, config.d_model) * 0.02
        )
        
        # Memory-state interaction
        self.memory_update = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, config.d_model),
        )
        
        self.state_update = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, config.d_model),
        )
        
    def forward(
        self,
        z: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Co-evolve state and memory.
        
        Args:
            z: Current state (batch, seq_len, d_model)
            memory: Current memory (batch, memory_size, d_model)
            dt: Time step
            
        Returns:
            Updated state and memory
        """
        batch = z.shape[0]
        
        if memory is None:
            memory = self.memory_init.unsqueeze(0).expand(batch, -1, -1)
            
        # State attends to memory
        state_summary = z.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        memory_summary = memory.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        
        # Update memory based on state
        mem_input = torch.cat([
            memory,
            state_summary.expand(-1, self.memory_size, -1)
        ], dim=-1)
        memory_delta = self.memory_update(mem_input)
        memory_new = memory + dt * memory_delta
        
        # Update state based on memory
        state_input = torch.cat([
            z,
            memory_summary.expand(-1, z.shape[1], -1)
        ], dim=-1)
        state_delta = self.state_update(state_input)
        z_new = z + dt * state_delta
        
        return z_new, memory_new
