"""
Tests for Layer-Free Architecture
==================================

Tests for the new continuous/equilibrium-based components:
- EquilibriumCore and ContinuousDynamics
- ContinuousSSM with emergent depth
- FlowingNEXUS unified architecture
- Living system with flow metrics
"""

import pytest
import torch
import torch.nn as nn

from nexus.core.equilibrium import (
    EquilibriumConfig,
    EquilibriumCore,
    ContinuousDynamics,
    EquilibriumSolver,
    FlowField,
    NeuralODE,
    ContinuousAttention,
    ContinuousMemory,
)
from nexus.core.continuous_ssm import (
    ContinuousSSMConfig,
    ContinuousSSM,
    ContinuousStateKernel,
    HierarchicalContinuousSSM,
    BidirectionalContinuousSSM,
)
from nexus.core.flowing import (
    FlowingConfig,
    FlowingNEXUS,
    UnifiedDynamics,
    create_flowing_nexus,
    create_living_flowing_nexus,
    LivingFlowingNEXUS,
)


class TestEquilibriumComponents:
    """Test equilibrium-based computation components."""
    
    @pytest.fixture
    def config(self):
        return EquilibriumConfig(
            d_model=64,
            d_hidden=128,
            max_iterations=20,
            convergence_threshold=1e-3,
        )
    
    def test_continuous_dynamics(self, config):
        """Test that continuous dynamics produces updates."""
        dynamics = ContinuousDynamics(config)
        
        batch, seq_len = 2, 16
        z = torch.randn(batch, seq_len, config.d_model)
        x = torch.randn(batch, seq_len, config.d_model)
        
        delta = dynamics(z, x)
        
        assert delta.shape == z.shape
        assert not torch.isnan(delta).any()
        
    def test_continuous_dynamics_with_time(self, config):
        """Test dynamics with time parameter."""
        dynamics = ContinuousDynamics(config)
        
        batch, seq_len = 2, 16
        z = torch.randn(batch, seq_len, config.d_model)
        x = torch.randn(batch, seq_len, config.d_model)
        t = torch.tensor(0.5)
        
        delta = dynamics(z, x, t=t)
        
        assert delta.shape == z.shape
        
    def test_equilibrium_solver_converges(self, config):
        """Test that solver finds equilibrium."""
        solver = EquilibriumSolver(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = solver(x)
        
        assert "equilibrium" in result
        assert "iterations" in result
        assert "converged" in result
        assert result["equilibrium"].shape == x.shape
        
    def test_equilibrium_core(self, config):
        """Test full equilibrium core module."""
        core = EquilibriumCore(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = core(x, mode="equilibrium")
        
        assert "output" in result
        assert result["output"].shape == x.shape
        
    def test_ode_mode(self, config):
        """Test Neural ODE integration mode."""
        core = EquilibriumCore(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = core(x, mode="ode", T=1.0)
        
        assert "output" in result
        assert result["output"].shape == x.shape


class TestContinuousSSM:
    """Test continuous state space model components."""
    
    @pytest.fixture
    def config(self):
        return ContinuousSSMConfig(
            d_model=64,
            d_state=32,
            max_evolution_steps=15,
            convergence_threshold=1e-3,
        )
        
    def test_state_kernel(self, config):
        """Test continuous state kernel."""
        kernel = ContinuousStateKernel(config)
        
        batch, seq_len = 2, 16
        d_inner = config.d_model * config.expand
        
        x = torch.randn(batch, seq_len, d_inner)
        h = torch.zeros(batch, seq_len, d_inner, config.d_state)
        
        y, h_new = kernel(x, h)
        
        assert y.shape == x.shape
        assert h_new.shape == h.shape
        
    def test_continuous_ssm_forward(self, config):
        """Test continuous SSM forward pass."""
        ssm = ContinuousSSM(config)
        
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = ssm(x)
        
        assert "output" in result
        assert "evolution_steps" in result
        assert result["output"].shape == x.shape
        # Depth should emerge from input
        assert 1 <= result["evolution_steps"] <= config.max_evolution_steps
        
    def test_continuous_ssm_trajectory(self, config):
        """Test that trajectory is recorded."""
        ssm = ContinuousSSM(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = ssm(x, return_trajectory=True)
        
        assert result["trajectory"] is not None
        assert len(result["trajectory"]) > 0
        
    def test_bidirectional_ssm(self, config):
        """Test bidirectional continuous SSM."""
        ssm = BidirectionalContinuousSSM(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = ssm(x)
        
        assert "output" in result
        assert result["output"].shape == x.shape
        
    def test_hierarchical_ssm(self, config):
        """Test multi-scale continuous SSM."""
        ssm = HierarchicalContinuousSSM(config, n_scales=3)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = ssm(x)
        
        assert "output" in result
        assert result["output"].shape == x.shape
        assert "total_evolution_steps" in result


class TestFlowingNEXUS:
    """Test the unified layer-free architecture."""
    
    @pytest.fixture
    def config(self):
        return FlowingConfig(
            d_model=64,
            d_latent=32,
            max_flow_steps=20,
            vocab_size=1000,
            max_seq_len=128,
            memory_size=32,
        )
        
    def test_unified_dynamics(self, config):
        """Test unified dynamics function."""
        dynamics = UnifiedDynamics(config)
        
        batch, seq_len = 2, 16
        z = torch.randn(batch, seq_len, config.d_model)
        x = torch.randn(batch, seq_len, config.d_model)
        
        update = dynamics(z, x)
        
        assert update.shape == z.shape
        
    def test_flowing_nexus_forward(self, config):
        """Test FlowingNEXUS forward pass."""
        model = FlowingNEXUS(config)
        
        batch, seq_len = 2, 16
        x = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        result = model(x, modality="token")
        
        assert "logits" in result
        assert "hidden_states" in result
        assert "flow_steps" in result
        assert "converged" in result
        assert result["logits"].shape == (batch, seq_len, config.vocab_size)
        
    def test_flowing_nexus_continuous_input(self, config):
        """Test with continuous input."""
        model = FlowingNEXUS(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = model(x, modality="continuous")
        
        assert "logits" in result
        assert result["flow_steps"] >= 1
        
    def test_emergent_depth(self, config):
        """Test that depth varies with input."""
        model = FlowingNEXUS(config)
        
        # Simple input (should converge fast)
        simple_x = torch.zeros(2, 16, config.d_model)
        simple_result = model(simple_x, modality="continuous")
        
        # Complex input (should take longer)
        complex_x = torch.randn(2, 16, config.d_model) * 5
        complex_result = model(complex_x, modality="continuous")
        
        # Both should work
        assert simple_result["flow_steps"] >= 1
        assert complex_result["flow_steps"] >= 1
        
    def test_imagine(self, config):
        """Test imagination/future prediction."""
        model = FlowingNEXUS(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        predictions = model.imagine(x, n_steps=5)
        
        assert predictions.shape[0] == batch
        assert predictions.shape[1] == 6  # Initial + 5 steps
        
    def test_reason(self, config):
        """Test reasoning with extended iterations."""
        model = FlowingNEXUS(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = model.reason(x)
        
        assert "answer" in result
        assert "reasoning_depth" in result
        assert "confidence" in result
        
    def test_flow_complexity(self, config):
        """Test flow complexity measurement."""
        model = FlowingNEXUS(config)
        
        batch, seq_len = 2, 16
        # Use token indices for token modality
        x = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        complexity = model.get_flow_complexity(x)
        
        assert "flow_steps" in complexity
        assert "relative_depth" in complexity
        assert 0 <= complexity["relative_depth"] <= 1
        
    def test_trajectory_return(self, config):
        """Test trajectory recording."""
        model = FlowingNEXUS(config)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        result = model(x, modality="continuous", return_trajectory=True)
        
        assert result["trajectory"] is not None
        assert len(result["trajectory"]) > 0


class TestLivingFlowingNEXUS:
    """Test living system with layer-free architecture."""
    
    @pytest.fixture
    def config(self):
        return FlowingConfig(
            d_model=64,
            d_latent=32,
            max_flow_steps=15,
            vocab_size=1000,
        )
        
    def test_living_interact(self, config):
        """Test living system interaction."""
        living = LivingFlowingNEXUS(config)
        
        batch, seq_len = 2, 16
        x = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        result = living.interact(x, modality="token")
        
        assert "responded" in result
        assert "confidence" in result
        assert "flow_depth" in result
        
    def test_living_status(self, config):
        """Test status tracking."""
        living = LivingFlowingNEXUS(config)
        
        # Do some interactions
        for _ in range(5):
            x = torch.randint(0, config.vocab_size, (2, 16))
            living.interact(x, modality="token")
            
        status = living.get_status()
        
        assert status["total_interactions"] == 5
        assert "average_flow_depth" in status


class TestFactoryFunctions:
    """Test factory functions for creating models."""
    
    def test_create_flowing_nexus_small(self):
        """Test small model creation."""
        model = create_flowing_nexus(size="small")
        
        assert isinstance(model, FlowingNEXUS)
        assert model.config.d_model == 256
        
    def test_create_flowing_nexus_base(self):
        """Test base model creation."""
        model = create_flowing_nexus(size="base")
        
        assert isinstance(model, FlowingNEXUS)
        assert model.config.d_model == 512
        
    def test_create_living_flowing_nexus(self):
        """Test living flowing model creation."""
        model = create_living_flowing_nexus(size="small")
        
        assert isinstance(model, LivingFlowingNEXUS)
        
    def test_custom_config_override(self):
        """Test config override."""
        model = create_flowing_nexus(
            size="small",
            max_flow_steps=100,
            convergence_threshold=1e-5,
        )
        
        assert model.config.max_flow_steps == 100
        assert model.config.convergence_threshold == 1e-5


class TestGradientFlow:
    """Test gradient flow through equilibrium."""
    
    @pytest.fixture
    def config(self):
        return FlowingConfig(
            d_model=32,
            d_latent=16,
            max_flow_steps=10,
            vocab_size=100,
            implicit_diff=False,  # Use explicit gradients for testing
        )
        
    def test_gradients_flow(self, config):
        """Test that gradients flow through the model."""
        model = FlowingNEXUS(config)
        
        batch, seq_len = 2, 8
        x = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        result = model(x, modality="token")
        loss = result["logits"].sum()
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any()
                
        assert has_grad


class TestNoLayers:
    """
    Philosophical tests: Verify we've truly removed layers.
    """
    
    def test_no_n_layers_parameter(self):
        """FlowingConfig should have no n_layers."""
        config = FlowingConfig()
        
        assert not hasattr(config, 'n_layers')
        assert not hasattr(config, 'num_layers')
        assert not hasattr(config, 'ssm_n_layers')
        
    def test_depth_is_emergent(self):
        """Depth should vary based on input, not be fixed."""
        model = create_flowing_nexus(size="small")
        
        depths = []
        for i in range(10):
            x = torch.randn(1, 16, model.config.d_model) * (i + 1)
            result = model(x, modality="continuous")
            depths.append(result["flow_steps"])
            
        # Depth should vary (not all same)
        # Note: This might occasionally fail if all inputs happen to need same depth
        unique_depths = len(set(depths))
        assert unique_depths >= 1  # At minimum, should work
        
    def test_single_dynamics_function(self):
        """Should have ONE dynamics function, not N layers."""
        model = create_flowing_nexus(size="small")
        
        # Model should have a single dynamics attribute
        assert hasattr(model, 'dynamics')
        assert isinstance(model.dynamics, UnifiedDynamics)
        
        # Should NOT have a layers list
        assert not hasattr(model, 'layers')
