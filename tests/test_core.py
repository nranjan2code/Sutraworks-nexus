"""Unit tests for NEXUS core modules.

These tests verify the fundamental functionality of each NEXUS component:
- SelectiveStateSpace: O(n) sequence processing
- HierarchicalWorldModel: JEPA-style prediction
- NeuroSymbolicReasoner: Reasoning with proof traces
- AdaptiveEnergyModule: Energy-based computation
- CausalInferenceEngine: Causal reasoning
- NEXUSCore: Integrated architecture
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

# Test configuration - using smaller sizes for faster tests
BATCH_SIZE = 2
SEQ_LEN = 64
D_MODEL = 128
D_LATENT = 64
VOCAB_SIZE = 1000
D_STATE = 16
N_HEADS = 4
N_LAYERS = 2


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_input(device):
    """Create sample input tensor (token IDs)."""
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)


@pytest.fixture
def sample_hidden(device):
    """Create sample hidden state tensor (continuous)."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device)


class TestSelectiveStateSpace:
    """Tests for SelectiveStateSpace module."""
    
    def test_initialization(self, device):
        """Test module initialization."""
        from nexus.core.state_space import SelectiveStateSpace, StateSpaceConfig
        
        config = StateSpaceConfig(
            d_model=D_MODEL,
            d_state=D_STATE,
            d_conv=4,
            expand=2,
        )
        
        model = SelectiveStateSpace(config).to(device)
        
        assert model is not None
        assert hasattr(model, 'A_log')
        assert hasattr(model, 'in_proj')
        assert hasattr(model, 'out_proj')
    
    def test_forward_pass(self, device, sample_hidden):
        """Test forward pass produces correct output shape."""
        from nexus.core.state_space import SelectiveStateSpace, StateSpaceConfig
        
        config = StateSpaceConfig(
            d_model=D_MODEL,
            d_state=D_STATE,
            d_conv=4,
            expand=2,
        )
        
        model = SelectiveStateSpace(config).to(device)
        
        output, cache = model(sample_hidden)
        
        assert output.shape == sample_hidden.shape
        assert not torch.isnan(output).any()
    
    def test_linear_scaling(self, device):
        """Test that computation scales linearly with sequence length."""
        import time
        from nexus.core.state_space import SelectiveStateSpace, StateSpaceConfig
        
        config = StateSpaceConfig(
            d_model=D_MODEL,
            d_state=D_STATE,
            d_conv=4,
            expand=2,
        )
        
        model = SelectiveStateSpace(config).to(device)
        model.eval()
        
        times = []
        lengths = [64, 128, 256]
        
        with torch.no_grad():
            for seq_len in lengths:
                x = torch.randn(1, seq_len, D_MODEL, device=device)
                
                # Warmup
                _ = model(x)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.time()
                for _ in range(10):
                    _ = model(x)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                times.append((time.time() - start) / 10)
        
        # Check roughly linear scaling (ratio should be ~2 when length doubles)
        ratio = times[2] / times[0]  # 256 / 64
        assert ratio < 6, f"Scaling ratio {ratio} suggests non-linear complexity"
    
    def test_stack_multiple_layers(self, device, sample_hidden):
        """Test stacking multiple state-space layers."""
        from nexus.core.state_space import SelectiveStateSpaceStack, StateSpaceConfig
        
        config = StateSpaceConfig(
            d_model=D_MODEL,
            d_state=D_STATE,
            d_conv=4,
            expand=2,
        )
        
        model = SelectiveStateSpaceStack(config, n_layers=N_LAYERS).to(device)
        
        output, cache = model(sample_hidden)
        
        assert output.shape == sample_hidden.shape


class TestHierarchicalWorldModel:
    """Tests for HierarchicalWorldModel (JEPA-style)."""
    
    def test_initialization(self, device):
        """Test module initialization."""
        from nexus.core.world_model import HierarchicalWorldModel, WorldModelConfig
        
        config = WorldModelConfig(
            d_model=D_MODEL,
            d_latent=D_LATENT,
            n_levels=2,
            predictor_depth=2,
            n_heads=N_HEADS,
        )
        
        model = HierarchicalWorldModel(config).to(device)
        
        assert model is not None
        assert hasattr(model, 'context_encoder')
        assert hasattr(model, 'target_encoder')
        assert hasattr(model, 'predictor')
    
    def test_forward_pass(self, device, sample_hidden):
        """Test forward pass."""
        from nexus.core.world_model import HierarchicalWorldModel, WorldModelConfig
        
        config = WorldModelConfig(
            d_model=D_MODEL,
            d_latent=D_LATENT,
            n_levels=2,
            predictor_depth=2,
            n_heads=N_HEADS,
        )
        
        model = HierarchicalWorldModel(config).to(device)
        
        # Create masks
        context_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool, device=device)
        context_mask[:, :SEQ_LEN//2] = True
        target_mask = ~context_mask
        
        output = model(sample_hidden, context_mask=context_mask, target_mask=target_mask)
        
        assert 'predicted' in output
        assert 'target' in output
        assert output['predicted'].shape[0] == BATCH_SIZE
    
    def test_predict_method(self, device, sample_hidden):
        """Test multi-step future prediction."""
        from nexus.core.world_model import HierarchicalWorldModel, WorldModelConfig
        
        config = WorldModelConfig(
            d_model=D_MODEL,
            d_latent=D_LATENT,
            n_levels=2,
            predictor_depth=2,
            n_heads=N_HEADS,
        )
        
        model = HierarchicalWorldModel(config).to(device)
        
        predictions = model.predict(sample_hidden, n_steps=3)
        
        assert predictions.shape[0] == BATCH_SIZE
        assert predictions.shape[1] == 3  # n_steps


class TestNeuroSymbolicReasoner:
    """Tests for NeuroSymbolicReasoner."""
    
    def test_initialization(self, device):
        """Test module initialization."""
        from nexus.core.reasoning import NeuroSymbolicReasoner, ReasoningConfig
        
        config = ReasoningConfig(
            d_model=D_MODEL,
            d_symbol=64,
            n_predicates=100,
            n_entities=1000,
            n_rules=50,
        )
        
        model = NeuroSymbolicReasoner(config).to(device)
        
        assert model is not None
        assert hasattr(model, 'rule_base')
        assert hasattr(model, 'soft_unification')
    
    def test_forward_pass(self, device):
        """Test forward pass with reasoning."""
        from nexus.core.reasoning import NeuroSymbolicReasoner, ReasoningConfig
        
        config = ReasoningConfig(
            d_model=D_MODEL,
            d_symbol=64,
            n_predicates=100,
            n_entities=1000,
            n_rules=50,
        )
        
        model = NeuroSymbolicReasoner(config).to(device)
        
        # Query should be (batch, d_model)
        query = torch.randn(BATCH_SIZE, D_MODEL, device=device)
        # Context should be (batch, n_facts, d_model)
        context = torch.randn(BATCH_SIZE, 10, D_MODEL, device=device)
        
        output = model(query, context=context)
        
        assert 'answer' in output
        assert output['answer'].shape == (BATCH_SIZE, D_MODEL)


class TestAdaptiveEnergyModule:
    """Tests for AdaptiveEnergyModule."""
    
    def test_initialization(self, device):
        """Test module initialization."""
        from nexus.core.energy import AdaptiveEnergyModule, EnergyConfig
        
        config = EnergyConfig(
            d_model=D_MODEL,
            d_energy=64,
            max_iterations=5,
        )
        
        model = AdaptiveEnergyModule(config).to(device)
        
        assert model is not None
        assert hasattr(model, 'energy_fn')
    
    def test_forward_pass(self, device, sample_hidden):
        """Test forward pass."""
        from nexus.core.energy import AdaptiveEnergyModule, EnergyConfig
        
        config = EnergyConfig(
            d_model=D_MODEL,
            d_energy=64,
            max_iterations=5,
        )
        
        model = AdaptiveEnergyModule(config).to(device)
        
        output = model(sample_hidden)
        
        assert 'output' in output
        assert output['output'].shape == sample_hidden.shape
        assert 'energy' in output
        assert 'confidence' in output


class TestCausalInferenceEngine:
    """Tests for CausalInferenceEngine."""
    
    def test_initialization(self, device):
        """Test module initialization."""
        from nexus.core.causal import CausalInferenceEngine, CausalConfig
        
        config = CausalConfig(
            d_model=D_MODEL,
            n_variables=16,
            d_mechanism=64,
        )
        
        model = CausalInferenceEngine(config).to(device)
        
        assert model is not None
        assert hasattr(model, 'scm')
        assert hasattr(model, 'causal_attention')
    
    def test_forward_pass(self, device, sample_hidden):
        """Test forward pass."""
        from nexus.core.causal import CausalInferenceEngine, CausalConfig
        
        config = CausalConfig(
            d_model=D_MODEL,
            n_variables=16,
            d_mechanism=64,
        )
        
        model = CausalInferenceEngine(config).to(device)
        
        output = model(sample_hidden)
        
        assert 'output' in output
        assert 'causal_graph' in output


class TestNEXUSCore:
    """Tests for integrated NEXUSCore architecture."""
    
    def test_initialization(self, device):
        """Test full model initialization."""
        from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
        
        config = NEXUSConfig(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            d_latent=D_LATENT,
            ssm_n_layers=N_LAYERS,
            n_heads=N_HEADS,
            ssm_d_state=D_STATE,
        )
        
        model = NEXUSCore(config).to(device)
        
        assert model is not None
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
    
    def test_forward_pass(self, device, sample_input):
        """Test full forward pass."""
        from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
        
        config = NEXUSConfig(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            d_latent=D_LATENT,
            ssm_n_layers=N_LAYERS,
            n_heads=N_HEADS,
            ssm_d_state=D_STATE,
        )
        
        model = NEXUSCore(config).to(device)
        
        output = model(sample_input)
        
        assert 'logits' in output
        assert output['logits'].shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    def test_generation(self, device):
        """Test autoregressive generation."""
        from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
        
        config = NEXUSConfig(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            d_latent=D_LATENT,
            ssm_n_layers=N_LAYERS,
            n_heads=N_HEADS,
            ssm_d_state=D_STATE,
        )
        
        model = NEXUSCore(config).to(device)
        model.eval()
        
        prompt = torch.randint(0, VOCAB_SIZE, (1, 16), device=device)
        
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=32)
        
        assert generated.shape[1] == 16 + 32
    
    def test_reasoning_mode(self, device, sample_hidden):
        """Test reasoning mode."""
        from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
        
        config = NEXUSConfig(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            d_latent=D_LATENT,
            ssm_n_layers=N_LAYERS,
            n_heads=N_HEADS,
            ssm_d_state=D_STATE,
        )
        
        model = NEXUSCore(config).to(device)
        
        output = model.reason(sample_hidden)
        
        assert 'answer' in output
    
    def test_imagination_mode(self, device, sample_hidden):
        """Test imagination (world model) mode."""
        from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
        
        config = NEXUSConfig(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            d_latent=D_LATENT,
            ssm_n_layers=N_LAYERS,
            n_heads=N_HEADS,
            ssm_d_state=D_STATE,
        )
        
        model = NEXUSCore(config).to(device)
        
        output = model.imagine(sample_hidden, n_steps=3)
        
        # imagine returns a tensor of predictions
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 3  # n_steps
    
    def test_gradient_flow(self, device, sample_input):
        """Test gradient flow through full model."""
        from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
        
        config = NEXUSConfig(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            d_latent=D_LATENT,
            ssm_n_layers=N_LAYERS,
            n_heads=N_HEADS,
            ssm_d_state=D_STATE,
        )
        
        model = NEXUSCore(config).to(device)
        
        output = model(sample_input)
        logits = output['logits']
        
        # Create dummy loss
        loss = logits.mean()
        loss.backward()
        
        # Check gradients exist for key parameters
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        assert grad_count > 0, "No gradients computed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
