"""
NEXUS Core Architecture
========================

The main NEXUS (Neural EXploratory Unified Synthesis) architecture that integrates
all components into a unified next-generation AI system.

NEXUS represents a paradigm shift from Transformer-based LLMs by combining:

1. Selective State Space Models → O(n) sequence processing
2. Hierarchical World Models → Abstract predictive representations
3. Neuro-Symbolic Reasoning → Grounded logical inference
4. Energy-Based Computation → Adaptive depth and uncertainty
5. Causal Inference → Principled cause-effect reasoning

This creates an AI system that can:
- Process long sequences efficiently (linear time)
- Build abstract world models (not just next-token prediction)
- Reason logically with explainable chains
- Quantify uncertainty meaningfully
- Understand causation (not just correlation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.core.state_space import SelectiveStateSpace, StateSpaceConfig, SelectiveStateSpaceStack
from nexus.core.world_model import HierarchicalWorldModel, WorldModelConfig, jepa_loss
from nexus.core.reasoning import NeuroSymbolicReasoner, ReasoningConfig
from nexus.core.energy import AdaptiveEnergyModule, EnergyConfig
from nexus.core.causal import CausalInferenceEngine, CausalConfig


@dataclass
class NEXUSConfig:
    """
    Unified configuration for NEXUS architecture.

    This configuration controls all sub-modules and their interactions.
    """

    # Model dimensions
    d_model: int = 512  # Main model dimension
    d_latent: int = 256  # Latent representation dimension

    # Sequence processing (State Space)
    ssm_d_state: int = 16  # SSM state dimension
    ssm_d_conv: int = 4  # SSM convolution width
    ssm_expand: int = 2  # SSM expansion factor
    ssm_n_layers: int = 8  # Number of SSM layers

    # World modeling
    world_model_levels: int = 3  # Hierarchical abstraction levels
    world_model_depth: int = 4  # Predictor depth
    world_model_context_ratio: float = 0.5

    # Reasoning
    n_predicates: int = 1000  # Symbolic predicates
    n_entities: int = 10000  # Symbolic entities
    max_proof_depth: int = 5  # Maximum reasoning depth

    # Energy-based computation
    max_iterations: int = 10  # Maximum refinement iterations
    energy_threshold: float = 0.1  # Convergence threshold

    # Causal reasoning
    n_causal_variables: int = 32  # Causal variable capacity

    # General
    n_heads: int = 8  # Attention heads
    dropout: float = 0.1  # Dropout rate
    vocab_size: int = 50000  # Vocabulary size (if using tokens)
    max_seq_len: int = 8192  # Maximum sequence length

    def get_ssm_config(self) -> StateSpaceConfig:
        """Get configuration for state space module."""
        return StateSpaceConfig(
            d_model=self.d_model,
            d_state=self.ssm_d_state,
            d_conv=self.ssm_d_conv,
            expand=self.ssm_expand,
        )

    def get_world_model_config(self) -> WorldModelConfig:
        """Get configuration for world model."""
        return WorldModelConfig(
            d_model=self.d_model,
            d_latent=self.d_latent,
            n_levels=self.world_model_levels,
            context_ratio=self.world_model_context_ratio,
            predictor_depth=self.world_model_depth,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )

    def get_reasoning_config(self) -> ReasoningConfig:
        """Get configuration for reasoning module."""
        return ReasoningConfig(
            d_model=self.d_model,
            n_predicates=self.n_predicates,
            n_entities=self.n_entities,
            max_proof_depth=self.max_proof_depth,
            dropout=self.dropout,
        )

    def get_energy_config(self) -> EnergyConfig:
        """Get configuration for energy module."""
        return EnergyConfig(
            d_model=self.d_model,
            max_iterations=self.max_iterations,
            energy_threshold=self.energy_threshold,
            dropout=self.dropout,
        )

    def get_causal_config(self) -> CausalConfig:
        """Get configuration for causal module."""
        return CausalConfig(
            d_model=self.d_model,
            n_variables=self.n_causal_variables,
            dropout=self.dropout,
        )


class NEXUSEmbedding(nn.Module):
    """
    Input embedding layer for NEXUS.

    Supports multiple input modalities:
    - Token sequences (language)
    - Continuous vectors (features)
    - Structured inputs (graphs, tables)
    """

    def __init__(self, config: NEXUSConfig):
        super().__init__()
        self.config = config

        # Token embedding (for discrete inputs)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Continuous projection (for feature inputs)
        self.continuous_proj = nn.Linear(config.d_model, config.d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)

        # Modality type embedding
        self.modality_embedding = nn.Embedding(4, config.d_model)  # 4 modalities

        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        modality: str = "token",
    ) -> torch.Tensor:
        """
        Embed input into model space.

        Args:
            x: Input tensor
               - For tokens: (batch, seq_len) long tensor
               - For continuous: (batch, seq_len, d_model) float tensor
            modality: Input type - "token", "continuous", "graph", "table"

        Returns:
            Embedded tensor (batch, seq_len, d_model)
        """
        if modality == "token":
            x = self.token_embedding(x)
        elif modality == "continuous":
            x = self.continuous_proj(x)
        else:
            x = self.continuous_proj(x)

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]

        # Add modality embedding
        modality_map = {"token": 0, "continuous": 1, "graph": 2, "table": 3}
        modality_idx = modality_map.get(modality, 0)
        modality_emb = self.modality_embedding(torch.tensor([modality_idx], device=x.device))
        x = x + modality_emb

        x = self.norm(x)
        x = self.dropout(x)

        return x


class NEXUSCore(nn.Module):
    """
    NEXUS Core Architecture - Next-Generation AI System.

    This is the main model class that orchestrates all NEXUS components:

    ┌─────────────────────────────────────────────────────────────┐
    │                         NEXUS                                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │  Selective  │  │ Hierarchical│  │  Neuro-Symbolic    │  │
    │  │ State Space │──│ World Model │──│    Reasoner        │  │
    │  │   O(n)      │  │   (JEPA)    │  │  (Logic+Neural)    │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    │         │                │                    │             │
    │         └────────────────┼────────────────────┘             │
    │                          ▼                                  │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              Adaptive Energy Module                  │   │
    │  │        (Uncertainty + Adaptive Computation)          │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                          │                                  │
    │                          ▼                                  │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │            Causal Inference Engine                   │   │
    │  │         (Planning + Decision Making)                 │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

    Key Innovations:
    1. Linear-time sequence processing (not quadratic attention)
    2. Predictive world modeling (not just next-token)
    3. Grounded symbolic reasoning (not hallucination)
    4. Adaptive computation (not fixed depth)
    5. Causal understanding (not just correlation)
    """

    def __init__(self, config: NEXUSConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.embedding = NEXUSEmbedding(config)

        # Core modules
        self.state_space = SelectiveStateSpaceStack(
            config.get_ssm_config(),
            n_layers=config.ssm_n_layers,
            dropout=config.dropout,
        )

        self.world_model = HierarchicalWorldModel(config.get_world_model_config())

        self.reasoner = NeuroSymbolicReasoner(config.get_reasoning_config())

        self.energy_module = AdaptiveEnergyModule(config.get_energy_config())

        self.causal_engine = CausalInferenceEngine(config.get_causal_config())

        # Integration layers
        self.integration = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Output heads
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.regression_head = nn.Linear(config.d_model, config.d_model)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Tie embedding weights with lm_head
        self.lm_head.weight = self.embedding.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        modality: str = "token",
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        state: Optional[List[Any]] = None,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through NEXUS.

        Args:
            x: Input tensor
            modality: Input modality type
            context_mask: Mask for world model context
            target_mask: Mask for world model targets
            state: Optional cached state for autoregressive generation
            return_all: If True, return all intermediate outputs

        Returns:
            Dictionary with predictions and metrics.
            If state is provided/returned, it's in result["state"].
        """
        batch, seq_len = x.shape[:2]

        # Embed input
        embedded = self.embedding(x, modality=modality)

        # Stage 1: Efficient sequence processing
        # Pass state/cache to SSM
        ssm_output, new_state = self.state_space(embedded, cache=state)

        # Stage 2: World modeling (if masks provided)
        # For generation loop with cache, we skip world/reasoning unless explicitly requested or applicable
        # because those modules might expect full context.
        # But for correctness, we'll run them if possible. World Model often needs context length > 1.

        world_output = None
        if context_mask is not None and target_mask is not None:
            world_output = self.world_model(
                ssm_output,
                context_mask=context_mask,
                target_mask=target_mask,
            )
        elif seq_len > 1:
            # Default: use first half as context, predict second half
            context_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=x.device)
            context_mask[:, : seq_len // 2] = True
            target_mask = ~context_mask

            world_output = self.world_model(
                ssm_output,
                context_mask=context_mask,
                target_mask=target_mask,
            )

        # Stage 3: Neuro-symbolic reasoning
        # Optimization: Use the last token's state for the query instead of mean pooling
        # This preserves the specific context at the point of reasoning
        if seq_len == 1:
            query = ssm_output.squeeze(1)  # (B, D)
        else:
            query = ssm_output[:, -1, :]  # (B, D)

        reasoning_output = self.reasoner(query, context=ssm_output)

        # Stage 4: Energy-based refinement
        energy_output = self.energy_module(ssm_output)
        refined = energy_output["output"]

        # Stage 5: Causal integration
        causal_output = self.causal_engine(refined)

        # Integrate all streams
        # Expand reasoning output to sequence length
        if seq_len == 1:
            reasoning_expanded = reasoning_output["answer"].unsqueeze(1)
        else:
            reasoning_expanded = reasoning_output["answer"].unsqueeze(1).expand(-1, seq_len, -1)

        integrated = self.integration(
            torch.cat(
                [
                    refined,
                    ssm_output,
                    reasoning_expanded,
                ],
                dim=-1,
            )
        )

        # Generate outputs
        lm_logits = self.lm_head(integrated)
        regression_output = self.regression_head(integrated)
        confidence = self.confidence_head(integrated.mean(dim=1))

        result = {
            "logits": lm_logits,
            "hidden_states": integrated,
            "regression": regression_output,
            "confidence": confidence.squeeze(-1),
            "energy": energy_output["energy"],
            "iterations": energy_output["iterations"],
            "state": new_state,
        }

        if return_all:
            result.update(
                {
                    "ssm_output": ssm_output,
                    "world_model": world_output,
                    "reasoning": reasoning_output,
                    "energy_output": energy_output,
                    "causal_output": causal_output,
                }
            )

        return result

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively using O(N) cached state.

        Args:
            prompt: (B, L)

        Returns:
            Generated sequence (B, L + max_new_tokens)
        """
        self.eval()

        # Initial forward to prefill cache
        with torch.no_grad():
            outputs = self.forward(prompt, modality="token")
            state = outputs["state"]

            # Get last token for next step
            next_token = prompt[:, -1].unsqueeze(-1)  # (B, 1)
            generated = prompt

            for _ in range(max_new_tokens):
                # Forward single step with state
                outputs = self.forward(next_token, modality="token", state=state)

                state = outputs["state"]
                logits = outputs["logits"][:, -1, :] / temperature
                confidence = outputs["confidence"]

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float("inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float("inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if confidence drops significantly (uncertainty-aware stopping)
                # Note: confidence is computed on the single step here
                if confidence.mean() < 0.1:
                    break

        return generated

    def reason(
        self,
        query: torch.Tensor,
        knowledge_base: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform explicit reasoning on a query.

        This provides:
        - Answer with confidence
        - Proof trace (explainable)
        - Grounding in knowledge
        """
        embedded = self.embedding(query, modality="continuous")

        # Process through state space
        ssm_output, _ = self.state_space(embedded)

        # Reason
        query_vec = ssm_output.mean(dim=1)
        result = self.reasoner(query_vec, context=knowledge_base)

        return result

    def imagine(
        self,
        context: torch.Tensor,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """
        Imagine future states using world model.

        This enables planning by predicting what future representations
        might look like without actual observation.
        """
        embedded = self.embedding(context, modality="continuous")
        ssm_output, _ = self.state_space(embedded)

        predictions = self.world_model.predict(ssm_output, n_steps=n_steps)
        return predictions

    def intervene(
        self,
        observation: torch.Tensor,
        intervention: Tuple[int, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict effect of causal intervention.

        This answers "what would happen if we set variable X to value v?"
        """
        embedded = self.embedding(observation, modality="continuous")
        ssm_output, _ = self.state_space(embedded)

        result = self.causal_engine(ssm_output, intervention=intervention)
        return result


def create_nexus_model(
    size: str = "base",
    **kwargs,
) -> NEXUSCore:
    """
    Factory function to create NEXUS models of different sizes.

    Args:
        size: Model size - "small", "base", "large", "xl"
        **kwargs: Override specific config parameters

    Returns:
        Configured NEXUSCore model
    """
    size_configs = {
        "small": {
            "d_model": 256,
            "d_latent": 128,
            "ssm_n_layers": 4,
            "n_heads": 4,
        },
        "base": {
            "d_model": 512,
            "d_latent": 256,
            "ssm_n_layers": 8,
            "n_heads": 8,
        },
        "large": {
            "d_model": 1024,
            "d_latent": 512,
            "ssm_n_layers": 16,
            "n_heads": 16,
        },
        "xl": {
            "d_model": 2048,
            "d_latent": 1024,
            "ssm_n_layers": 24,
            "n_heads": 32,
        },
    }

    config_dict = size_configs.get(size, size_configs["base"])
    config_dict.update(kwargs)

    config = NEXUSConfig(**config_dict)
    model = NEXUSCore(config)

    return model
