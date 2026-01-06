"""
NEXUS Living System - Unified Learn + Respond Interface
========================================================

The "Living" wrapper that makes NEXUS a continuously evolving entity:
- Responds to queries while learning from every interaction
- Never hallucinates - refuses politely when uncertain
- Evolves continuously through experience (no stages or labels)
- Tracks knowledge accumulation and calibrates confidence

NOW WITH LAYER-FREE ARCHITECTURE:
- FlowingNEXUS: Computation flows to equilibrium, no discrete layers
- Depth emerges from input complexity, not architecture
- Continuous dynamics replace stacked transformations

This is the primary interface for interacting with NEXUS as a living system.

Usage:
------
```python
from nexus.core import create_living_nexus

# Traditional layered NEXUS
nexus = create_living_nexus(size="small", architecture="layered")

# NEW: Layer-free flowing NEXUS (recommended)
nexus = create_living_nexus(size="small", architecture="flowing")

# Interact - it learns and responds simultaneously
result = nexus.interact(query_batch)

if result.responded:
    print("Answer:", result.logits)
    print(f"Flow depth: {result.flow_depth}")  # Emergent depth!
else:
    print("NEXUS: I don't know enough about this yet.")

# Check its evolution
print(nexus.get_status())
# {'total_interactions': 1523, 'average_flow_depth': 12.3, ...}
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from nexus.core.nexus_core import NEXUSCore
from nexus.core.lifecycle import (
    LifecycleConfig,
    LifecycleManager,
    UncertaintyGate,
    RefusalGenerator,
)
from nexus.training.continual import ContinualConfig, ContinualLearner
from nexus.training.trainer import TrainingConfig

# Import layer-free architecture
from nexus.core.flowing import FlowingNEXUS, FlowingConfig, create_flowing_nexus


@dataclass
class LivingConfig:
    """Configuration for Living NEXUS system."""
    
    # Lifecycle settings
    lifecycle: LifecycleConfig = None
    
    # Continual learning settings
    continual: ContinualConfig = None
    training: TrainingConfig = None
    
    # Response settings
    min_confidence_to_respond: float = 0.5  # Override by lifecycle stage
    use_lifecycle_threshold: bool = True     # Use stage-based thresholds
    
    # Refusal messages (token IDs would be set by tokenizer)
    polite_refusal_prefix: str = "I don't know enough about this yet. "
    
    # Learning triggers
    learn_from_every_interaction: bool = True
    min_batch_size_to_learn: int = 1
    
    # Safety
    max_response_length: int = 2048
    block_harmful_content: bool = True
    
    def __post_init__(self):
        if self.lifecycle is None:
            self.lifecycle = LifecycleConfig()
        if self.continual is None:
            self.continual = ContinualConfig()
        if self.training is None:
            self.training = TrainingConfig()


@dataclass
class InteractionResult:
    """Result of an interaction with Living NEXUS."""
    
    # Response
    logits: torch.Tensor                    # Output logits
    hidden_states: torch.Tensor             # Hidden representations
    
    # Confidence & decision
    confidence: float                       # Model's confidence in response
    uncertainty: float                      # Uncertainty estimate
    responded: bool                         # True if answered, False if refused
    
    # Learning
    learned: bool                           # Whether learning occurred
    learning_metrics: Dict[str, float]      # Loss, etc. if learned
    
    # Evolution (continuous, no stages)
    experience_factor: float                # 0→1 continuous experience measure
    threshold_used: float                   # Confidence threshold used
    
    # Flow metrics (layer-free architecture)
    flow_depth: Optional[int] = None        # Emergent depth (iterations to equilibrium)
    converged: Optional[bool] = None        # Whether equilibrium was reached
    flow_energy: Optional[float] = None     # Final energy (residual)
    
    # Metadata
    interaction_id: int = 0                 # Total interactions so far


class LivingNEXUS(nn.Module):
    """
    Living NEXUS - A continuously evolving AI system.
    
    Core Principles:
    1. NEVER HALLUCINATE - Refuse politely when uncertain
    2. LEARN CONTINUOUSLY - Every interaction is a learning opportunity
    3. EVOLVE ORGANICALLY - No stages, no labels, just smooth growth
    4. KNOW YOUR LIMITS - Track what you know and don't know
    
    Now supports TWO architectures:
    - "layered": Traditional stacked layers (NEXUSCore)
    - "flowing": Layer-free equilibrium dynamics (FlowingNEXUS)
    
    The flowing architecture embodies the philosophy:
    "Growth is not a ladder with rungs to climb.
     It is water finding its level."
    """
    
    def __init__(
        self,
        model: Union[NEXUSCore, FlowingNEXUS],
        config: LivingConfig = None,
        architecture: str = "layered",
    ):
        super().__init__()
        self.config = config or LivingConfig()
        self.architecture = architecture
        
        # Core model (either layered or flowing)
        self.model = model
        
        # Get dimensions based on architecture
        if isinstance(model, FlowingNEXUS):
            d_model = model.config.d_model
            vocab_size = model.config.vocab_size
            self.architecture = "flowing"
        else:
            d_model = model.config.d_model
            vocab_size = model.config.vocab_size
            self.architecture = "layered"
        
        # Uncertainty gating (anti-hallucination)
        self.uncertainty_gate = UncertaintyGate(d_model)
        
        # Refusal generation
        self.refusal_generator = RefusalGenerator(d_model, vocab_size)
        
        # Lifecycle management
        self.lifecycle = LifecycleManager(self.config.lifecycle)
        
        # Continual learner (for layered architecture)
        if self.architecture == "layered":
            self.learner = ContinualLearner(
                model=model,
                train_config=self.config.training,
                continual_config=self.config.continual,
            )
        else:
            self.learner = None  # Flowing NEXUS has built-in adaptation
        
        # Pending samples for batch learning
        self._pending_samples: List[Dict[str, torch.Tensor]] = []
        
        # Flow metrics tracking (for flowing architecture)
        self._total_flow_steps = 0
        self._flow_count = 0
    
    @property
    def confidence_threshold(self) -> float:
        """Current confidence threshold for responding."""
        if self.config.use_lifecycle_threshold:
            return self.lifecycle.confidence_threshold
        return self.config.min_confidence_to_respond
    
    def interact(
        self,
        batch: Dict[str, torch.Tensor],
        modality: str = "token",
        learn: bool = None,
        domain: Optional[str] = None,
    ) -> InteractionResult:
        """
        Primary interaction method - respond and optionally learn.
        
        This is how you talk to Living NEXUS. It will:
        1. Process your query (via layers OR continuous flow)
        2. Decide if it's confident enough to respond
        3. Either respond or politely refuse
        4. Learn from the interaction (if enabled)
        
        Args:
            batch: Input batch with 'input_ids' or 'features'
            modality: Input type ("token" or "continuous")
            learn: Override learning (None = use config)
            domain: Optional domain label for knowledge tracking
            
        Returns:
            InteractionResult with response, confidence, and metadata
        """
        # Determine if we should learn
        should_learn = learn if learn is not None else self.config.learn_from_every_interaction
        
        # Get response from model (different paths for different architectures)
        flow_depth = None
        converged = None
        flow_energy = None
        
        with torch.set_grad_enabled(should_learn):
            if self.architecture == "flowing":
                # Layer-free: flow to equilibrium
                x = batch.get("input_ids", batch.get("features"))
                outputs = self.model(x, modality=modality)
                
                # Extract flow metrics
                flow_depth = outputs.get("flow_steps")
                converged = outputs.get("converged")
                flow_energy = outputs.get("final_energy")
                if isinstance(flow_energy, torch.Tensor):
                    flow_energy = flow_energy.item()
                    
                # Track flow statistics
                if flow_depth is not None:
                    self._total_flow_steps += flow_depth
                    self._flow_count += 1
            else:
                # Traditional layered architecture
                outputs = self.learner.respond(batch, modality=modality)
        
        logits = outputs["logits"]
        hidden_states = outputs["hidden_states"]
        
        # Compute confidence and decide whether to respond
        confidence, uncertainty, should_respond = self.uncertainty_gate(
            hidden_states,
            threshold=self.confidence_threshold,
        )
        
        # Average across batch for scalar metrics
        conf_scalar = confidence.mean().item()
        unc_scalar = uncertainty.mean().item()
        responded = should_respond.all().item()
        
        # If not confident, generate refusal
        if not responded:
            batch_size, seq_len = logits.shape[:2]
            refusal_logits = self.refusal_generator(
                batch_size, seq_len, logits.device
            )
            # Blend: use refusal for low-confidence samples
            respond_mask = should_respond.view(-1, 1, 1).float()
            logits = logits * respond_mask + refusal_logits * (1 - respond_mask)
        
        # Learn from this interaction
        learning_metrics = {}
        learned = False
        
        if should_learn and self.architecture == "layered":
            # Add to pending samples
            self._pending_samples.append(batch)
            
            # Learn if we have enough samples
            if len(self._pending_samples) >= self.config.min_batch_size_to_learn:
                learning_metrics = self.learner.observe_and_learn(self._pending_samples)
                self._pending_samples = []
                learned = True
                self.lifecycle.record_learning()
        
        # Record interaction in lifecycle
        self.lifecycle.record_interaction(
            confidence=conf_scalar,
            responded=responded,
            domain=domain,
        )
        
        return InteractionResult(
            logits=logits,
            hidden_states=hidden_states,
            confidence=conf_scalar,
            uncertainty=unc_scalar,
            responded=responded,
            learned=learned,
            learning_metrics=learning_metrics,
            experience_factor=self.lifecycle._experience_factor(),
            threshold_used=self.confidence_threshold,
            flow_depth=flow_depth,
            converged=converged,
            flow_energy=flow_energy,
            interaction_id=self.lifecycle.state.total_interactions,
        )
    
    def ask(
        self,
        batch: Dict[str, torch.Tensor],
        modality: str = "token",
    ) -> Tuple[torch.Tensor, bool, float]:
        """
        Simple ask interface - get response or refusal.
        
        Returns:
            (logits, responded, confidence)
        """
        result = self.interact(batch, modality=modality, learn=False)
        return result.logits, result.responded, result.confidence
    
    def teach(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Explicitly teach NEXUS with labeled samples.
        
        Use this for supervised learning with ground truth.
        """
        metrics = self.learner.observe_and_learn(samples)
        for _ in samples:
            self.lifecycle.record_learning()
        return metrics
    
    def knows_about(self, domain: str) -> float:
        """
        Check NEXUS's confidence in a domain.
        
        Returns confidence score (0-1) for the domain.
        """
        return self.lifecycle.state.domain_confidence.get(domain, 0.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of Living NEXUS."""
        status = self.lifecycle.get_status()
        status.update({
            "pending_samples": len(self._pending_samples),
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "architecture": self.architecture,
        })
        
        # Add flow metrics for flowing architecture
        if self.architecture == "flowing" and self._flow_count > 0:
            status.update({
                "average_flow_depth": self._total_flow_steps / self._flow_count,
                "total_flow_steps": self._total_flow_steps,
                "flow_interactions": self._flow_count,
            })
            
        return status
    
    def save(self, path: str) -> None:
        """Save Living NEXUS state (model + lifecycle)."""
        torch.save({
            "model_state": self.model.state_dict(),
            "uncertainty_gate_state": self.uncertainty_gate.state_dict(),
            "refusal_generator_state": self.refusal_generator.state_dict(),
            "lifecycle_state": self.lifecycle.save_state(),
            "config": self.config,
        }, path)
    
    def load(self, path: str) -> None:
        """Load Living NEXUS state."""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.uncertainty_gate.load_state_dict(checkpoint["uncertainty_gate_state"])
        self.refusal_generator.load_state_dict(checkpoint["refusal_generator_state"])
        self.lifecycle.load_state(checkpoint["lifecycle_state"])
    
    def forward(
        self,
        x: torch.Tensor,
        modality: str = "token",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass (delegates to model).
        
        For the full living experience, use `interact()` instead.
        """
        return self.model(x, modality=modality, **kwargs)


def create_living_nexus(
    size: str = "small",
    architecture: str = "flowing",  # Default to layer-free!
    start_fresh: bool = True,
    checkpoint_path: Optional[str] = None,
    **config_overrides,
) -> LivingNEXUS:
    """
    Factory function to create a Living NEXUS.
    
    Args:
        size: Model size ("small", "base", "large")
        architecture: "flowing" (layer-free, recommended) or "layered" (traditional)
        start_fresh: If True, start as newborn; else load checkpoint
        checkpoint_path: Path to load existing NEXUS
        **config_overrides: Override config parameters
        
    Returns:
        A Living NEXUS ready to learn and respond
        
    Note:
        "flowing" architecture is now the default - it embodies the philosophy
        of continuous evolution with emergent depth. Use "layered" only for
        comparison or backward compatibility.
    """
    # Create config
    config = LivingConfig(**config_overrides)
    
    if architecture == "flowing":
        # Layer-free architecture (recommended)
        model = create_flowing_nexus(size=size)
    else:
        # Traditional layered architecture
        from nexus.core.nexus_core import create_nexus_model
        model = create_nexus_model(size=size)
    
    # Create living system
    nexus = LivingNEXUS(model, config, architecture=architecture)
    
    # Load checkpoint if provided
    if not start_fresh and checkpoint_path:
        nexus.load(checkpoint_path)
    
    return nexus
