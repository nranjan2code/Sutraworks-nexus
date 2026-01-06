"""
NEXUS Lifecycle - Continuous Organic Evolution
==============================================

A living system evolves naturally through experience without labels.
We don't categorize or stage growth - the system simply IS what its
accumulated experience has shaped it to be.

Core Principles:
1. Evolution is CONTINUOUS - no discrete stages or labels
2. Confidence and learning emerge organically from experience
3. NEVER hallucinate - refuse politely when uncertain
4. Wisdom grows naturally, not by crossing arbitrary thresholds

Philosophy:
    Growth is not a ladder with rungs to climb.
    It is water finding its level.
    The system doesn't "become" something new -
    it continuously IS, shaped by all it has experienced.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LifecycleConfig:
    """
    Configuration for continuous lifecycle evolution.
    
    No stages, no labels - just smooth curves that emerge from experience.
    The system's behavior is a continuous function of its accumulated wisdom.
    """
    
    # Experience scaling - a soft reference point, not a boundary
    # The system never "arrives" - it just keeps evolving
    experience_scale: float = 1_000_000
    
    # Confidence threshold curve
    # When young: very cautious (high threshold)
    # With experience: knows its limits better (lower threshold)
    # threshold(exp) smoothly interpolates between these
    initial_confidence_threshold: float = 0.95   # Very cautious when new
    mature_confidence_threshold: float = 0.35    # Knows limits with experience
    confidence_curve_steepness: float = 3.0      # How quickly confidence develops
    
    # Learning rate curve
    # When young: absorb everything quickly
    # With experience: more selective, focused learning
    initial_lr_multiplier: float = 2.5   # Learn fast when new
    mature_lr_multiplier: float = 0.1    # Very selective when experienced
    learning_curve_steepness: float = 2.0
    
    # Domain-specific tracking
    track_domains: bool = True
    max_domains: int = 10000


@dataclass
class ExperienceState:
    """
    The accumulated experience of the system.
    
    Not "knowledge" in a static sense - but the living record
    of all interactions that have shaped this system.
    """
    
    total_interactions: int = 0
    total_learning_steps: int = 0
    total_refusals: int = 0      # Times it wisely said "I don't know"
    total_responses: int = 0     # Times it felt confident to respond
    
    # Domain-specific confidence (emerges from experience)
    domain_confidence: Dict[str, float] = field(default_factory=dict)
    domain_interactions: Dict[str, int] = field(default_factory=dict)
    
    # Recent history for calibration
    recent_confidences: List[float] = field(default_factory=list)
    recent_outcomes: List[bool] = field(default_factory=list)
    
    # When this instance came into being
    born_at: float = field(default_factory=time.time)
    
    def age(self) -> float:
        """Time since creation in seconds."""
        return time.time() - self.born_at
    
    def wisdom_ratio(self) -> float:
        """
        How often the system chose silence over guessing.
        
        A wise system knows when to say "I don't know".
        This isn't a failure rate - it's a wisdom indicator.
        """
        total = self.total_refusals + self.total_responses
        if total == 0:
            return 1.0  # No experience yet = maximum caution
        return self.total_refusals / total


class UncertaintyGate(nn.Module):
    """
    The anti-hallucination guardian.
    
    When confidence is below threshold, the system will gracefully
    decline to answer rather than fabricate a response.
    
    This is not a limitation - it is wisdom.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Confidence estimation - how sure are we?
        self.confidence_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # Uncertainty estimation - what don't we know?
        self.uncertainty_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus(),  # Always positive
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate confidence and decide whether to respond.
        
        Args:
            hidden_states: Representations (batch, seq_len, d_model) or (batch, d_model)
            threshold: Confidence threshold - below this, we stay silent
            
        Returns:
            confidence: How confident we are (batch,)
            uncertainty: How uncertain we are (batch,)
            should_respond: Whether to respond or gracefully decline (batch,)
        """
        # Pool sequence if needed
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
            
        confidence = self.confidence_net(pooled).squeeze(-1)
        uncertainty = self.uncertainty_net(pooled).squeeze(-1)
        
        should_respond = confidence >= threshold
        
        return confidence, uncertainty, should_respond


class RefusalGenerator(nn.Module):
    """
    Generates graceful "I don't know" responses.
    
    Saying "I don't know" is not failure - it is honesty.
    This module creates responses that acknowledge uncertainty
    without pretending to knowledge we don't have.
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Learnable "I don't know" representation
        self.refusal_embedding = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )
        
        # Project to vocabulary
        self.to_vocab = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate logits for a graceful refusal."""
        refusal = self.refusal_embedding.expand(batch_size, seq_len, -1)
        refusal = refusal.to(device)
        return self.to_vocab(refusal)


class LifecycleManager:
    """
    Manages the continuous evolution of the system.
    
    There are no stages. No labels. No announcements of "now I'm an adult".
    Just smooth, continuous adaptation based on accumulated experience.
    
    The confidence threshold and learning rate are continuous functions
    of experience - they flow naturally, not in steps.
    """
    
    def __init__(self, config: Optional[LifecycleConfig] = None):
        self.config = config or LifecycleConfig()
        self.state = ExperienceState()
        
    def _experience_factor(self) -> float:
        """
        A smooth 0→1 factor representing accumulated experience.
        
        Uses a soft exponential curve so there are no hard transitions.
        Approaches but never quite reaches 1.0 - always room to grow.
        """
        x = self.state.total_interactions / self.config.experience_scale
        # Smooth curve: 1 - e^(-kx) approaches 1 asymptotically
        return 1.0 - math.exp(-self.config.confidence_curve_steepness * x)
    
    def _learning_factor(self) -> float:
        """Experience factor for learning rate curve (may have different steepness)."""
        x = self.state.total_interactions / self.config.experience_scale
        return 1.0 - math.exp(-self.config.learning_curve_steepness * x)
    
    @property
    def confidence_threshold(self) -> float:
        """
        Current confidence threshold - a continuous function of experience.
        
        Starts high (cautious) and smoothly decreases as wisdom accumulates.
        Never steps, never jumps - just flows.
        """
        factor = self._experience_factor()
        # Linear interpolation between initial (cautious) and mature (confident)
        return (
            self.config.initial_confidence_threshold * (1 - factor) +
            self.config.mature_confidence_threshold * factor
        )
    
    @property
    def learning_rate_multiplier(self) -> float:
        """
        Current learning rate multiplier - continuous function of experience.
        
        Starts high (absorb everything) and smoothly decreases (more selective).
        """
        factor = self._learning_factor()
        return (
            self.config.initial_lr_multiplier * (1 - factor) +
            self.config.mature_lr_multiplier * factor
        )
    
    def record_interaction(
        self,
        confidence: float,
        responded: bool,
        domain: Optional[str] = None,
        was_correct: Optional[bool] = None,
    ) -> None:
        """
        Record an interaction - every experience shapes the system.
        
        Args:
            confidence: How confident the system was
            responded: Whether it responded or wisely declined
            domain: Optional domain/topic identifier
            was_correct: If known, whether the response was correct
        """
        self.state.total_interactions += 1
        
        if responded:
            self.state.total_responses += 1
        else:
            self.state.total_refusals += 1
            
        # Track recent confidence for calibration
        self.state.recent_confidences.append(confidence)
        if len(self.state.recent_confidences) > 1000:
            self.state.recent_confidences.pop(0)
            
        # Track outcomes if known
        if was_correct is not None:
            self.state.recent_outcomes.append(was_correct)
            if len(self.state.recent_outcomes) > 1000:
                self.state.recent_outcomes.pop(0)
                
        # Track domain-specific confidence
        if domain and self.config.track_domains:
            if len(self.state.domain_confidence) >= self.config.max_domains:
                # At capacity - only update existing domains
                if domain not in self.state.domain_confidence:
                    return
                    
            if domain not in self.state.domain_interactions:
                self.state.domain_interactions[domain] = 0
                self.state.domain_confidence[domain] = 0.5  # Start neutral
                
            self.state.domain_interactions[domain] += 1
            
            # Exponential moving average for domain confidence
            alpha = 0.1
            self.state.domain_confidence[domain] = (
                (1 - alpha) * self.state.domain_confidence[domain] +
                alpha * confidence
            )
    
    def record_learning(self) -> None:
        """Record a learning step."""
        self.state.total_learning_steps += 1
    
    def get_domain_confidence(self, domain: str) -> float:
        """Get confidence for a specific domain, or default if unknown."""
        return self.state.domain_confidence.get(domain, 0.5)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status - no stage labels, just metrics.
        
        Returns experience metrics and current behavior parameters.
        """
        return {
            "age_seconds": self.state.age(),
            "total_interactions": self.state.total_interactions,
            "total_learning_steps": self.state.total_learning_steps,
            "responses": self.state.total_responses,
            "refusals": self.state.total_refusals,
            "wisdom_ratio": self.state.wisdom_ratio(),
            "confidence_threshold": self.confidence_threshold,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "experience_factor": self._experience_factor(),
            "known_domains": len(self.state.domain_confidence),
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "total_interactions": self.state.total_interactions,
            "total_learning_steps": self.state.total_learning_steps,
            "total_refusals": self.state.total_refusals,
            "total_responses": self.state.total_responses,
            "domain_confidence": dict(self.state.domain_confidence),
            "domain_interactions": dict(self.state.domain_interactions),
            "born_at": self.state.born_at,
        }
    
    def load_state(self, data: Dict[str, Any]) -> None:
        """Load state from checkpoint - resume evolution where we left off."""
        self.state.total_interactions = data.get("total_interactions", 0)
        self.state.total_learning_steps = data.get("total_learning_steps", 0)
        self.state.total_refusals = data.get("total_refusals", 0)
        self.state.total_responses = data.get("total_responses", 0)
        self.state.domain_confidence = data.get("domain_confidence", {})
        self.state.domain_interactions = data.get("domain_interactions", {})
        self.state.born_at = data.get("born_at", time.time())
