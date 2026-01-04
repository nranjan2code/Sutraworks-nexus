"""
NEXUS: Neural EXploratory Unified Synthesis
=============================================

A next-generation AI architecture that goes beyond Transformers and LLMs by combining:
1. Selective State Space Models (linear-time sequence processing)
2. Joint Embedding Predictive Architecture (abstract world modeling)
3. Neuro-Symbolic Integration (grounded reasoning)
4. Energy-Based Adaptive Computation (efficient inference)
5. Causal World Modeling (predictive planning)

Copyright 2026 - Research Implementation
"""

__version__ = "0.1.0"
__author__ = "NEXUS Research Team"

from nexus.core.nexus_core import NEXUSCore
from nexus.core.world_model import HierarchicalWorldModel
from nexus.core.reasoning import NeuroSymbolicReasoner
from nexus.core.state_space import SelectiveStateSpace
from nexus.core.energy import AdaptiveEnergyModule
from nexus.core.causal import CausalInferenceEngine

__all__ = [
    "NEXUSCore",
    "HierarchicalWorldModel",
    "NeuroSymbolicReasoner",
    "SelectiveStateSpace",
    "AdaptiveEnergyModule",
    "CausalInferenceEngine",
]
