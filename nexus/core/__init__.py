"""NEXUS Core Module - Contains fundamental building blocks."""

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
