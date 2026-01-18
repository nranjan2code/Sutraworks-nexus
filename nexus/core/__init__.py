"""NEXUS Core Module - Contains fundamental building blocks.

Now with LAYER-FREE architecture:
- FlowingNEXUS: Computation flows to equilibrium, no discrete layers
- EquilibriumCore: Single dynamics function iterated to convergence
- ContinuousSSM: State space model with emergent depth

The layer-free architecture embodies:
"Growth is not a ladder with rungs to climb.
 It is water finding its level."
"""

from nexus.core.nexus_core import NEXUSCore, NEXUSConfig, create_nexus_model
from nexus.core.world_model import HierarchicalWorldModel, WorldModelConfig
from nexus.core.reasoning import NeuroSymbolicReasoner, ReasoningConfig
from nexus.core.state_space import SelectiveStateSpace, StateSpaceConfig
from nexus.core.energy import AdaptiveEnergyModule, EnergyConfig
from nexus.core.causal import CausalInferenceEngine, CausalConfig
from nexus.core.lifecycle import (
    LifecycleManager,
    LifecycleConfig,
    ExperienceState,
    UncertaintyGate,
)
from nexus.core.living import LivingNEXUS, LivingConfig, create_living_nexus

# Layer-free architecture (NEW)
from nexus.core.equilibrium import (
    EquilibriumCore,
    EquilibriumConfig,
    ContinuousDynamics,
    FlowField,
    NeuralODE,
)
from nexus.core.continuous_ssm import (
    ContinuousSSM,
    ContinuousSSMConfig,
    HierarchicalContinuousSSM,
)
from nexus.core.flowing import (
    FlowingNEXUS,
    FlowingConfig,
    UnifiedDynamics,
    create_flowing_nexus,
    create_living_flowing_nexus,
    DynamicsDivergenceError,
)

# Type definitions
from nexus.core.types import (
    # Tensor types
    BatchTensor,
    SeqTensor,
    HiddenTensor,
    LogitsTensor,
    EnergyTensor,
    ConfidenceTensor,
    # Output types
    FlowingOutput,
    ReasoningOutput,
    EnergyOutput,
    CausalOutput,
    WorldModelOutput,
    LossOutput,
    # Batch types
    TrainingBatch,
    InferenceBatch,
    # Protocols
    NEXUSModel,
    HasWorldModel,
    HasReasoner,
    HasCausal,
    Learnable,
    # Utility types
    DeviceType,
    get_device,
    SSMCache,
    EquilibriumInfo,
)

__all__ = [
    # Traditional Core (layered)
    "NEXUSCore",
    "NEXUSConfig",
    "create_nexus_model",
    
    # Layer-Free Core (flowing) - RECOMMENDED
    "FlowingNEXUS",
    "FlowingConfig",
    "UnifiedDynamics",
    "create_flowing_nexus",
    "create_living_flowing_nexus",
    "DynamicsDivergenceError",
    
    # Equilibrium components
    "EquilibriumCore",
    "EquilibriumConfig",
    "ContinuousDynamics",
    "FlowField",
    "NeuralODE",
    
    # Continuous SSM
    "ContinuousSSM",
    "ContinuousSSMConfig",
    "HierarchicalContinuousSSM",
    
    # Modules (shared)
    "HierarchicalWorldModel",
    "WorldModelConfig",
    "NeuroSymbolicReasoner",
    "ReasoningConfig",
    "SelectiveStateSpace",
    "StateSpaceConfig",
    "AdaptiveEnergyModule",
    "EnergyConfig",
    "CausalInferenceEngine",
    "CausalConfig",
    
    # Lifecycle
    "LifecycleManager",
    "LifecycleConfig",
    "ExperienceState",
    "UncertaintyGate",
    
    # Living System
    "LivingNEXUS",
    "LivingConfig",
    "create_living_nexus",
    
    # Type definitions
    "BatchTensor",
    "SeqTensor",
    "HiddenTensor",
    "LogitsTensor",
    "EnergyTensor",
    "ConfidenceTensor",
    "FlowingOutput",
    "ReasoningOutput",
    "EnergyOutput",
    "CausalOutput",
    "WorldModelOutput",
    "LossOutput",
    "TrainingBatch",
    "InferenceBatch",
    "NEXUSModel",
    "HasWorldModel",
    "HasReasoner",
    "HasCausal",
    "Learnable",
    "DeviceType",
    "get_device",
    "SSMCache",
    "EquilibriumInfo",
]
