"""
NEXUS Type Definitions
======================

Centralized type definitions for NEXUS architecture.
Provides type aliases and protocols for consistent typing across modules.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor


# =============================================================================
# Core Tensor Types
# =============================================================================

# Generic tensor type variable
T = TypeVar("T", bound=Tensor)

# Common tensor shapes (annotated for documentation)
BatchTensor = Tensor  # Shape: (batch_size, ...)
SeqTensor = Tensor  # Shape: (batch_size, seq_len, ...)
HiddenTensor = Tensor  # Shape: (batch_size, seq_len, d_model)
LogitsTensor = Tensor  # Shape: (batch_size, seq_len, vocab_size)
EnergyTensor = Tensor  # Shape: (batch_size,) - scalar per sample
ConfidenceTensor = Tensor  # Shape: (batch_size,) - [0, 1] values


# =============================================================================
# Configuration Types
# =============================================================================

class ModelSizeConfig(TypedDict):
    """Configuration for model size presets."""
    d_model: int
    d_latent: int
    n_heads: int
    memory_size: int


ModelSize = Union[str, ModelSizeConfig]  # "small", "base", "large", or custom dict


# =============================================================================
# Output Types
# =============================================================================

class FlowingOutput(TypedDict, total=False):
    """Output dictionary from FlowingNEXUS forward pass."""
    logits: LogitsTensor
    hidden_states: HiddenTensor
    regression: HiddenTensor
    confidence: ConfidenceTensor
    flow_steps: int
    converged: bool
    final_energy: EnergyTensor
    trajectory: Optional[List[HiddenTensor]]
    memory: Optional[Tensor]


class ReasoningOutput(TypedDict, total=False):
    """Output dictionary from reasoning operations."""
    answer: HiddenTensor
    confidence: ConfidenceTensor
    proof_score: Tensor
    proof_trace: List[Dict[str, Any]]
    retrieval_weights: Tensor


class EnergyOutput(TypedDict, total=False):
    """Output dictionary from energy-based modules."""
    output: HiddenTensor
    energy: EnergyTensor
    confidence: ConfidenceTensor
    iterations: Tensor
    expected_iterations: Tensor
    complexity: Tensor
    contrastive_loss: Optional[Tensor]
    contrastive_accuracy: Optional[Tensor]


class CausalOutput(TypedDict, total=False):
    """Output dictionary from causal inference modules."""
    adjacency: Tensor
    interventional_output: Optional[HiddenTensor]
    acyclicity_loss: Tensor
    sparsity_loss: Tensor


class WorldModelOutput(TypedDict, total=False):
    """Output dictionary from world model."""
    predicted: HiddenTensor
    target: HiddenTensor
    context: HiddenTensor
    multi_scale: List[HiddenTensor]
    target_mask: Tensor


class LossOutput(TypedDict, total=False):
    """Output dictionary from loss computation."""
    total_loss: Tensor
    lm_loss: Optional[Tensor]
    world_model_loss: Optional[Tensor]
    reasoning_loss: Optional[Tensor]
    energy_loss: Optional[Tensor]
    causal_loss: Optional[Tensor]
    convergence_loss: Optional[Tensor]
    jac_reg: Optional[Tensor]


class ContinualMetrics(TypedDict, total=False):
    """Metrics from continual learning."""
    total_loss: float
    lm_loss: float
    avg_flow_steps: float
    buffer_size: int


# =============================================================================
# Batch Types
# =============================================================================

class TrainingBatch(TypedDict, total=False):
    """Standard training batch format."""
    input_ids: Tensor  # (batch, seq_len)
    labels: Tensor  # (batch, seq_len)
    attention_mask: Optional[Tensor]  # (batch, seq_len)
    context_mask: Optional[Tensor]  # (batch, seq_len)
    target_mask: Optional[Tensor]  # (batch, seq_len)
    features: Optional[HiddenTensor]  # For continuous input


class InferenceBatch(TypedDict, total=False):
    """Inference batch format."""
    input_ids: Optional[Tensor]
    features: Optional[HiddenTensor]
    attention_mask: Optional[Tensor]


# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class NEXUSModel(Protocol):
    """Protocol for NEXUS model implementations."""
    
    def forward(
        self,
        x: Tensor,
        modality: str = "token",
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        """Forward pass returning output dictionary."""
        ...
    
    def to(self, device: Union[str, torch.device]) -> "NEXUSModel":
        """Move model to device."""
        ...


@runtime_checkable
class HasWorldModel(Protocol):
    """Protocol for models with world model component."""
    
    world_model: Any  # HierarchicalWorldModel
    
    def imagine(
        self,
        context: Tensor,
        n_steps: int = 5,
    ) -> Tensor:
        """Imagination/prediction capability."""
        ...


@runtime_checkable
class HasReasoner(Protocol):
    """Protocol for models with reasoning component."""
    
    def reason(
        self,
        query: Tensor,
        context: Optional[Tensor] = None,
    ) -> ReasoningOutput:
        """Reasoning capability."""
        ...


@runtime_checkable
class HasCausal(Protocol):
    """Protocol for models with causal inference."""
    
    def intervene(
        self,
        x: Tensor,
        intervention_idx: int,
        intervention_value: float,
    ) -> Tensor:
        """Causal intervention capability."""
        ...


@runtime_checkable
class Learnable(Protocol):
    """Protocol for continual learners."""
    
    def observe_and_learn(
        self,
        new_samples: List[Dict[str, Tensor]],
    ) -> ContinualMetrics:
        """Learn from new samples."""
        ...
    
    def respond(
        self,
        batch: Dict[str, Tensor],
        modality: str = "token",
    ) -> Dict[str, Tensor]:
        """Generate response."""
        ...


# =============================================================================
# Callback Types
# =============================================================================

StepCallback = Callable[[], None]
LossCallback = Callable[[LossOutput], None]
CheckpointCallback = Callable[[int, Dict[str, Any]], None]


# =============================================================================
# Device Types
# =============================================================================

DeviceType = Union[str, torch.device]


def get_device(tensor_or_device: Union[Tensor, DeviceType]) -> torch.device:
    """Extract device from tensor or device specification."""
    if isinstance(tensor_or_device, Tensor):
        return tensor_or_device.device
    elif isinstance(tensor_or_device, torch.device):
        return tensor_or_device
    else:
        return torch.device(tensor_or_device)


# =============================================================================
# Utility Types
# =============================================================================

# Parameter group for optimizer
class ParamGroup(TypedDict, total=False):
    params: List[torch.nn.Parameter]
    lr: float
    weight_decay: float


# Cache for SSM recurrence
SSMCache = Tuple[Tensor, Tensor]  # (ssm_state, conv_state)


# Equilibrium solver info
class EquilibriumInfo(TypedDict):
    iterations: int
    converged: bool
    residual: Tensor
    trajectory: Optional[List[Tensor]]


__all__ = [
    # Tensor types
    "T",
    "BatchTensor",
    "SeqTensor", 
    "HiddenTensor",
    "LogitsTensor",
    "EnergyTensor",
    "ConfidenceTensor",
    # Config types
    "ModelSizeConfig",
    "ModelSize",
    # Output types
    "FlowingOutput",
    "ReasoningOutput",
    "EnergyOutput",
    "CausalOutput",
    "WorldModelOutput",
    "LossOutput",
    "ContinualMetrics",
    # Batch types
    "TrainingBatch",
    "InferenceBatch",
    # Protocols
    "NEXUSModel",
    "HasWorldModel",
    "HasReasoner",
    "HasCausal",
    "Learnable",
    # Callback types
    "StepCallback",
    "LossCallback",
    "CheckpointCallback",
    # Device types
    "DeviceType",
    "get_device",
    # Utility types
    "ParamGroup",
    "SSMCache",
    "EquilibriumInfo",
]
