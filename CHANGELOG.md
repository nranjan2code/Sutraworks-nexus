# NEXUS Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-01-18

### Added

#### Core Architecture
- **True Parallel Scan**: Implemented Blelloch's parallel associative scan algorithm for O(log n) parallel depth
  - Replaced sequential scan with true parallel implementation in `SelectiveStateSpace`
  - Added `_associative_scan()` method using up-sweep/down-sweep phases
  - Proper padding to power-of-2 lengths for clean parallel execution
  
- **Gradient Checkpointing**: Added memory-efficient training for FlowingNEXUS
  - New config options: `gradient_checkpointing`, `checkpoint_every_n_steps`
  - Uses PyTorch's `torch.utils.checkpoint` for automatic recomputation
  - Reduces memory usage by ~60% for long flow sequences
  
- **Memory Management**: Fixed trajectory storage in FlowingNEXUS
  - Added `max_trajectory_length` config to limit trajectory memory
  - Detaches energy tensors to prevent gradient accumulation
  - Efficient trajectory sampling (keeps first, last, and evenly spaced states)

#### Training
- **FlowingContinualLearner**: New continual learning wrapper for layer-free architecture
  - Curriculum learning based on flow depth (input complexity)
  - Convergence-aware loss with bonus for faster equilibrium
  - Adaptive learning rate scaling based on sample complexity
  - Confidence-based replay buffer filtering
  - Jacobian regularization for stable dynamics
  
- **FlowingLoss**: Specialized loss function for flowing architecture
  - Language modeling loss with convergence bonus
  - Spectral norm regularization via power iteration
  - Jacobian penalty for contraction mapping guarantee

- **EMA Updates**: Added automatic EMA update callback in NEXUSTrainer
  - New config: `ema_enabled`, `ema_decay`, `ema_update_every_n_steps`
  - Automatically updates world model target encoder during training

#### Type System
- **Comprehensive Type Definitions**: New `nexus/core/types.py` module
  - TypedDict definitions for all output formats
  - Protocol classes for model interfaces
  - Tensor type aliases for documentation
  - Utility types for callbacks and caching

### Fixed

- **Device Handling**: Standardized tensor device placement across all modules
  - Fixed `energy.py`: Ensure iteration tracking tensors on correct device
  - Fixed `reasoning.py`: Ensure fact bank moved to query device
  - Fixed `world_model.py`: Explicit dtype for target positions
  
- **Duplicate Endpoints**: Removed duplicate `/api/interact` endpoint definitions in server.py
  - Unified rate-limited and non-rate-limited paths
  - Cleaner conditional rate limiting logic

### Changed

- **FlowingConfig**: Added new memory optimization options
  - `gradient_checkpointing: bool = False`
  - `checkpoint_every_n_steps: int = 5`
  - `max_trajectory_length: int = 10`

- **TrainingConfig**: Added EMA configuration
  - `ema_enabled: bool = True`
  - `ema_decay: float = 0.996`
  - `ema_update_every_n_steps: int = 1`

### Deprecated

- None

### Removed

- Dead code in server.py (duplicate endpoint definitions)

### Security

- No security changes

## [2.1.0] - 2026-01-15

### Added

- Initial FlowingNEXUS layer-free architecture
- EquilibriumCore with implicit differentiation
- ContinuousSSM with emergent depth
- LivingNEXUS unified interface

## [2.0.0] - 2026-01-01

### Added

- NEXUSCore with integrated components
- SelectiveStateSpace (Mamba-inspired SSM)
- HierarchicalWorldModel (JEPA-style)
- NeuroSymbolicReasoner
- AdaptiveEnergyModule
- CausalInferenceEngine
- FastAPI production server

---

## Migration Guide

### From 2.1.x to 2.2.x

1. **FlowingNEXUS Users**: Enable gradient checkpointing for large models:
   ```python
   config = FlowingConfig(
       gradient_checkpointing=True,
       checkpoint_every_n_steps=5,
   )
   ```

2. **ContinualLearner Users**: Switch to FlowingContinualLearner for layer-free models:
   ```python
   # Old
   learner = ContinualLearner(nexus_core_model, train_config, continual_config)
   
   # New for FlowingNEXUS
   from nexus.training import FlowingContinualLearner, FlowingContinualConfig
   
   cont_cfg = FlowingContinualConfig(curriculum_enabled=True)
   learner = FlowingContinualLearner(flowing_model, train_config, cont_cfg)
   ```

3. **Type Hints**: Import from `nexus.core.types` for better IDE support:
   ```python
   from nexus.core import FlowingOutput, TrainingBatch
   
   def process_output(output: FlowingOutput) -> None:
       print(f"Flow steps: {output['flow_steps']}")
   ```

4. **Training with EMA**: EMA is now enabled by default:
   ```python
   # Disable if needed
   config = TrainingConfig(ema_enabled=False)
   ```
