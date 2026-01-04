"""
World Model Validation Tasks
=============================

Tasks to validate the JEPA-style world model component:

- Future State Prediction: Predict future representations from context
- Abstraction Learning: Learn hierarchical representations
- Counterfactual Imagination: "What would happen if..."

These tasks test the world model's ability to build predictive
representations without pixel/token-level reconstruction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import torch
from torch.utils.data import Dataset
import numpy as np


class WorldModelTaskType(Enum):
    """Types of world model validation tasks."""
    FUTURE_PREDICTION = "future_prediction"
    SEQUENCE_COMPLETION = "sequence_completion"
    STATE_TRANSITION = "state_transition"
    TRAJECTORY_PREDICTION = "trajectory_prediction"
    ABSTRACTION = "abstraction"


@dataclass
class WorldModelTaskConfig:
    """Configuration for world model validation tasks."""
    d_features: int = 64
    context_len: int = 32
    prediction_len: int = 16
    max_seq_len: int = 128
    
    # Dynamics parameters
    n_modes: int = 3  # Number of distinct dynamics patterns
    noise_std: float = 0.1


class DynamicsGenerator:
    """
    Generate sequences with known dynamics for world model validation.
    
    Supports multiple dynamics types:
    - Linear: x_{t+1} = Ax_t + noise
    - Oscillatory: Sinusoidal patterns
    - Switching: Mode-switching dynamics
    - Compositional: Combination of primitives
    """
    
    def __init__(self, d_features: int, dynamics_type: str = "linear"):
        self.d_features = d_features
        self.dynamics_type = dynamics_type
        
        # Initialize dynamics parameters
        if dynamics_type == "linear":
            # Random linear dynamics matrix (stable)
            A = np.random.randn(d_features, d_features) * 0.1
            # Make it stable by scaling eigenvalues
            eigenvalues, eigenvectors = np.linalg.eig(A)
            eigenvalues = eigenvalues / (np.abs(eigenvalues).max() + 0.1) * 0.95
            self.A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
            self.A = self.A.real.astype(np.float32)
            
        elif dynamics_type == "oscillatory":
            self.frequencies = np.random.uniform(0.1, 0.5, d_features)
            self.phases = np.random.uniform(0, 2 * np.pi, d_features)
            self.amplitudes = np.random.uniform(0.5, 1.5, d_features)
            
        elif dynamics_type == "switching":
            # Multiple dynamics matrices for mode switching
            self.modes = []
            for _ in range(3):
                A = np.random.randn(d_features, d_features) * 0.1
                eigenvalues, eigenvectors = np.linalg.eig(A)
                eigenvalues = eigenvalues / (np.abs(eigenvalues).max() + 0.1) * 0.9
                A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
                self.modes.append(A.real.astype(np.float32))
            self.switch_prob = 0.1
    
    def generate(self, seq_len: int, noise_std: float = 0.1) -> np.ndarray:
        """Generate a sequence following the dynamics."""
        if self.dynamics_type == "linear":
            return self._generate_linear(seq_len, noise_std)
        elif self.dynamics_type == "oscillatory":
            return self._generate_oscillatory(seq_len, noise_std)
        elif self.dynamics_type == "switching":
            return self._generate_switching(seq_len, noise_std)
        else:
            return self._generate_linear(seq_len, noise_std)
    
    def _generate_linear(self, seq_len: int, noise_std: float) -> np.ndarray:
        """Generate linear dynamics sequence."""
        x = np.zeros((seq_len, self.d_features), dtype=np.float32)
        x[0] = np.random.randn(self.d_features) * 0.5
        
        for t in range(1, seq_len):
            x[t] = self.A @ x[t-1] + np.random.randn(self.d_features) * noise_std
        
        return x
    
    def _generate_oscillatory(self, seq_len: int, noise_std: float) -> np.ndarray:
        """Generate oscillatory sequence."""
        t = np.arange(seq_len).reshape(-1, 1)
        x = self.amplitudes * np.sin(2 * np.pi * self.frequencies * t + self.phases)
        x = x + np.random.randn(seq_len, self.d_features) * noise_std
        return x.astype(np.float32)
    
    def _generate_switching(self, seq_len: int, noise_std: float) -> np.ndarray:
        """Generate mode-switching sequence."""
        x = np.zeros((seq_len, self.d_features), dtype=np.float32)
        x[0] = np.random.randn(self.d_features) * 0.5
        
        current_mode = 0
        mode_sequence = [current_mode]
        
        for t in range(1, seq_len):
            # Maybe switch mode
            if random.random() < self.switch_prob:
                current_mode = (current_mode + 1) % len(self.modes)
            mode_sequence.append(current_mode)
            
            x[t] = self.modes[current_mode] @ x[t-1] + np.random.randn(self.d_features) * noise_std
        
        return x
    
    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict future states from context (ground truth).
        
        This is used to evaluate predictions against known ground truth.
        """
        predictions = np.zeros((n_steps, self.d_features), dtype=np.float32)
        
        if self.dynamics_type == "linear":
            last_state = context[-1]
            for t in range(n_steps):
                if t == 0:
                    predictions[t] = self.A @ last_state
                else:
                    predictions[t] = self.A @ predictions[t-1]
                    
        elif self.dynamics_type == "oscillatory":
            start_t = len(context)
            t = np.arange(start_t, start_t + n_steps).reshape(-1, 1)
            predictions = self.amplitudes * np.sin(2 * np.pi * self.frequencies * t + self.phases)
            predictions = predictions.astype(np.float32)
            
        elif self.dynamics_type == "switching":
            # Use last mode for prediction (simplified)
            last_state = context[-1]
            current_mode = 0  # Default to first mode
            for t in range(n_steps):
                if t == 0:
                    predictions[t] = self.modes[current_mode] @ last_state
                else:
                    predictions[t] = self.modes[current_mode] @ predictions[t-1]
        
        return predictions


class WorldModelValidationDataset(Dataset):
    """
    Dataset for world model validation.
    
    Each sample contains:
    - context: Input sequence (context)
    - target: Target sequence (future to predict)
    - dynamics_type: Type of underlying dynamics
    - ground_truth_prediction: What perfect prediction should be
    """
    
    def __init__(
        self,
        num_samples: int = 5000,
        config: Optional[WorldModelTaskConfig] = None,
        dynamics_types: Optional[List[str]] = None,
    ):
        self.num_samples = num_samples
        self.config = config or WorldModelTaskConfig()
        self.dynamics_types = dynamics_types or ["linear", "oscillatory", "switching"]
        
        # Create generators for each dynamics type
        self.generators = {
            dtype: DynamicsGenerator(self.config.d_features, dtype)
            for dtype in self.dynamics_types
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a single world model validation sample."""
        # Select dynamics type
        dynamics_type = self.dynamics_types[idx % len(self.dynamics_types)]
        generator = self.generators[dynamics_type]
        
        # Generate full sequence
        total_len = self.config.context_len + self.config.prediction_len
        full_sequence = generator.generate(total_len, self.config.noise_std)
        
        # Split into context and target
        context = full_sequence[:self.config.context_len]
        target = full_sequence[self.config.context_len:]
        
        # Get ground truth prediction (what perfect model would predict)
        ground_truth = generator.predict(context, self.config.prediction_len)
        
        # Convert to tensors
        context_tensor = torch.tensor(context, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
        
        # Create masks
        context_mask = torch.ones(self.config.context_len, dtype=torch.bool)
        target_mask = torch.ones(self.config.prediction_len, dtype=torch.bool)
        
        return {
            "context": context_tensor,
            "target": target_tensor,
            "ground_truth_prediction": ground_truth_tensor,
            "context_mask": context_mask,
            "target_mask": target_mask,
            "dynamics_type": dynamics_type,
            "context_len": self.config.context_len,
            "prediction_len": self.config.prediction_len,
        }


class StateTransitionDataset(Dataset):
    """
    Dataset for state transition prediction.
    
    Simulates abstract state machines where we need to predict
    the next state given current state and action.
    
    This tests if the world model learns transition functions.
    """
    
    def __init__(
        self,
        num_samples: int = 5000,
        n_states: int = 10,
        n_actions: int = 4,
        d_features: int = 32,
    ):
        self.num_samples = num_samples
        self.n_states = n_states
        self.n_actions = n_actions
        self.d_features = d_features
        
        # Create deterministic transition function
        # transition[state, action] -> next_state
        self.transitions = np.random.randint(0, n_states, (n_states, n_actions))
        
        # Create state embeddings (ground truth representations)
        self.state_embeddings = np.random.randn(n_states, d_features).astype(np.float32)
        self.state_embeddings = self.state_embeddings / np.linalg.norm(
            self.state_embeddings, axis=1, keepdims=True
        )
        
        # Create action embeddings
        self.action_embeddings = np.random.randn(n_actions, d_features).astype(np.float32)
        self.action_embeddings = self.action_embeddings / np.linalg.norm(
            self.action_embeddings, axis=1, keepdims=True
        )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a single state transition sample."""
        # Random starting state
        current_state = random.randint(0, self.n_states - 1)
        
        # Generate trajectory
        trajectory_len = random.randint(5, 15)
        states = [current_state]
        actions = []
        
        for _ in range(trajectory_len):
            action = random.randint(0, self.n_actions - 1)
            next_state = self.transitions[current_state, action]
            
            actions.append(action)
            states.append(next_state)
            current_state = next_state
        
        # Convert to embeddings
        state_sequence = np.array([self.state_embeddings[s] for s in states], dtype=np.float32)
        action_sequence = np.array([self.action_embeddings[a] for a in actions], dtype=np.float32)
        
        # Create interleaved sequence: [s0, a0, s1, a1, ...]
        interleaved = []
        for i in range(len(actions)):
            interleaved.append(state_sequence[i])
            interleaved.append(action_sequence[i])
        interleaved.append(state_sequence[-1])
        
        interleaved = np.array(interleaved, dtype=np.float32)
        
        # Context: all but last state
        context = interleaved[:-1]
        
        # Target: next state embedding
        target = state_sequence[-1]
        target_state_id = states[-1]
        
        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_state_id": target_state_id,
            "states": states,
            "actions": actions,
            "trajectory_len": trajectory_len,
        }


class AbstractionDataset(Dataset):
    """
    Dataset for testing hierarchical abstraction learning.
    
    Contains sequences with multi-level structure:
    - Low-level: Raw features
    - Mid-level: Chunked patterns
    - High-level: Abstract categories
    
    Tests if the world model learns appropriate abstractions.
    """
    
    def __init__(
        self,
        num_samples: int = 2000,
        d_features: int = 32,
        n_categories: int = 5,
        n_patterns_per_category: int = 4,
        pattern_len: int = 8,
    ):
        self.num_samples = num_samples
        self.d_features = d_features
        self.n_categories = n_categories
        self.n_patterns_per_category = n_patterns_per_category
        self.pattern_len = pattern_len
        
        # Generate prototype patterns for each category
        self.prototypes = {}
        for cat in range(n_categories):
            cat_prototypes = []
            # Category center
            cat_center = np.random.randn(d_features).astype(np.float32)
            
            for _ in range(n_patterns_per_category):
                # Pattern is variation around category center
                pattern = []
                for _ in range(pattern_len):
                    variation = cat_center + np.random.randn(d_features).astype(np.float32) * 0.3
                    pattern.append(variation)
                cat_prototypes.append(np.array(pattern, dtype=np.float32))
            
            self.prototypes[cat] = cat_prototypes
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a sample with hierarchical structure."""
        # Select categories for this sequence
        n_patterns = random.randint(3, 6)
        categories = [random.randint(0, self.n_categories - 1) for _ in range(n_patterns)]
        
        # Build sequence from patterns
        sequence = []
        pattern_boundaries = [0]
        
        for cat in categories:
            pattern_idx = random.randint(0, self.n_patterns_per_category - 1)
            pattern = self.prototypes[cat][pattern_idx]
            
            # Add noise to pattern
            noisy_pattern = pattern + np.random.randn(*pattern.shape).astype(np.float32) * 0.1
            sequence.append(noisy_pattern)
            pattern_boundaries.append(pattern_boundaries[-1] + len(noisy_pattern))
        
        sequence = np.concatenate(sequence, axis=0)
        
        # Split into context and target
        split_point = pattern_boundaries[len(pattern_boundaries) // 2]
        context = sequence[:split_point]
        target = sequence[split_point:]
        
        # Ground truth: category of target portion
        target_categories = categories[len(pattern_boundaries) // 2:]
        
        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "categories": categories,
            "target_categories": target_categories,
            "pattern_boundaries": pattern_boundaries,
            "n_patterns": n_patterns,
        }


def evaluate_world_model(
    model: torch.nn.Module,
    dataset: WorldModelValidationDataset,
    device: torch.device,
    max_samples: int = 200,
) -> Dict[str, float]:
    """
    Evaluate world model predictions.
    
    Metrics:
    - MSE: Mean squared error between predictions and targets
    - Cosine similarity: Direction accuracy
    - Per-dynamics accuracy
    """
    model.eval()
    
    mse_total = 0.0
    cosine_total = 0.0
    dynamics_mse = {}
    n_samples = 0
    
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            sample = dataset[i]
            
            context = sample["context"].unsqueeze(0).to(device)
            target = sample["target"].to(device)
            ground_truth = sample["ground_truth_prediction"].to(device)
            dynamics_type = sample["dynamics_type"]
            
            # Get model predictions
            # Assuming model has an 'imagine' method for world model predictions
            if hasattr(model, 'imagine'):
                predictions = model.imagine(context, n_steps=target.shape[0])
                predictions = predictions.squeeze(0)
            else:
                # Fallback: use forward pass
                full_seq = context
                outputs = model(full_seq, modality="continuous")
                predictions = outputs.get("hidden_states", outputs.get("regression"))
                if predictions is not None:
                    predictions = predictions[0, -target.shape[0]:]
                else:
                    continue
            
            # Compute metrics
            if predictions.shape == target.shape:
                mse = torch.nn.functional.mse_loss(predictions, target).item()
                mse_total += mse
                
                # Cosine similarity
                pred_flat = predictions.view(-1)
                target_flat = target.view(-1)
                cosine = torch.nn.functional.cosine_similarity(
                    pred_flat.unsqueeze(0),
                    target_flat.unsqueeze(0)
                ).item()
                cosine_total += cosine
                
                # Per-dynamics MSE
                if dynamics_type not in dynamics_mse:
                    dynamics_mse[dynamics_type] = []
                dynamics_mse[dynamics_type].append(mse)
                
                n_samples += 1
    
    metrics = {
        "world_model_mse": mse_total / max(n_samples, 1),
        "world_model_cosine": cosine_total / max(n_samples, 1),
        "n_samples_evaluated": n_samples,
    }
    
    # Per-dynamics metrics
    for dtype, mses in dynamics_mse.items():
        metrics[f"mse_{dtype}"] = np.mean(mses)
    
    return metrics


def create_world_model_benchmark(
    num_samples: int = 2000,
    dynamics_types: Optional[List[str]] = None,
    context_len: int = 32,
    prediction_len: int = 16,
) -> WorldModelValidationDataset:
    """
    Factory function to create world model validation dataset.
    
    Args:
        num_samples: Number of samples
        dynamics_types: Which dynamics to include
        context_len: Length of context sequence
        prediction_len: Length of target sequence
    
    Returns:
        WorldModelValidationDataset
    """
    config = WorldModelTaskConfig(
        context_len=context_len,
        prediction_len=prediction_len,
    )
    
    return WorldModelValidationDataset(
        num_samples=num_samples,
        config=config,
        dynamics_types=dynamics_types,
    )
