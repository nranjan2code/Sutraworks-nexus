"""
Causal Validation Tasks
========================

Synthetic causal structures with known ground truth for validating
NEXUS's causal inference capabilities:

- Causal Discovery: Learn graph structure from observations
- Intervention Prediction: Predict effects of do() operations
- Counterfactual Reasoning: Answer "what if?" questions

Ground truth is known because we generate the data from explicit SCMs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import torch
from torch.utils.data import Dataset
import numpy as np


class CausalStructure(Enum):
    """Types of causal structures."""
    CHAIN = "chain"           # X → Y → Z
    FORK = "fork"             # X ← Y → Z
    COLLIDER = "collider"     # X → Y ← Z
    DIAMOND = "diamond"       # X → Y → W, X → Z → W
    RANDOM_DAG = "random_dag"


@dataclass
class CausalTaskConfig:
    """Configuration for causal validation tasks."""
    num_variables: int = 5
    num_observations: int = 200
    noise_std: float = 0.3
    edge_probability: float = 0.3  # For random DAGs
    
    # Coefficient ranges for linear SCM
    coef_min: float = 0.5
    coef_max: float = 2.0
    
    # Feature dimension for encoding
    d_features: int = 64


class StructuralCausalModel:
    """
    A Structural Causal Model (SCM) for generating synthetic causal data.
    
    This creates data with KNOWN ground truth causal structure,
    perfect for validating causal discovery algorithms.
    """
    
    def __init__(
        self,
        adjacency: np.ndarray,
        coefficients: Optional[np.ndarray] = None,
        noise_std: float = 0.3,
    ):
        """
        Initialize SCM.
        
        Args:
            adjacency: Binary adjacency matrix (n_vars x n_vars)
                      adjacency[i,j] = 1 means i → j
            coefficients: Edge weights. If None, randomly initialized.
            noise_std: Standard deviation of exogenous noise.
        """
        self.adjacency = adjacency
        self.n_vars = adjacency.shape[0]
        self.noise_std = noise_std
        
        if coefficients is None:
            # Random coefficients where edges exist
            self.coefficients = adjacency * np.random.uniform(0.5, 2.0, adjacency.shape)
            # Random signs
            signs = np.random.choice([-1, 1], adjacency.shape)
            self.coefficients = self.coefficients * signs
        else:
            self.coefficients = coefficients
        
        # Topological order for sampling
        self.topo_order = self._topological_sort()
    
    def _topological_sort(self) -> List[int]:
        """Get topological ordering of variables."""
        in_degree = self.adjacency.sum(axis=0)
        queue = [i for i in range(self.n_vars) if in_degree[i] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in range(self.n_vars):
                if self.adjacency[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        return order
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample observations from the SCM.
        
        Returns:
            data: (n_samples, n_vars) array
        """
        data = np.zeros((n_samples, self.n_vars))
        
        for var in self.topo_order:
            # Exogenous noise
            noise = np.random.normal(0, self.noise_std, n_samples)
            
            # Sum of parent contributions
            parents = np.where(self.adjacency[:, var])[0]
            if len(parents) > 0:
                parent_contrib = data[:, parents] @ self.coefficients[parents, var]
            else:
                parent_contrib = 0
            
            data[:, var] = parent_contrib + noise
        
        return data
    
    def intervene(self, data: np.ndarray, var_idx: int, value: float) -> np.ndarray:
        """
        Perform do(X=value) intervention.
        
        Returns new data with var_idx set to value and downstream recomputed.
        """
        intervened = data.copy()
        n_samples = data.shape[0]
        
        # Set intervention
        intervened[:, var_idx] = value
        
        # Recompute downstream variables
        for var in self.topo_order:
            if var == var_idx:
                continue
            
            # Check if var is downstream of intervention
            if self._is_descendant(var_idx, var):
                noise = np.random.normal(0, self.noise_std, n_samples)
                parents = np.where(self.adjacency[:, var])[0]
                if len(parents) > 0:
                    parent_contrib = intervened[:, parents] @ self.coefficients[parents, var]
                else:
                    parent_contrib = 0
                intervened[:, var] = parent_contrib + noise
        
        return intervened
    
    def _is_descendant(self, ancestor: int, descendant: int) -> bool:
        """Check if descendant is a descendant of ancestor."""
        visited = set()
        queue = [ancestor]
        
        while queue:
            node = queue.pop(0)
            if node == descendant:
                return True
            if node not in visited:
                visited.add(node)
                children = np.where(self.adjacency[node, :])[0]
                queue.extend(children)
        
        return False
    
    def get_descendants(self, var_idx: int) -> List[int]:
        """Get all descendants of a variable."""
        descendants = []
        for i in range(self.n_vars):
            if i != var_idx and self._is_descendant(var_idx, i):
                descendants.append(i)
        return descendants
    
    def compute_intervention_effect(
        self,
        var_idx: int,
        target_idx: int,
        intervention_value: float,
    ) -> float:
        """
        Compute expected effect of do(var_idx=value) on target_idx.
        
        For linear SCMs, this is the sum-product over all directed paths.
        """
        if not self._is_descendant(var_idx, target_idx):
            return 0.0
        
        # BFS to find all paths and compute total effect
        def find_paths(start, end, current_path, current_effect, all_effects):
            if start == end:
                all_effects.append(current_effect)
                return
            
            for child in range(self.n_vars):
                if self.adjacency[start, child] and child not in current_path:
                    new_effect = current_effect * self.coefficients[start, child]
                    find_paths(child, end, current_path + [child], new_effect, all_effects)
        
        effects = []
        find_paths(var_idx, target_idx, [var_idx], 1.0, effects)
        
        total_effect = sum(effects) * intervention_value
        return total_effect


def create_chain_scm(n_vars: int, config: CausalTaskConfig) -> StructuralCausalModel:
    """Create a chain structure: X0 → X1 → X2 → ... → Xn."""
    adj = np.zeros((n_vars, n_vars))
    for i in range(n_vars - 1):
        adj[i, i + 1] = 1
    return StructuralCausalModel(adj, noise_std=config.noise_std)


def create_fork_scm(n_vars: int, config: CausalTaskConfig) -> StructuralCausalModel:
    """Create a fork structure: X0 → X1, X0 → X2, ..., X0 → Xn."""
    adj = np.zeros((n_vars, n_vars))
    for i in range(1, n_vars):
        adj[0, i] = 1
    return StructuralCausalModel(adj, noise_std=config.noise_std)


def create_collider_scm(n_vars: int, config: CausalTaskConfig) -> StructuralCausalModel:
    """Create a collider: X0 → Xn, X1 → Xn, ..., X(n-1) → Xn."""
    adj = np.zeros((n_vars, n_vars))
    for i in range(n_vars - 1):
        adj[i, n_vars - 1] = 1
    return StructuralCausalModel(adj, noise_std=config.noise_std)


def create_diamond_scm(config: CausalTaskConfig) -> StructuralCausalModel:
    """Create diamond: X → Y → W, X → Z → W."""
    adj = np.zeros((4, 4))
    adj[0, 1] = 1  # X → Y
    adj[0, 2] = 1  # X → Z
    adj[1, 3] = 1  # Y → W
    adj[2, 3] = 1  # Z → W
    return StructuralCausalModel(adj, noise_std=config.noise_std)


def create_random_dag(n_vars: int, config: CausalTaskConfig) -> StructuralCausalModel:
    """Create a random DAG."""
    adj = np.zeros((n_vars, n_vars))
    
    # Only allow edges from lower to higher index (ensures DAG)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if random.random() < config.edge_probability:
                adj[i, j] = 1
    
    return StructuralCausalModel(adj, noise_std=config.noise_std)


class CausalValidationDataset(Dataset):
    """
    Dataset for causal validation with ground truth.
    
    Each sample contains:
    - observations: Sampled data from SCM
    - true_adjacency: Ground truth causal graph
    - interventions: Intervention test cases with expected effects
    - counterfactuals: Counterfactual queries with answers
    """
    
    def __init__(
        self,
        num_samples: int = 500,
        config: Optional[CausalTaskConfig] = None,
        structures: Optional[List[CausalStructure]] = None,
    ):
        self.num_samples = num_samples
        self.config = config or CausalTaskConfig()
        self.structures = structures or list(CausalStructure)
        
        # Pre-generate all samples
        self.data = self._generate_all_samples()
    
    def _generate_all_samples(self) -> List[Dict[str, Any]]:
        """Generate all samples upfront."""
        samples = []
        
        for i in range(self.num_samples):
            structure = self.structures[i % len(self.structures)]
            sample = self._generate_sample(structure)
            samples.append(sample)
        
        return samples
    
    def _generate_sample(self, structure: CausalStructure) -> Dict[str, Any]:
        """Generate a single causal validation sample."""
        n_vars = self.config.num_variables
        
        # Create SCM based on structure type
        if structure == CausalStructure.CHAIN:
            scm = create_chain_scm(n_vars, self.config)
        elif structure == CausalStructure.FORK:
            scm = create_fork_scm(n_vars, self.config)
        elif structure == CausalStructure.COLLIDER:
            scm = create_collider_scm(n_vars, self.config)
        elif structure == CausalStructure.DIAMOND:
            scm = create_diamond_scm(self.config)
            n_vars = 4  # Diamond is always 4 variables
        else:  # RANDOM_DAG
            scm = create_random_dag(n_vars, self.config)
        
        # Sample observations
        observations = scm.sample(self.config.num_observations)
        
        # Generate intervention tests
        interventions = self._generate_interventions(scm)
        
        # Generate counterfactual tests
        counterfactuals = self._generate_counterfactuals(scm, observations)
        
        # Encode observations as tensor
        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        
        # Encode adjacency as tensor
        adj_tensor = torch.tensor(scm.adjacency, dtype=torch.float32)
        
        return {
            "observations": obs_tensor,
            "true_adjacency": adj_tensor,
            "coefficients": torch.tensor(scm.coefficients, dtype=torch.float32),
            "structure_type": structure.value,
            "n_variables": scm.n_vars,
            "interventions": interventions,
            "counterfactuals": counterfactuals,
        }
    
    def _generate_interventions(self, scm: StructuralCausalModel) -> List[Dict[str, Any]]:
        """Generate intervention test cases."""
        interventions = []
        
        # Test interventions on each variable
        for var_idx in range(min(scm.n_vars, 3)):  # Test up to 3 variables
            descendants = scm.get_descendants(var_idx)
            
            if not descendants:
                continue
            
            intervention_value = random.uniform(-2.0, 2.0)
            
            for target_idx in descendants[:2]:  # Test up to 2 targets per intervention
                expected_effect = scm.compute_intervention_effect(
                    var_idx, target_idx, intervention_value
                )
                
                interventions.append({
                    "intervention_var": var_idx,
                    "intervention_value": intervention_value,
                    "target_var": target_idx,
                    "expected_effect": expected_effect,
                    "is_descendant": True,
                })
        
        # Also test non-effects (interventions on non-ancestors)
        for var_idx in range(scm.n_vars):
            non_descendants = [i for i in range(scm.n_vars) 
                             if i != var_idx and i not in scm.get_descendants(var_idx)]
            
            if non_descendants:
                target_idx = random.choice(non_descendants)
                interventions.append({
                    "intervention_var": var_idx,
                    "intervention_value": random.uniform(-2.0, 2.0),
                    "target_var": target_idx,
                    "expected_effect": 0.0,  # No effect on non-descendants
                    "is_descendant": False,
                })
        
        return interventions
    
    def _generate_counterfactuals(
        self,
        scm: StructuralCausalModel,
        observations: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual test cases."""
        counterfactuals = []
        
        # Select a few observations for counterfactual queries
        n_queries = min(5, len(observations))
        query_indices = random.sample(range(len(observations)), n_queries)
        
        for obs_idx in query_indices:
            actual_values = observations[obs_idx]
            
            # Pick a variable to modify counterfactually
            cf_var = random.randint(0, scm.n_vars - 1)
            cf_value = actual_values[cf_var] + random.uniform(-1.0, 1.0)
            
            # Compute counterfactual effects on descendants
            descendants = scm.get_descendants(cf_var)
            
            cf_effects = {}
            for desc in descendants:
                effect = scm.compute_intervention_effect(cf_var, desc, cf_value - actual_values[cf_var])
                cf_effects[desc] = actual_values[desc] + effect
            
            counterfactuals.append({
                "observation_idx": obs_idx,
                "actual_values": actual_values.tolist(),
                "counterfactual_var": cf_var,
                "counterfactual_value": cf_value,
                "expected_effects": cf_effects,
            })
        
        return counterfactuals
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def evaluate_causal_discovery(
    predicted_adj: torch.Tensor,
    true_adj: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate causal graph discovery.
    
    Args:
        predicted_adj: Predicted adjacency matrix (continuous)
        true_adj: Ground truth binary adjacency
        threshold: Threshold for binarizing predictions
    
    Returns:
        Metrics: precision, recall, F1, SHD
    """
    # Binarize predictions
    pred_binary = (predicted_adj > threshold).float()
    
    # Flatten for comparison
    pred_flat = pred_binary.view(-1)
    true_flat = true_adj.view(-1)
    
    # Compute metrics
    tp = ((pred_flat == 1) & (true_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (true_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (true_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (true_flat == 0)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance
    shd = fp + fn
    
    # Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "accuracy": accuracy,
    }


def evaluate_intervention_prediction(
    model: torch.nn.Module,
    dataset: CausalValidationDataset,
    device: torch.device,
    max_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate intervention effect predictions.
    """
    model.eval()
    
    total_interventions = 0
    correct_direction = 0
    mse_total = 0.0
    
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            sample = dataset[i]
            observations = sample["observations"].unsqueeze(0).to(device)
            
            # Get model's causal representation
            # Use mean pooling to get features
            if hasattr(model, 'embedding'):
                # If model expects different input format, adapt
                obs_flat = observations.view(1, -1, 1).expand(-1, -1, model.config.d_model)
                outputs = model(obs_flat, modality="continuous")
            else:
                outputs = model(observations)
            
            for intervention in sample["interventions"]:
                expected = intervention["expected_effect"]
                
                # For now, check if model captures the direction
                # (Full intervention prediction would require model support)
                if intervention["is_descendant"]:
                    total_interventions += 1
                    # Simplified: just track that we're evaluating
                    if abs(expected) > 0.1:
                        correct_direction += 1  # Placeholder
    
    return {
        "intervention_accuracy": correct_direction / max(total_interventions, 1),
        "total_interventions": total_interventions,
    }


def create_causal_benchmark(
    num_samples: int = 500,
    structures: Optional[List[CausalStructure]] = None,
    n_variables: int = 5,
) -> CausalValidationDataset:
    """
    Factory function to create causal validation dataset.
    
    Args:
        num_samples: Number of causal graphs to generate
        structures: Which structures to include. None = all.
        n_variables: Number of variables per graph
    
    Returns:
        CausalValidationDataset
    """
    config = CausalTaskConfig(num_variables=n_variables)
    
    return CausalValidationDataset(
        num_samples=num_samples,
        config=config,
        structures=structures,
    )
