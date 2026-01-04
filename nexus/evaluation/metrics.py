"""Evaluation metrics for NEXUS model.

This module provides comprehensive evaluation metrics for:
- Sequence modeling (perplexity, accuracy)
- Reasoning quality (proof validity, grounding)
- Causal inference accuracy
- Computational efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import time


@dataclass
class MetricConfig:
    """Configuration for metric computation."""
    compute_perplexity: bool = True
    compute_reasoning: bool = True
    compute_causal: bool = True
    compute_efficiency: bool = True
    
    # Thresholds
    reasoning_confidence_threshold: float = 0.5
    causal_edge_threshold: float = 0.5


class MetricAccumulator:
    """Accumulator for computing running metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self._values: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
    
    def update(self, name: str, value: float, count: int = 1):
        """Update metric accumulator."""
        self._values[name].append(value * count)
        self._counts[name] += count
    
    def compute(self, name: str) -> float:
        """Compute average for a metric."""
        if self._counts[name] == 0:
            return 0.0
        return sum(self._values[name]) / self._counts[name]
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics."""
        return {name: self.compute(name) for name in self._values}


def compute_perplexity(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100
) -> Tensor:
    """Compute perplexity from logits and targets.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        targets: Target token indices [batch, seq_len]
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Perplexity score
    """
    # Reshape for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Perplexity is exp of cross entropy
    perplexity = torch.exp(loss)
    
    return perplexity


def compute_accuracy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
    top_k: int = 1
) -> Tensor:
    """Compute top-k accuracy.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        targets: Target token indices [batch, seq_len]
        ignore_index: Index to ignore in computation
        top_k: Compute top-k accuracy
        
    Returns:
        Accuracy score
    """
    # Get top-k predictions
    _, top_k_preds = logits.topk(top_k, dim=-1)
    
    # Expand targets for comparison
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    
    # Check if target is in top-k
    correct = (top_k_preds == targets_expanded).any(dim=-1)
    
    # Mask ignored positions
    mask = targets != ignore_index
    
    # Compute accuracy
    accuracy = correct[mask].float().mean()
    
    return accuracy


def compute_reasoning_accuracy(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    config: Optional[MetricConfig] = None
) -> Dict[str, float]:
    """Compute reasoning evaluation metrics.
    
    Args:
        predictions: List of predicted reasoning outputs
        ground_truth: List of ground truth reasoning outputs
        config: Metric configuration
        
    Returns:
        Dictionary of reasoning metrics
    """
    config = config or MetricConfig()
    
    metrics = {
        'proof_validity': 0.0,
        'conclusion_accuracy': 0.0,
        'step_accuracy': 0.0,
        'grounding_score': 0.0,
    }
    
    if not predictions:
        return metrics
    
    valid_proofs = 0
    correct_conclusions = 0
    total_steps = 0
    correct_steps = 0
    grounding_scores = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Check proof validity (all steps follow from premises)
        if 'proof_trace' in pred:
            is_valid = _validate_proof_trace(pred['proof_trace'])
            valid_proofs += int(is_valid)
        
        # Check conclusion accuracy
        if 'conclusion' in pred and 'conclusion' in gt:
            if pred['conclusion'] == gt['conclusion']:
                correct_conclusions += 1
        
        # Check intermediate step accuracy
        if 'steps' in pred and 'steps' in gt:
            for p_step, g_step in zip(pred['steps'], gt['steps']):
                total_steps += 1
                if p_step == g_step:
                    correct_steps += 1
        
        # Compute grounding score (how well connected to knowledge)
        if 'grounding' in pred:
            grounding_scores.append(pred['grounding'].get('score', 0.0))
    
    n = len(predictions)
    metrics['proof_validity'] = valid_proofs / n if n > 0 else 0.0
    metrics['conclusion_accuracy'] = correct_conclusions / n if n > 0 else 0.0
    metrics['step_accuracy'] = correct_steps / total_steps if total_steps > 0 else 0.0
    metrics['grounding_score'] = np.mean(grounding_scores) if grounding_scores else 0.0
    
    return metrics


def _validate_proof_trace(proof_trace: Dict) -> bool:
    """Validate that a proof trace is logically valid.
    
    This performs basic consistency checks on the proof structure.
    """
    if not proof_trace:
        return False
    
    # Check required fields
    required_fields = ['premises', 'steps', 'conclusion']
    if not all(field in proof_trace for field in required_fields):
        return False
    
    # Check that each step references valid previous steps/premises
    available = set(range(len(proof_trace.get('premises', []))))
    
    for i, step in enumerate(proof_trace.get('steps', [])):
        refs = step.get('references', [])
        if not all(ref in available for ref in refs):
            return False
        # Add this step's index as available for future steps
        available.add(len(proof_trace.get('premises', [])) + i)
    
    return True


def compute_causal_accuracy(
    predicted_graph: Dict[str, List[str]],
    true_graph: Dict[str, List[str]],
    interventions: Optional[List[Dict]] = None,
    counterfactuals: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """Compute causal inference evaluation metrics.
    
    Args:
        predicted_graph: Predicted causal graph (adjacency dict)
        true_graph: Ground truth causal graph
        interventions: List of intervention test cases
        counterfactuals: List of counterfactual test cases
        
    Returns:
        Dictionary of causal metrics
    """
    metrics = {}
    
    # Graph structure metrics
    structure_metrics = _compute_graph_metrics(predicted_graph, true_graph)
    metrics.update(structure_metrics)
    
    # Intervention accuracy
    if interventions:
        intervention_acc = _compute_intervention_accuracy(interventions)
        metrics['intervention_accuracy'] = intervention_acc
    
    # Counterfactual accuracy
    if counterfactuals:
        cf_acc = _compute_counterfactual_accuracy(counterfactuals)
        metrics['counterfactual_accuracy'] = cf_acc
    
    return metrics


def _compute_graph_metrics(
    predicted: Dict[str, List[str]],
    true: Dict[str, List[str]]
) -> Dict[str, float]:
    """Compute causal graph structure metrics."""
    # Extract edges
    pred_edges = set()
    for node, children in predicted.items():
        for child in children:
            pred_edges.add((node, child))
    
    true_edges = set()
    for node, children in true.items():
        for child in children:
            true_edges.add((node, child))
    
    # Compute precision, recall, F1
    true_positives = len(pred_edges & true_edges)
    false_positives = len(pred_edges - true_edges)
    false_negatives = len(true_edges - pred_edges)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance (SHD)
    shd = false_positives + false_negatives
    
    return {
        'edge_precision': precision,
        'edge_recall': recall,
        'edge_f1': f1,
        'structural_hamming_distance': shd,
    }


def _compute_intervention_accuracy(interventions: List[Dict]) -> float:
    """Compute accuracy of intervention effect predictions."""
    if not interventions:
        return 0.0
    
    correct = 0
    for intervention in interventions:
        pred_effect = intervention.get('predicted_effect')
        true_effect = intervention.get('true_effect')
        
        if pred_effect is not None and true_effect is not None:
            # Check if prediction matches within tolerance
            if isinstance(pred_effect, (int, float)) and isinstance(true_effect, (int, float)):
                if abs(pred_effect - true_effect) < 0.1:
                    correct += 1
            elif pred_effect == true_effect:
                correct += 1
    
    return correct / len(interventions)


def _compute_counterfactual_accuracy(counterfactuals: List[Dict]) -> float:
    """Compute accuracy of counterfactual reasoning."""
    if not counterfactuals:
        return 0.0
    
    correct = 0
    for cf in counterfactuals:
        pred_outcome = cf.get('predicted_outcome')
        true_outcome = cf.get('true_outcome')
        
        if pred_outcome == true_outcome:
            correct += 1
    
    return correct / len(counterfactuals)


class EfficiencyMetrics:
    """Compute computational efficiency metrics."""
    
    def __init__(self):
        self.inference_times: List[float] = []
        self.memory_usage: List[int] = []
        self.flops: List[int] = []
    
    def reset(self):
        """Reset accumulated metrics."""
        self.inference_times = []
        self.memory_usage = []
        self.flops = []
    
    def start_timer(self) -> float:
        """Start timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()
    
    def end_timer(self, start_time: float) -> float:
        """End timing and record."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        return elapsed
    
    def record_memory(self):
        """Record current GPU memory usage."""
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated()
            self.memory_usage.append(memory)
            return memory
        return 0
    
    def compute_metrics(self, sequence_lengths: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute efficiency metrics."""
        metrics = {}
        
        if self.inference_times:
            metrics['avg_inference_time_ms'] = np.mean(self.inference_times) * 1000
            metrics['std_inference_time_ms'] = np.std(self.inference_times) * 1000
            metrics['throughput_samples_per_sec'] = len(self.inference_times) / sum(self.inference_times)
        
        if self.memory_usage:
            metrics['avg_memory_mb'] = np.mean(self.memory_usage) / (1024 ** 2)
            metrics['peak_memory_mb'] = max(self.memory_usage) / (1024 ** 2)
        
        # Scaling analysis if sequence lengths provided
        if sequence_lengths and len(sequence_lengths) == len(self.inference_times):
            # Fit linear scaling (expected for NEXUS)
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(
                sequence_lengths, self.inference_times
            )
            metrics['time_complexity_slope'] = slope
            metrics['linear_scaling_r2'] = r_value ** 2
        
        return metrics


def compare_models(
    nexus_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    metric_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Compare NEXUS metrics against baseline (e.g., Transformer).
    
    Args:
        nexus_metrics: NEXUS model metrics
        baseline_metrics: Baseline model metrics
        metric_weights: Optional weights for each metric
        
    Returns:
        Comparison results including improvements and summary
    """
    default_weights = {
        'perplexity': -1.0,  # Lower is better
        'accuracy': 1.0,     # Higher is better
        'inference_time_ms': -1.0,  # Lower is better
        'memory_mb': -1.0,   # Lower is better
        'reasoning_accuracy': 1.0,
        'causal_f1': 1.0,
    }
    
    weights = {**default_weights, **(metric_weights or {})}
    
    comparison = {
        'individual_improvements': {},
        'nexus_better_metrics': [],
        'baseline_better_metrics': [],
        'weighted_score_nexus': 0.0,
        'weighted_score_baseline': 0.0,
    }
    
    for metric in set(nexus_metrics.keys()) & set(baseline_metrics.keys()):
        nexus_val = nexus_metrics[metric]
        baseline_val = baseline_metrics[metric]
        
        # Compute relative improvement
        if baseline_val != 0:
            improvement = (nexus_val - baseline_val) / abs(baseline_val) * 100
        else:
            improvement = 0.0 if nexus_val == 0 else float('inf')
        
        comparison['individual_improvements'][metric] = improvement
        
        # Determine which is better based on metric direction
        weight = weights.get(metric, 1.0)
        if weight > 0:  # Higher is better
            if nexus_val > baseline_val:
                comparison['nexus_better_metrics'].append(metric)
            else:
                comparison['baseline_better_metrics'].append(metric)
        else:  # Lower is better
            if nexus_val < baseline_val:
                comparison['nexus_better_metrics'].append(metric)
            else:
                comparison['baseline_better_metrics'].append(metric)
        
        # Accumulate weighted scores
        comparison['weighted_score_nexus'] += abs(weight) * nexus_val * np.sign(weight)
        comparison['weighted_score_baseline'] += abs(weight) * baseline_val * np.sign(weight)
    
    # Summary
    comparison['summary'] = {
        'nexus_wins': len(comparison['nexus_better_metrics']),
        'baseline_wins': len(comparison['baseline_better_metrics']),
        'overall_winner': 'NEXUS' if comparison['weighted_score_nexus'] > comparison['weighted_score_baseline'] else 'Baseline'
    }
    
    return comparison


def generate_report(
    metrics: Dict[str, float],
    comparison: Optional[Dict[str, Any]] = None,
    config: Optional[MetricConfig] = None
) -> str:
    """Generate a human-readable evaluation report.
    
    Args:
        metrics: Computed metrics
        comparison: Optional model comparison results
        config: Metric configuration
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("NEXUS Model Evaluation Report")
    report.append("=" * 60)
    report.append("")
    
    # Group metrics by category
    categories = {
        'Language Modeling': ['perplexity', 'accuracy', 'top_5_accuracy'],
        'Reasoning': ['proof_validity', 'conclusion_accuracy', 'step_accuracy', 'grounding_score'],
        'Causal Inference': ['edge_precision', 'edge_recall', 'edge_f1', 'intervention_accuracy', 'counterfactual_accuracy'],
        'Efficiency': ['avg_inference_time_ms', 'throughput_samples_per_sec', 'avg_memory_mb', 'linear_scaling_r2'],
    }
    
    for category, metric_names in categories.items():
        category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
        if category_metrics:
            report.append(f"\n{category}:")
            report.append("-" * 40)
            for name, value in category_metrics.items():
                report.append(f"  {name}: {value:.4f}")
    
    # Add comparison if provided
    if comparison:
        report.append("\n" + "=" * 60)
        report.append("Model Comparison: NEXUS vs Baseline")
        report.append("=" * 60)
        report.append("")
        
        for metric, improvement in comparison.get('individual_improvements', {}).items():
            direction = "↑" if improvement > 0 else "↓"
            report.append(f"  {metric}: {improvement:+.2f}% {direction}")
        
        report.append("")
        report.append(f"Overall Winner: {comparison.get('summary', {}).get('overall_winner', 'N/A')}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)
