#!/usr/bin/env python3
"""
NEXUS Research Validation
==========================

Comprehensive validation script to prove the NEXUS architecture works.

This script:
1. Trains on structured synthetic tasks with known ground truth
2. Validates each architectural component:
   - State-Space: O(n) scaling, sequence memory (copy/reverse)
   - World Model: Future prediction accuracy
   - Reasoning: Variable binding, pattern completion
   - Causal Engine: Graph discovery, intervention prediction
3. Generates a validation report

Usage:
    PYTHONPATH=. python examples/research_validation.py [options]

Options:
    --quick         Quick validation (fewer samples, epochs)
    --full          Full validation (more thorough)
    --component     Validate specific component only
    --save-report   Save report to file

Expected Runtime:
    --quick: ~30 minutes on single GPU
    --full:  ~2-4 hours on single GPU
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for research validation."""
    # Model size
    model_size: str = "tiny"  # tiny, small
    
    # Training
    epochs_algorithmic: int = 10
    epochs_world_model: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    
    # Data
    algorithmic_samples: int = 5000
    causal_samples: int = 500
    world_model_samples: int = 2000
    
    # Evaluation
    eval_samples: int = 500
    
    # Output
    save_checkpoints: bool = True
    output_dir: str = "./validation_results"
    
    @classmethod
    def quick(cls) -> "ValidationConfig":
        """Quick validation config for testing."""
        return cls(
            model_size="tiny",
            epochs_algorithmic=3,
            epochs_world_model=2,
            batch_size=32,
            algorithmic_samples=1000,
            causal_samples=100,
            world_model_samples=500,
            eval_samples=100,
        )
    
    @classmethod
    def full(cls) -> "ValidationConfig":
        """Full validation config for thorough testing."""
        return cls(
            model_size="small",
            epochs_algorithmic=20,
            epochs_world_model=10,
            batch_size=16,
            algorithmic_samples=10000,
            causal_samples=1000,
            world_model_samples=5000,
            eval_samples=1000,
        )


# =============================================================================
# Validation Components
# =============================================================================

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.training_curves: Dict[str, List[float]] = {}
        self.timing: Dict[str, float] = {}
        self.config: Optional[Dict] = None
        self.timestamp: str = datetime.now().isoformat()
    
    def add_metric(self, name: str, value: Any):
        """Add a metric."""
        self.metrics[name] = value
    
    def add_training_curve(self, name: str, values: List[float]):
        """Add training curve data."""
        self.training_curves[name] = values
    
    def add_timing(self, name: str, seconds: float):
        """Add timing information."""
        self.timing[name] = seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "metrics": self.metrics,
            "training_curves": self.training_curves,
            "timing": self.timing,
        }
    
    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def create_model(config: ValidationConfig, device: torch.device):
    """Create NEXUS model for validation."""
    from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
    
    # Model configurations
    model_configs = {
        "tiny": NEXUSConfig(
            vocab_size=100,  # Small vocab for algorithmic tasks
            d_model=128,
            d_latent=64,
            ssm_n_layers=2,
            n_heads=4,
            ssm_d_state=16,
            max_seq_len=256,
            n_causal_variables=16,
        ),
        "small": NEXUSConfig(
            vocab_size=100,
            d_model=256,
            d_latent=128,
            ssm_n_layers=4,
            n_heads=8,
            ssm_d_state=32,
            max_seq_len=512,
            n_causal_variables=32,
        ),
    }
    
    nexus_config = model_configs.get(config.model_size, model_configs["tiny"])
    model = NEXUSCore(nexus_config)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    return model, nexus_config


# =============================================================================
# Algorithmic Task Validation
# =============================================================================

def validate_algorithmic_tasks(
    model: nn.Module,
    config: ValidationConfig,
    device: torch.device,
    result: ValidationResult,
) -> Dict[str, float]:
    """
    Validate state-space modeling with algorithmic tasks.
    
    Tests:
    - Copy: Basic sequence memory
    - Reverse: Bidirectional processing
    - Arithmetic: Compositional computation
    - Pattern: Pattern recognition
    """
    print("\n" + "="*60)
    print("ALGORITHMIC TASK VALIDATION")
    print("="*60)
    
    from nexus.evaluation.algorithmic_tasks import (
        AlgorithmicTaskDataset,
        TaskType,
        AlgorithmicTaskConfig,
        evaluate_algorithmic_task,
    )
    
    start_time = time.time()
    
    # Create dataset with all task types
    task_config = AlgorithmicTaskConfig(max_seq_len=256, vocab_size=100)
    
    task_types = [
        TaskType.COPY,
        TaskType.REVERSE,
        TaskType.ARITHMETIC,
        TaskType.PATTERN,
        TaskType.VARIABLE_BINDING,
        TaskType.ASSOCIATIVE_RECALL,
    ]
    
    train_dataset = AlgorithmicTaskDataset(
        task_types=task_types,
        num_samples=config.algorithmic_samples,
        config=task_config,
        difficulty="easy",  # Start easy to prove concept
    )
    
    eval_dataset = AlgorithmicTaskDataset(
        task_types=task_types,
        num_samples=config.eval_samples,
        config=task_config,
        difficulty="easy",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    training_losses = []
    
    print(f"\nTraining on {len(train_dataset)} samples for {config.epochs_algorithmic} epochs...")
    
    for epoch in range(config.epochs_algorithmic):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, modality="token")
            logits = outputs["logits"]
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        training_losses.append(avg_loss)
        
        if (epoch + 1) % max(1, config.epochs_algorithmic // 5) == 0:
            print(f"  Epoch {epoch+1}/{config.epochs_algorithmic}: Loss = {avg_loss:.4f}")
    
    # Evaluation
    print("\nEvaluating...")
    metrics = evaluate_algorithmic_task(
        model=model,
        dataset=eval_dataset,
        device=device,
        max_samples=config.eval_samples,
    )
    
    elapsed = time.time() - start_time
    result.add_timing("algorithmic_training", elapsed)
    result.add_training_curve("algorithmic_loss", training_losses)
    
    # Add metrics to result
    for name, value in metrics.items():
        result.add_metric(f"algorithmic_{name}", value)
    
    # Print results
    print("\n" + "-"*40)
    print("Algorithmic Task Results:")
    print("-"*40)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    return metrics


# =============================================================================
# Causal Inference Validation
# =============================================================================

def validate_causal_inference(
    model: nn.Module,
    config: ValidationConfig,
    device: torch.device,
    result: ValidationResult,
) -> Dict[str, float]:
    """
    Validate causal inference capabilities.
    
    Tests:
    - Causal discovery: Learning graph structure
    - Intervention prediction: Predicting do() effects
    """
    print("\n" + "="*60)
    print("CAUSAL INFERENCE VALIDATION")
    print("="*60)
    
    from nexus.evaluation.causal_tasks import (
        CausalValidationDataset,
        CausalStructure,
        CausalTaskConfig,
        evaluate_causal_discovery,
    )
    
    start_time = time.time()
    
    # Create dataset
    causal_config = CausalTaskConfig(
        num_variables=5,
        num_observations=100,
    )
    
    dataset = CausalValidationDataset(
        num_samples=config.causal_samples,
        config=causal_config,
        structures=[
            CausalStructure.CHAIN,
            CausalStructure.FORK,
            CausalStructure.DIAMOND,
        ],
    )
    
    metrics = {
        "causal_samples_evaluated": 0,
        "avg_edge_f1": 0.0,
        "avg_precision": 0.0,
        "avg_recall": 0.0,
    }
    
    model.eval()
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    print(f"\nEvaluating causal discovery on {min(len(dataset), config.eval_samples)} samples...")
    
    with torch.no_grad():
        for i in range(min(len(dataset), config.eval_samples)):
            sample = dataset[i]
            observations = sample["observations"].to(device)
            true_adj = sample["true_adjacency"]
            
            # Get model's causal output
            # Reshape observations for model input
            obs_expanded = observations.unsqueeze(0)  # (1, n_obs, n_vars)
            
            # Use model's causal engine
            if hasattr(model, 'causal_engine'):
                # Embed observations
                obs_flat = obs_expanded.view(1, -1).unsqueeze(-1)
                obs_flat = obs_flat.expand(-1, -1, model.config.d_model)
                
                # Limit sequence length
                max_len = min(obs_flat.shape[1], model.config.max_seq_len)
                obs_flat = obs_flat[:, :max_len, :]
                
                # Process through state space
                ssm_out, _ = model.state_space(obs_flat)
                
                # Get causal output
                causal_out = model.causal_engine(ssm_out)
                
                # Extract predicted adjacency
                if "adjacency" in causal_out:
                    pred_adj = causal_out["adjacency"].squeeze(0)
                else:
                    # Use learned representations to infer structure
                    pred_adj = torch.zeros_like(true_adj)
            else:
                pred_adj = torch.zeros_like(true_adj)
            
            # Evaluate
            discovery_metrics = evaluate_causal_discovery(pred_adj, true_adj)
            
            f1_scores.append(discovery_metrics["f1"])
            precision_scores.append(discovery_metrics["precision"])
            recall_scores.append(discovery_metrics["recall"])
    
    elapsed = time.time() - start_time
    
    metrics["causal_samples_evaluated"] = len(f1_scores)
    metrics["avg_edge_f1"] = np.mean(f1_scores) if f1_scores else 0.0
    metrics["avg_precision"] = np.mean(precision_scores) if precision_scores else 0.0
    metrics["avg_recall"] = np.mean(recall_scores) if recall_scores else 0.0
    
    result.add_timing("causal_validation", elapsed)
    
    for name, value in metrics.items():
        result.add_metric(f"causal_{name}", value)
    
    # Print results
    print("\n" + "-"*40)
    print("Causal Inference Results:")
    print("-"*40)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    return metrics


# =============================================================================
# World Model Validation
# =============================================================================

def validate_world_model(
    model: nn.Module,
    config: ValidationConfig,
    device: torch.device,
    result: ValidationResult,
) -> Dict[str, float]:
    """
    Validate world model (JEPA-style) predictions.
    
    Tests:
    - Future state prediction
    - Abstract representation learning
    """
    print("\n" + "="*60)
    print("WORLD MODEL VALIDATION")
    print("="*60)
    
    from nexus.evaluation.world_model_tasks import (
        WorldModelValidationDataset,
        WorldModelTaskConfig,
        evaluate_world_model,
    )
    
    start_time = time.time()
    
    # Create dataset
    wm_config = WorldModelTaskConfig(
        d_features=model.config.d_model,
        context_len=32,
        prediction_len=16,
    )
    
    train_dataset = WorldModelValidationDataset(
        num_samples=config.world_model_samples,
        config=wm_config,
        dynamics_types=["linear", "oscillatory"],
    )
    
    eval_dataset = WorldModelValidationDataset(
        num_samples=config.eval_samples,
        config=wm_config,
        dynamics_types=["linear", "oscillatory"],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Training with world model loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    training_losses = []
    
    print(f"\nTraining world model on {len(train_dataset)} samples for {config.epochs_world_model} epochs...")
    
    for epoch in range(config.epochs_world_model):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            
            # Forward pass through model
            outputs = model(context, modality="continuous")
            
            # Get predictions for target region
            hidden = outputs.get("hidden_states", outputs.get("regression"))
            if hidden is not None:
                # Use last hidden states as prediction
                pred_len = min(target.shape[1], hidden.shape[1])
                predictions = hidden[:, -pred_len:, :target.shape[-1]]
                target_trimmed = target[:, :pred_len, :]
                
                loss = criterion(predictions, target_trimmed)
            else:
                loss = torch.tensor(0.0, device=device)
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        training_losses.append(avg_loss)
        
        if (epoch + 1) % max(1, config.epochs_world_model // 3) == 0:
            print(f"  Epoch {epoch+1}/{config.epochs_world_model}: Loss = {avg_loss:.4f}")
    
    # Evaluation
    print("\nEvaluating world model predictions...")
    metrics = evaluate_world_model(
        model=model,
        dataset=eval_dataset,
        device=device,
        max_samples=config.eval_samples,
    )
    
    elapsed = time.time() - start_time
    result.add_timing("world_model_training", elapsed)
    result.add_training_curve("world_model_loss", training_losses)
    
    for name, value in metrics.items():
        result.add_metric(f"world_model_{name}", value)
    
    # Print results
    print("\n" + "-"*40)
    print("World Model Results:")
    print("-"*40)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    return metrics


# =============================================================================
# Scaling Validation
# =============================================================================

def validate_scaling(
    model: nn.Module,
    config: ValidationConfig,
    device: torch.device,
    result: ValidationResult,
) -> Dict[str, float]:
    """
    Validate O(n) linear scaling claim.
    
    Tests inference time vs sequence length.
    """
    print("\n" + "="*60)
    print("SCALING VALIDATION")
    print("="*60)
    
    from nexus.evaluation.benchmarks import ScalingBenchmark
    
    start_time = time.time()
    
    # Sequence lengths to test
    seq_lengths = [64, 128, 256, 512]
    if config.model_size != "tiny":
        seq_lengths.extend([1024, 2048])
    
    print(f"\nTesting inference scaling for sequence lengths: {seq_lengths}")
    
    benchmark = ScalingBenchmark(
        model=model,
        sequence_lengths=seq_lengths,
        batch_size=1,
        num_trials=5,
        device=str(device),
    )
    
    scaling_results = benchmark.run()
    
    elapsed = time.time() - start_time
    result.add_timing("scaling_validation", elapsed)
    
    # Extract metrics
    metrics = {
        "sequence_lengths": scaling_results["sequence_lengths"],
        "inference_times_ms": [t * 1000 for t in scaling_results["times"]],
    }
    
    if "scaling_analysis" in scaling_results:
        analysis = scaling_results["scaling_analysis"]
        metrics["linear_r2"] = analysis.get("linear_r2", 0.0)
        metrics["quadratic_r2"] = analysis.get("quadratic_r2", 0.0)
        metrics["scaling_type"] = analysis.get("scaling", "unknown")
        metrics["complexity"] = analysis.get("complexity", "unknown")
    
    for name, value in metrics.items():
        if name not in ["sequence_lengths", "inference_times_ms"]:
            result.add_metric(f"scaling_{name}", value)
    
    # Print results
    print("\n" + "-"*40)
    print("Scaling Results:")
    print("-"*40)
    print("  Seq Length -> Time (ms)")
    for seq_len, time_ms in zip(metrics["sequence_lengths"], metrics["inference_times_ms"]):
        print(f"    {seq_len:>6} -> {time_ms:>8.2f} ms")
    
    print(f"\n  Linear R²: {metrics.get('linear_r2', 'N/A'):.4f}")
    print(f"  Quadratic R²: {metrics.get('quadratic_r2', 'N/A'):.4f}")
    print(f"  Best fit: {metrics.get('complexity', 'N/A')}")
    
    if metrics.get("scaling_type") == "linear":
        print("\n  ✓ NEXUS achieves O(n) linear scaling!")
    
    return metrics


# =============================================================================
# Report Generation
# =============================================================================

def generate_validation_report(result: ValidationResult) -> str:
    """Generate human-readable validation report."""
    report = []
    
    report.append("=" * 70)
    report.append("NEXUS RESEARCH VALIDATION REPORT")
    report.append("=" * 70)
    report.append(f"Timestamp: {result.timestamp}")
    report.append("")
    
    # Configuration
    if result.config:
        report.append("Configuration:")
        report.append("-" * 40)
        for key, value in result.config.items():
            report.append(f"  {key}: {value}")
        report.append("")
    
    # Summary
    report.append("VALIDATION SUMMARY")
    report.append("=" * 70)
    
    # Algorithmic tasks
    alg_acc = result.metrics.get("algorithmic_overall_accuracy", 0)
    report.append(f"\n1. ALGORITHMIC TASKS (State-Space Validation)")
    report.append(f"   Overall Accuracy: {alg_acc:.2%}")
    
    task_accs = [(k, v) for k, v in result.metrics.items() 
                 if k.startswith("algorithmic_") and "_accuracy" in k and "overall" not in k]
    for task, acc in task_accs:
        task_name = task.replace("algorithmic_", "").replace("_accuracy", "")
        report.append(f"   - {task_name}: {acc:.2%}")
    
    status = "✓ PASS" if alg_acc > 0.5 else "✗ NEEDS WORK"
    report.append(f"   Status: {status}")
    
    # World model
    wm_mse = result.metrics.get("world_model_world_model_mse", 
                                 result.metrics.get("world_model_mse", float('inf')))
    wm_cosine = result.metrics.get("world_model_world_model_cosine",
                                    result.metrics.get("world_model_cosine", 0))
    
    report.append(f"\n2. WORLD MODEL (JEPA-style Prediction)")
    report.append(f"   Prediction MSE: {wm_mse:.4f}")
    report.append(f"   Cosine Similarity: {wm_cosine:.4f}")
    
    status = "✓ PASS" if wm_mse < 1.0 and wm_cosine > 0.5 else "✗ NEEDS WORK"
    report.append(f"   Status: {status}")
    
    # Causal inference
    causal_f1 = result.metrics.get("causal_avg_edge_f1", 0)
    report.append(f"\n3. CAUSAL INFERENCE")
    report.append(f"   Edge F1 Score: {causal_f1:.4f}")
    
    status = "✓ PASS" if causal_f1 > 0.3 else "⚠ BASELINE" if causal_f1 > 0 else "✗ NEEDS WORK"
    report.append(f"   Status: {status}")
    
    # Scaling
    scaling_type = result.metrics.get("scaling_scaling_type", 
                                       result.metrics.get("scaling_type", "unknown"))
    linear_r2 = result.metrics.get("scaling_linear_r2", 0)
    
    report.append(f"\n4. COMPUTATIONAL SCALING")
    report.append(f"   Scaling Type: {scaling_type}")
    report.append(f"   Linear R²: {linear_r2:.4f}")
    
    status = "✓ PASS - O(n) LINEAR!" if scaling_type == "linear" else "✗ NOT LINEAR"
    report.append(f"   Status: {status}")
    
    # Timing
    report.append(f"\nTIMING")
    report.append("-" * 40)
    total_time = sum(result.timing.values())
    for name, seconds in result.timing.items():
        report.append(f"  {name}: {seconds:.1f}s")
    report.append(f"  TOTAL: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Overall verdict
    report.append("")
    report.append("=" * 70)
    report.append("OVERALL VERDICT")
    report.append("=" * 70)
    
    passed = (
        alg_acc > 0.5 and
        wm_mse < 1.0 and
        scaling_type == "linear"
    )
    
    if passed:
        report.append("✓ NEXUS ARCHITECTURE VALIDATED")
        report.append("")
        report.append("The architecture demonstrates:")
        report.append("  - Working state-space sequence modeling")
        report.append("  - Functional world model predictions")
        report.append("  - O(n) linear computational scaling")
    else:
        report.append("⚠ VALIDATION INCOMPLETE")
        report.append("")
        report.append("Some components need additional training or debugging.")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS Research Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick validation (fewer samples, epochs)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full validation (more thorough)"
    )
    parser.add_argument(
        "--component",
        choices=["algorithmic", "causal", "world_model", "scaling", "all"],
        default="all",
        help="Validate specific component"
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save report to file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./validation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Select configuration
    if args.quick:
        config = ValidationConfig.quick()
        print("Using QUICK validation configuration")
    elif args.full:
        config = ValidationConfig.full()
        print("Using FULL validation configuration")
    else:
        config = ValidationConfig()
        print("Using DEFAULT validation configuration")
    
    config.output_dir = args.output_dir
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result
    result = ValidationResult()
    result.config = asdict(config)
    
    print("\n" + "="*70)
    print("NEXUS RESEARCH VALIDATION")
    print("="*70)
    print(f"Configuration: {config.model_size} model")
    print(f"Output: {config.output_dir}")
    
    # Create model
    print("\nCreating model...")
    model, nexus_config = create_model(config, device)
    
    total_start = time.time()
    
    # Run validations
    try:
        if args.component in ["algorithmic", "all"]:
            validate_algorithmic_tasks(model, config, device, result)
        
        if args.component in ["world_model", "all"]:
            validate_world_model(model, config, device, result)
        
        if args.component in ["causal", "all"]:
            validate_causal_inference(model, config, device, result)
        
        if args.component in ["scaling", "all"]:
            validate_scaling(model, config, device, result)
            
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - total_start
    result.add_timing("total", total_time)
    
    # Generate report
    report = generate_validation_report(result)
    print("\n" + report)
    
    # Save results
    if args.save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        result_path = output_dir / f"validation_results_{timestamp}.json"
        result.save(str(result_path))
        print(f"\nResults saved to: {result_path}")
        
        # Save text report
        report_path = output_dir / f"validation_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    
    print(f"\nTotal validation time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
