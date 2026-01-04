# Evaluation Guide

## Comprehensive Evaluation for NEXUS Models

This document covers the complete evaluation framework for NEXUS, including benchmarks, metrics, and analysis tools.

---

## Evaluation Philosophy

NEXUS requires multi-dimensional evaluation because it combines multiple capabilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NEXUS Evaluation Dimensions                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │  Language   │  │  Reasoning  │  │   Causal    │  │ Efficiency  │      │
│   │  Modeling   │  │    Tasks    │  │  Inference  │  │  Metrics    │      │
│   ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤      │
│   │ • Perplexity│  │ • Logic     │  │ • Intervent │  │ • Latency   │      │
│   │ • Accuracy  │  │ • Math      │  │ • Counterfct│  │ • Memory    │      │
│   │ • BLEU      │  │ • Common    │  │ • Discovery │  │ • Throughput│      │
│   │ • BPC       │  │   Sense     │  │             │  │ • Params    │      │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                                             │
│                           ┌─────────────┐                                  │
│                           │  Adaptive   │                                  │
│                           │ Computation │                                  │
│                           ├─────────────┤                                  │
│                           │ • Iter dist │                                  │
│                           │ • Early exit│                                  │
│                           │ • Accuracy/ │                                  │
│                           │   compute   │                                  │
│                           └─────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```python
from nexus.core import NEXUSCore
from nexus.evaluation import NEXUSBenchmark

# Load model
model = NEXUSCore.from_pretrained('path/to/checkpoint')

# Run all benchmarks
benchmark = NEXUSBenchmark(model)
results = benchmark.run_all()

# Print summary
benchmark.print_summary(results)
```

---

## Benchmark Suite

### NEXUSBenchmark Class

```python
import torch
import numpy as np
from typing import Dict, Any
from tqdm import tqdm


class NEXUSBenchmark:
    """Comprehensive benchmark suite for NEXUS models."""
    
    def __init__(
        self,
        model,
        device: str = 'cuda',
        batch_size: int = 16,
        max_length: int = 2048
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.model.eval()
    
    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        
        results = {}
        
        print("Running language modeling benchmarks...")
        results['language_modeling'] = self.language_modeling()
        
        print("Running reasoning benchmarks...")
        results['reasoning'] = self.reasoning()
        
        print("Running causal inference benchmarks...")
        results['causal'] = self.causal_inference()
        
        print("Running efficiency benchmarks...")
        results['efficiency'] = self.efficiency()
        
        print("Running adaptive computation analysis...")
        results['adaptive'] = self.adaptive_computation()
        
        return results
    
    # === Language Modeling ===
    
    def language_modeling(self) -> Dict[str, float]:
        """Evaluate language modeling capabilities."""
        
        results = {}
        
        # Perplexity on various datasets
        datasets = ['wikitext', 'lambada', 'ptb']
        
        for dataset in datasets:
            try:
                ppl = self._compute_perplexity(dataset)
                results[f'{dataset}_perplexity'] = ppl
            except Exception as e:
                print(f"Skipping {dataset}: {e}")
        
        # Bits per character (BPC)
        results['bpc'] = self._compute_bpc()
        
        return results
    
    def _compute_perplexity(self, dataset_name: str) -> float:
        """Compute perplexity on a dataset."""
        
        # Load evaluation data (simplified)
        eval_data = self._load_eval_data(dataset_name)
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_data, desc=f"Perplexity ({dataset_name})"):
                batch = batch.to(self.device)
                
                logits = self.model(batch[:, :-1])
                targets = batch[:, 1:]
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += targets.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def _compute_bpc(self) -> float:
        """Compute bits per character."""
        # BPC = cross_entropy / ln(2)
        # Using enwik8 or similar character-level dataset
        pass
    
    # === Reasoning ===
    
    def reasoning(self) -> Dict[str, float]:
        """Evaluate reasoning capabilities."""
        
        results = {}
        
        # Logic reasoning
        results['logic'] = self._eval_logic_reasoning()
        
        # Mathematical reasoning
        results['math'] = self._eval_math_reasoning()
        
        # Common sense reasoning
        results['commonsense'] = self._eval_commonsense()
        
        return results
    
    def _eval_logic_reasoning(self) -> Dict[str, float]:
        """Evaluate logical reasoning."""
        
        # ProofWriter-style evaluation
        test_cases = [
            {
                'premises': ['All A are B', 'All B are C'],
                'query': 'All A are C',
                'answer': True
            },
            {
                'premises': ['Some A are B', 'All B are C'],
                'query': 'Some A are C',
                'answer': True
            },
            # More cases...
        ]
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            prediction = self._predict_logic(
                case['premises'],
                case['query']
            )
            if prediction == case['answer']:
                correct += 1
        
        return {
            'accuracy': correct / total,
            'total': total
        }
    
    def _eval_math_reasoning(self) -> Dict[str, float]:
        """Evaluate mathematical reasoning."""
        
        # GSM8K-style problems
        results = {
            'arithmetic': self._eval_arithmetic(),
            'word_problems': self._eval_word_problems(),
            'algebra': self._eval_algebra()
        }
        
        return results
    
    def _eval_commonsense(self) -> Dict[str, float]:
        """Evaluate common sense reasoning."""
        
        # CommonsenseQA, PIQA, etc.
        results = {
            'commonsenseqa': self._eval_dataset('commonsenseqa'),
            'piqa': self._eval_dataset('piqa'),
            'winogrande': self._eval_dataset('winogrande')
        }
        
        return results
    
    # === Causal Inference ===
    
    def causal_inference(self) -> Dict[str, float]:
        """Evaluate causal reasoning capabilities."""
        
        results = {}
        
        # Interventional accuracy
        results['interventional'] = self._eval_interventional()
        
        # Counterfactual reasoning
        results['counterfactual'] = self._eval_counterfactual()
        
        # Causal discovery
        results['discovery'] = self._eval_causal_discovery()
        
        return results
    
    def _eval_interventional(self) -> Dict[str, float]:
        """Evaluate do(X) predictions."""
        
        # Test cases with known causal structure
        test_cases = [
            {
                'graph': 'X → Y → Z',
                'intervention': 'do(Y=1)',
                'target': 'Z',
                'expected_change': True  # Z should change
            },
            {
                'graph': 'X → Y → Z',
                'intervention': 'do(Y=1)',
                'target': 'X',
                'expected_change': False  # X shouldn't change (no back-door)
            }
        ]
        
        correct = 0
        for case in test_cases:
            prediction = self._predict_intervention(case)
            if prediction['changes'] == case['expected_change']:
                correct += 1
        
        return {'accuracy': correct / len(test_cases)}
    
    def _eval_counterfactual(self) -> Dict[str, float]:
        """Evaluate counterfactual reasoning."""
        
        # "What if X had been different?"
        test_cases = [
            {
                'observation': {'treatment': 0, 'outcome': 0},
                'counterfactual': 'treatment=1',
                'query': 'outcome',
                'expected': 1
            }
        ]
        
        # Evaluate...
        return {'accuracy': 0.0}  # Placeholder
    
    def _eval_causal_discovery(self) -> Dict[str, float]:
        """Evaluate ability to discover causal structure."""
        
        # Generate data from known DAG, see if model can recover it
        true_dag = self._generate_random_dag(num_nodes=5)
        data = self._sample_from_dag(true_dag, num_samples=1000)
        
        # Get model's prediction
        predicted_dag = self.model.causal.discover_structure(data)
        
        # Compare
        metrics = self._dag_metrics(true_dag, predicted_dag)
        
        return metrics
    
    # === Efficiency ===
    
    def efficiency(self) -> Dict[str, float]:
        """Evaluate computational efficiency."""
        
        results = {}
        
        # Latency at various sequence lengths
        for seq_len in [128, 512, 1024, 2048, 4096]:
            latency = self._measure_latency(seq_len)
            results[f'latency_{seq_len}'] = latency
        
        # Memory usage
        results['peak_memory_mb'] = self._measure_memory()
        
        # Throughput
        results['tokens_per_second'] = self._measure_throughput()
        
        # Parameter count
        results['parameters_millions'] = sum(
            p.numel() for p in self.model.parameters()
        ) / 1e6
        
        # FLOPs estimate
        results['gflops_per_token'] = self._estimate_flops()
        
        return results
    
    def _measure_latency(self, seq_len: int, num_runs: int = 100) -> float:
        """Measure inference latency."""
        
        # Warmup
        dummy_input = torch.randint(0, 1000, (1, seq_len)).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Synchronize
        torch.cuda.synchronize()
        
        # Measure
        import time
        times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
        
        return np.mean(times)
    
    def _measure_memory(self) -> float:
        """Measure peak GPU memory usage."""
        
        torch.cuda.reset_peak_memory_stats()
        
        # Run forward pass with max sequence length
        dummy_input = torch.randint(
            0, 1000, (self.batch_size, self.max_length)
        ).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return peak_memory
    
    def _measure_throughput(self) -> float:
        """Measure tokens per second."""
        
        import time
        
        total_tokens = 0
        total_time = 0
        
        for _ in range(10):
            batch = torch.randint(
                0, 1000, (self.batch_size, self.max_length)
            ).to(self.device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(batch)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            total_tokens += self.batch_size * self.max_length
            total_time += end - start
        
        return total_tokens / total_time
    
    # === Adaptive Computation ===
    
    def adaptive_computation(self) -> Dict[str, Any]:
        """Analyze adaptive computation behavior."""
        
        results = {}
        
        # Iteration distribution
        results['iteration_distribution'] = self._analyze_iterations()
        
        # Early exit rate
        results['early_exit_rate'] = self._analyze_early_exit()
        
        # Accuracy vs compute trade-off
        results['accuracy_compute_curve'] = self._analyze_accuracy_compute()
        
        return results
    
    def _analyze_iterations(self) -> Dict[str, Any]:
        """Analyze how many iterations are used."""
        
        iteration_counts = []
        
        # Run on diverse inputs
        for batch in self._get_diverse_inputs():
            with torch.no_grad():
                _, info = self.model(batch, return_all_outputs=True)
                
                if 'energy_info' in info:
                    iteration_counts.append(info['energy_info']['iterations'])
        
        return {
            'mean': np.mean(iteration_counts),
            'std': np.std(iteration_counts),
            'min': min(iteration_counts),
            'max': max(iteration_counts),
            'histogram': np.histogram(iteration_counts, bins=10)
        }
    
    # === Utilities ===
    
    def _load_eval_data(self, dataset_name: str):
        """Load evaluation dataset."""
        # Implementation depends on data format
        pass
    
    def _get_diverse_inputs(self):
        """Generate diverse inputs for testing."""
        # Mix of simple and complex inputs
        pass
    
    def print_summary(self, results: Dict[str, Any]):
        """Print formatted results summary."""
        
        print("\n" + "=" * 60)
        print("NEXUS Evaluation Summary")
        print("=" * 60)
        
        for category, metrics in results.items():
            print(f"\n{category.upper()}")
            print("-" * 40)
            
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {name}: {value:.4f}")
                    else:
                        print(f"  {name}: {value}")
            else:
                print(f"  {metrics}")
        
        print("\n" + "=" * 60)
```

---

## Standard Metrics

### Language Modeling Metrics

```python
def compute_perplexity(model, data, device='cuda'):
    """Compute perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    return np.exp(total_loss / total_tokens)


def compute_accuracy(model, data, device='cuda'):
    """Compute next-token prediction accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            predictions = logits.argmax(dim=-1)
            targets = batch[:, 1:]
            
            correct += (predictions == targets).sum().item()
            total += targets.numel()
    
    return correct / total
```

### Reasoning Metrics

```python
def compute_reasoning_metrics(model, test_cases):
    """Compute reasoning benchmark metrics."""
    
    results = {
        'correct': 0,
        'total': len(test_cases),
        'by_difficulty': {'easy': 0, 'medium': 0, 'hard': 0},
        'by_type': {}
    }
    
    for case in test_cases:
        prediction = model.reason(case['input'])
        is_correct = prediction == case['expected']
        
        if is_correct:
            results['correct'] += 1
            results['by_difficulty'][case.get('difficulty', 'medium')] += 1
        
        case_type = case.get('type', 'general')
        if case_type not in results['by_type']:
            results['by_type'][case_type] = {'correct': 0, 'total': 0}
        
        results['by_type'][case_type]['total'] += 1
        if is_correct:
            results['by_type'][case_type]['correct'] += 1
    
    results['accuracy'] = results['correct'] / results['total']
    
    return results
```

### Efficiency Metrics

```python
def compute_efficiency_metrics(model, seq_lengths=[128, 512, 1024, 2048]):
    """Compute comprehensive efficiency metrics."""
    
    import time
    
    results = {
        'latency': {},
        'memory': {},
        'throughput': {}
    }
    
    for seq_len in seq_lengths:
        # Prepare input
        batch = torch.randint(0, 1000, (1, seq_len)).cuda()
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(batch)
        
        torch.cuda.synchronize()
        
        # Latency
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        results['latency'][seq_len] = np.mean(times) * 1000  # ms
        
        # Memory
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(batch)
        results['memory'][seq_len] = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # Throughput
        results['throughput'][seq_len] = seq_len / np.mean(times)  # tokens/sec
    
    return results
```

---

## Comparison Framework

### Compare with Baselines

```python
class ModelComparison:
    """Compare NEXUS with baseline models."""
    
    def __init__(self, models: Dict[str, nn.Module]):
        self.models = models
        self.benchmark = NEXUSBenchmark
    
    def compare_all(self, dataset):
        """Run all models on same benchmark."""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            benchmark = self.benchmark(model)
            results[name] = benchmark.run_all()
        
        return results
    
    def generate_report(self, results):
        """Generate comparison report."""
        
        # Create comparison table
        metrics = ['perplexity', 'reasoning_accuracy', 'latency', 'memory']
        
        table = []
        for model_name, model_results in results.items():
            row = {'model': model_name}
            for metric in metrics:
                row[metric] = self._extract_metric(model_results, metric)
            table.append(row)
        
        return table
    
    def plot_comparison(self, results):
        """Plot comparison charts."""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Perplexity comparison
        # Latency vs accuracy
        # Memory usage
        # etc.
        
        return fig
```

---

## Scaling Analysis

### Analyze Scaling Behavior

```python
class ScalingAnalysis:
    """Analyze how NEXUS scales."""
    
    def __init__(self, model_configs: list):
        self.configs = model_configs
    
    def run_scaling_analysis(self):
        """Test different model sizes."""
        
        results = []
        
        for config in self.configs:
            model = NEXUSCore(**config)
            
            # Train briefly
            # ...
            
            # Evaluate
            benchmark = NEXUSBenchmark(model)
            eval_results = benchmark.run_all()
            
            results.append({
                'config': config,
                'params': self._count_params(model),
                'results': eval_results
            })
        
        return results
    
    def plot_scaling_curves(self, results):
        """Plot parameter count vs performance."""
        
        import matplotlib.pyplot as plt
        
        params = [r['params'] for r in results]
        perplexity = [r['results']['language_modeling']['wikitext_perplexity'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(params, perplexity, 'o-')
        plt.xlabel('Parameters')
        plt.ylabel('Perplexity')
        plt.title('NEXUS Scaling Behavior')
        
        return plt.gcf()
```

---

## Reporting

### Generate Evaluation Report

```python
def generate_report(results: Dict, output_path: str = 'evaluation_report.md'):
    """Generate markdown evaluation report."""
    
    report = []
    report.append("# NEXUS Evaluation Report\n")
    report.append(f"Generated: {datetime.now().isoformat()}\n\n")
    
    # Summary
    report.append("## Summary\n")
    report.append(f"- Perplexity: {results['language_modeling']['wikitext_perplexity']:.2f}\n")
    report.append(f"- Reasoning Accuracy: {results['reasoning']['logic']['accuracy']:.2%}\n")
    report.append(f"- Latency (1024 tokens): {results['efficiency']['latency_1024']:.1f}ms\n")
    
    # Detailed results
    report.append("\n## Detailed Results\n")
    
    for category, metrics in results.items():
        report.append(f"\n### {category.replace('_', ' ').title()}\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                report.append(f"| {name} | {value:.4f} |\n")
            else:
                report.append(f"| {name} | {value} |\n")
    
    # Write
    with open(output_path, 'w') as f:
        f.write(''.join(report))
    
    return output_path
```

---

## Example Evaluation Script

```python
#!/usr/bin/env python
"""Evaluate NEXUS model."""

import argparse
from nexus.core import NEXUSCore
from nexus.evaluation import NEXUSBenchmark, generate_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='results.json')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = NEXUSCore.from_pretrained(args.checkpoint)
    
    # Run benchmarks
    benchmark = NEXUSBenchmark(model)
    results = benchmark.run_all()
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report_path = generate_report(results)
    
    # Print summary
    benchmark.print_summary(results)
    print(f"\nResults saved to {args.output}")
    print(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()
```

---

## Next Steps

- [Optimization Guide](optimization.md) - Improve performance
- [API Reference](../api/evaluation.md) - Evaluation API
- [Benchmarks Reference](benchmarks-reference.md) - Benchmark details

---

*Evaluation reveals truth. Test rigorously.*
