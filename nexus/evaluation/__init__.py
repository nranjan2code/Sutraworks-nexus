"""NEXUS Evaluation and Benchmarking."""

from nexus.evaluation.benchmarks import (
    NEXUSBenchmark,
    ReasoningBenchmark,
    LongContextBenchmark,
    CausalBenchmark,
    ScalingBenchmark,
)
from nexus.evaluation.metrics import (
    compute_perplexity,
    compute_reasoning_accuracy,
    compute_causal_accuracy,
)
from nexus.evaluation.algorithmic_tasks import (
    AlgorithmicTaskDataset,
    TaskType,
    AlgorithmicTaskConfig,
    create_algorithmic_benchmark,
    evaluate_algorithmic_task,
)
from nexus.evaluation.causal_tasks import (
    CausalValidationDataset,
    CausalStructure,
    CausalTaskConfig,
    StructuralCausalModel,
    create_causal_benchmark,
    evaluate_causal_discovery,
)
from nexus.evaluation.world_model_tasks import (
    WorldModelValidationDataset,
    WorldModelTaskConfig,
    WorldModelTaskType,
    create_world_model_benchmark,
    evaluate_world_model,
)

__all__ = [
    # Benchmarks
    "NEXUSBenchmark",
    "ReasoningBenchmark",
    "LongContextBenchmark",
    "CausalBenchmark",
    "ScalingBenchmark",
    # Metrics
    "compute_perplexity",
    "compute_reasoning_accuracy",
    "compute_causal_accuracy",
    # Algorithmic tasks
    "AlgorithmicTaskDataset",
    "TaskType",
    "AlgorithmicTaskConfig",
    "create_algorithmic_benchmark",
    "evaluate_algorithmic_task",
    # Causal tasks
    "CausalValidationDataset",
    "CausalStructure",
    "CausalTaskConfig",
    "StructuralCausalModel",
    "create_causal_benchmark",
    "evaluate_causal_discovery",
    # World model tasks
    "WorldModelValidationDataset",
    "WorldModelTaskConfig",
    "WorldModelTaskType",
    "create_world_model_benchmark",
    "evaluate_world_model",
]
