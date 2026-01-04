"""
Algorithmic Tasks for Research Validation
==========================================

Synthetic, structured tasks to prove NEXUS components work:
- Copy: Tests sequence memory (state-space)
- Reverse: Tests bidirectional state tracking
- Arithmetic: Tests compositional computation
- Pattern completion: Tests pattern recognition
- Variable binding: Tests symbolic grounding

These tasks have known ground truth, making them ideal for research validation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import torch
from torch.utils.data import Dataset
import numpy as np


class TaskType(Enum):
    """Types of algorithmic tasks."""
    COPY = "copy"
    REVERSE = "reverse"
    SORT = "sort"
    ARITHMETIC = "arithmetic"
    PATTERN = "pattern"
    VARIABLE_BINDING = "variable_binding"
    ASSOCIATIVE_RECALL = "associative_recall"


@dataclass
class AlgorithmicTaskConfig:
    """Configuration for algorithmic tasks."""
    max_seq_len: int = 256
    vocab_size: int = 100  # Small vocab for clean algorithmic tasks
    min_task_len: int = 8
    max_task_len: int = 64
    
    # Special tokens
    pad_token: int = 0
    sep_token: int = 1  # Separator between input and output
    bos_token: int = 2  # Beginning of sequence
    eos_token: int = 3  # End of sequence
    
    # Task-specific
    num_variables: int = 10  # For variable binding
    max_number: int = 50  # For arithmetic


class AlgorithmicTaskDataset(Dataset):
    """
    Dataset generating algorithmic tasks with ground truth.
    
    Each sample contains:
    - input_ids: Full sequence (input + separator + expected output)
    - labels: Same as input_ids (for teacher forcing)
    - task_type: Which algorithm this tests
    - input_len: Length of input portion
    - output_len: Length of output portion
    """
    
    def __init__(
        self,
        task_types: List[TaskType],
        num_samples: int = 10000,
        config: Optional[AlgorithmicTaskConfig] = None,
        difficulty: str = "medium",  # easy, medium, hard
    ):
        self.task_types = task_types
        self.num_samples = num_samples
        self.config = config or AlgorithmicTaskConfig()
        self.difficulty = difficulty
        
        # Adjust task length based on difficulty
        self.task_len_range = self._get_task_len_range()
        
    def _get_task_len_range(self) -> Tuple[int, int]:
        """Get task length range based on difficulty."""
        if self.difficulty == "easy":
            return (4, 16)
        elif self.difficulty == "medium":
            return (8, 32)
        else:  # hard
            return (16, 64)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a single algorithmic task sample."""
        # Select task type (round-robin or random)
        task_type = self.task_types[idx % len(self.task_types)]
        
        # Generate based on task type
        if task_type == TaskType.COPY:
            return self._generate_copy_task()
        elif task_type == TaskType.REVERSE:
            return self._generate_reverse_task()
        elif task_type == TaskType.SORT:
            return self._generate_sort_task()
        elif task_type == TaskType.ARITHMETIC:
            return self._generate_arithmetic_task()
        elif task_type == TaskType.PATTERN:
            return self._generate_pattern_task()
        elif task_type == TaskType.VARIABLE_BINDING:
            return self._generate_variable_binding_task()
        elif task_type == TaskType.ASSOCIATIVE_RECALL:
            return self._generate_associative_recall_task()
        else:
            return self._generate_copy_task()
    
    def _generate_copy_task(self) -> Dict[str, Any]:
        """
        Copy task: Input a sequence, output the same sequence.
        Tests: Basic sequence memory and state-space modeling.
        """
        min_len, max_len = self.task_len_range
        seq_len = random.randint(min_len, max_len)
        
        # Generate random sequence (avoiding special tokens)
        sequence = torch.randint(
            4, self.config.vocab_size, (seq_len,), dtype=torch.long
        )
        
        # Build full sequence: [BOS] input [SEP] output [EOS]
        full_seq = torch.cat([
            torch.tensor([self.config.bos_token], dtype=torch.long),
            sequence,
            torch.tensor([self.config.sep_token], dtype=torch.long),
            sequence.clone(),  # Copy as output
            torch.tensor([self.config.eos_token], dtype=torch.long),
        ])
        
        # Pad to max length
        padded = self._pad_sequence(full_seq)
        
        # Labels: -100 for input portion (don't compute loss), actual tokens for output
        labels = padded.clone()
        # Mask everything before separator (input portion)
        sep_pos = (padded == self.config.sep_token).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            labels[:sep_pos[0] + 1] = -100
        
        return {
            "input_ids": padded,
            "labels": labels,
            "attention_mask": (padded != self.config.pad_token).long(),
            "task_type": "copy",
            "input_len": seq_len,
            "output_len": seq_len,
        }
    
    def _generate_reverse_task(self) -> Dict[str, Any]:
        """
        Reverse task: Input a sequence, output it reversed.
        Tests: Bidirectional state tracking, memory.
        """
        min_len, max_len = self.task_len_range
        seq_len = random.randint(min_len, max_len)
        
        sequence = torch.randint(
            4, self.config.vocab_size, (seq_len,), dtype=torch.long
        )
        reversed_seq = torch.flip(sequence, dims=[0])
        
        full_seq = torch.cat([
            torch.tensor([self.config.bos_token], dtype=torch.long),
            sequence,
            torch.tensor([self.config.sep_token], dtype=torch.long),
            reversed_seq,
            torch.tensor([self.config.eos_token], dtype=torch.long),
        ])
        
        padded = self._pad_sequence(full_seq)
        labels = padded.clone()
        sep_pos = (padded == self.config.sep_token).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            labels[:sep_pos[0] + 1] = -100
        
        return {
            "input_ids": padded,
            "labels": labels,
            "attention_mask": (padded != self.config.pad_token).long(),
            "task_type": "reverse",
            "input_len": seq_len,
            "output_len": seq_len,
        }
    
    def _generate_sort_task(self) -> Dict[str, Any]:
        """
        Sort task: Input numbers, output them sorted.
        Tests: Compositional comparison operations.
        """
        min_len, max_len = self.task_len_range
        seq_len = random.randint(min_len, min(max_len, 20))  # Keep sorting tractable
        
        # Use a subset of vocab as "numbers" (4 to 4+max_number)
        sequence = torch.randint(
            4, 4 + self.config.max_number, (seq_len,), dtype=torch.long
        )
        sorted_seq = torch.sort(sequence)[0]
        
        full_seq = torch.cat([
            torch.tensor([self.config.bos_token], dtype=torch.long),
            sequence,
            torch.tensor([self.config.sep_token], dtype=torch.long),
            sorted_seq,
            torch.tensor([self.config.eos_token], dtype=torch.long),
        ])
        
        padded = self._pad_sequence(full_seq)
        labels = padded.clone()
        sep_pos = (padded == self.config.sep_token).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            labels[:sep_pos[0] + 1] = -100
        
        return {
            "input_ids": padded,
            "labels": labels,
            "attention_mask": (padded != self.config.pad_token).long(),
            "task_type": "sort",
            "input_len": seq_len,
            "output_len": seq_len,
        }
    
    def _generate_arithmetic_task(self) -> Dict[str, Any]:
        """
        Arithmetic task: Simple addition/subtraction.
        Tests: Compositional computation, carry operations.
        
        Format: [num1] [op] [num2] [SEP] [result]
        """
        max_num = self.config.max_number
        
        # Generate operands based on difficulty
        if self.difficulty == "easy":
            a = random.randint(1, 10)
            b = random.randint(1, 10)
        elif self.difficulty == "medium":
            a = random.randint(1, 30)
            b = random.randint(1, 30)
        else:
            a = random.randint(1, max_num)
            b = random.randint(1, max_num)
        
        # Choose operation (+ encoded as token 50, - as 51)
        op_add = 50
        op_sub = 51
        
        if random.random() < 0.5:
            op = op_add
            result = a + b
        else:
            op = op_sub
            if a < b:
                a, b = b, a  # Ensure non-negative result
            result = a - b
        
        # Encode numbers as tokens (offset by 4 for special tokens)
        # Simple encoding: each digit as a token
        def encode_number(n):
            if n == 0:
                return [4]  # Token 4 = digit 0
            digits = []
            while n > 0:
                digits.append(4 + (n % 10))
                n //= 10
            return list(reversed(digits))
        
        input_seq = encode_number(a) + [op] + encode_number(b)
        output_seq = encode_number(result)
        
        full_seq = torch.tensor(
            [self.config.bos_token] + input_seq + [self.config.sep_token] + 
            output_seq + [self.config.eos_token],
            dtype=torch.long
        )
        
        padded = self._pad_sequence(full_seq)
        labels = padded.clone()
        sep_pos = (padded == self.config.sep_token).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            labels[:sep_pos[0] + 1] = -100
        
        return {
            "input_ids": padded,
            "labels": labels,
            "attention_mask": (padded != self.config.pad_token).long(),
            "task_type": "arithmetic",
            "input_len": len(input_seq),
            "output_len": len(output_seq),
        }
    
    def _generate_pattern_task(self) -> Dict[str, Any]:
        """
        Pattern completion: Given a pattern, continue it.
        Tests: Pattern recognition, abstraction.
        
        Patterns: ABAB..., AABB..., ABC..., etc.
        """
        # Pattern types
        pattern_funcs = [
            self._pattern_repeat,      # ABAB...
            self._pattern_double,      # AABB...
            self._pattern_increasing,  # ABC... (increasing)
            self._pattern_fibonacci,   # Fibonacci-like
        ]
        
        pattern_func = random.choice(pattern_funcs)
        context, continuation, pattern_name = pattern_func()
        
        full_seq = torch.tensor(
            [self.config.bos_token] + context + [self.config.sep_token] + 
            continuation + [self.config.eos_token],
            dtype=torch.long
        )
        
        padded = self._pad_sequence(full_seq)
        labels = padded.clone()
        sep_pos = (padded == self.config.sep_token).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            labels[:sep_pos[0] + 1] = -100
        
        return {
            "input_ids": padded,
            "labels": labels,
            "attention_mask": (padded != self.config.pad_token).long(),
            "task_type": "pattern",
            "input_len": len(context),
            "output_len": len(continuation),
        }
    
    def _pattern_repeat(self) -> Tuple[List[int], List[int], str]:
        """ABAB... pattern."""
        base_len = random.randint(2, 4)
        base = [random.randint(4, 20) for _ in range(base_len)]
        
        num_repeats = random.randint(3, 6)
        context = base * num_repeats
        continuation = base * 2  # Predict 2 more repeats
        
        return context, continuation, "repeat"
    
    def _pattern_double(self) -> Tuple[List[int], List[int], str]:
        """AABB... pattern."""
        unique_tokens = random.randint(3, 6)
        tokens = [random.randint(4, 20) for _ in range(unique_tokens)]
        
        context = []
        for t in tokens:
            context.extend([t, t])
        
        next_tokens = [random.randint(4, 20) for _ in range(2)]
        continuation = []
        for t in next_tokens:
            continuation.extend([t, t])
        
        return context, continuation, "double"
    
    def _pattern_increasing(self) -> Tuple[List[int], List[int], str]:
        """Increasing sequence: 5, 7, 9, 11..."""
        start = random.randint(4, 15)
        step = random.randint(1, 4)
        
        context_len = random.randint(4, 8)
        context = [start + i * step for i in range(context_len)]
        
        cont_len = random.randint(2, 4)
        continuation = [start + (context_len + i) * step for i in range(cont_len)]
        
        # Clip to vocab range
        context = [min(x, self.config.vocab_size - 1) for x in context]
        continuation = [min(x, self.config.vocab_size - 1) for x in continuation]
        
        return context, continuation, "increasing"
    
    def _pattern_fibonacci(self) -> Tuple[List[int], List[int], str]:
        """Fibonacci-like: each element is sum of previous two (mod vocab)."""
        a = random.randint(4, 10)
        b = random.randint(4, 10)
        
        context = [a, b]
        for _ in range(6):
            next_val = (context[-1] + context[-2]) % (self.config.vocab_size - 4) + 4
            context.append(next_val)
        
        continuation = []
        for _ in range(3):
            next_val = (context[-1] + context[-2]) % (self.config.vocab_size - 4) + 4
            context.append(next_val)
            continuation.append(next_val)
        
        context = context[:-3]  # Remove what we added to continuation
        
        return context, continuation, "fibonacci"
    
    def _generate_variable_binding_task(self) -> Dict[str, Any]:
        """
        Variable binding: X=5, Y=3, X+Y=?
        Tests: Symbolic grounding, variable resolution.
        """
        # Variable tokens: 60-69 represent variables X0-X9
        # Assignment token: 70
        # Query token: 71
        
        var_base = 60
        assign_tok = 70
        query_tok = 71
        
        num_vars = random.randint(2, min(5, self.config.num_variables))
        
        # Create bindings
        bindings = {}
        input_seq = []
        for i in range(num_vars):
            var_tok = var_base + i
            value = random.randint(4, 15)  # Values encoded as tokens
            bindings[var_tok] = value
            input_seq.extend([var_tok, assign_tok, value])
        
        # Query: ask for one variable's value
        query_var = random.choice(list(bindings.keys()))
        input_seq.extend([query_tok, query_var])
        
        output_seq = [bindings[query_var]]
        
        full_seq = torch.tensor(
            [self.config.bos_token] + input_seq + [self.config.sep_token] + 
            output_seq + [self.config.eos_token],
            dtype=torch.long
        )
        
        padded = self._pad_sequence(full_seq)
        labels = padded.clone()
        sep_pos = (padded == self.config.sep_token).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            labels[:sep_pos[0] + 1] = -100
        
        return {
            "input_ids": padded,
            "labels": labels,
            "attention_mask": (padded != self.config.pad_token).long(),
            "task_type": "variable_binding",
            "input_len": len(input_seq),
            "output_len": len(output_seq),
        }
    
    def _generate_associative_recall_task(self) -> Dict[str, Any]:
        """
        Associative recall: Given key-value pairs, recall value for query key.
        Tests: Long-range memory, key-value retrieval.
        
        Format: [k1 v1 k2 v2 ... kn vn] [SEP] [query_key] [SEP] [value]
        """
        num_pairs = random.randint(3, 8)
        
        # Generate unique keys
        keys = random.sample(range(4, 30), num_pairs)
        values = [random.randint(30, 60) for _ in range(num_pairs)]
        
        # Build input: key-value pairs
        input_seq = []
        for k, v in zip(keys, values):
            input_seq.extend([k, v])
        
        # Add noise/distractors
        if self.difficulty != "easy":
            noise_len = random.randint(5, 15)
            noise = [random.randint(61, 80) for _ in range(noise_len)]
            input_seq.extend(noise)
        
        # Query
        query_idx = random.randint(0, num_pairs - 1)
        query_key = keys[query_idx]
        expected_value = values[query_idx]
        
        full_seq = torch.tensor(
            [self.config.bos_token] + input_seq + [self.config.sep_token] +
            [query_key] + [self.config.sep_token] + [expected_value] +
            [self.config.eos_token],
            dtype=torch.long
        )
        
        padded = self._pad_sequence(full_seq)
        labels = padded.clone()
        
        # Find the LAST separator (before the answer)
        sep_positions = (padded == self.config.sep_token).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 2:
            labels[:sep_positions[-1] + 1] = -100
        
        return {
            "input_ids": padded,
            "labels": labels,
            "attention_mask": (padded != self.config.pad_token).long(),
            "task_type": "associative_recall",
            "input_len": len(input_seq),
            "output_len": 1,
        }
    
    def _pad_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Pad sequence to max length."""
        if len(seq) >= self.config.max_seq_len:
            return seq[:self.config.max_seq_len]
        
        padding = torch.full(
            (self.config.max_seq_len - len(seq),),
            self.config.pad_token,
            dtype=torch.long
        )
        return torch.cat([seq, padding])


def create_algorithmic_benchmark(
    task_types: Optional[List[TaskType]] = None,
    num_samples: int = 1000,
    difficulty: str = "medium",
) -> AlgorithmicTaskDataset:
    """
    Factory function to create algorithmic task dataset.
    
    Args:
        task_types: List of tasks to include. None = all tasks.
        num_samples: Number of samples to generate.
        difficulty: "easy", "medium", or "hard"
    
    Returns:
        AlgorithmicTaskDataset
    """
    if task_types is None:
        task_types = list(TaskType)
    
    return AlgorithmicTaskDataset(
        task_types=task_types,
        num_samples=num_samples,
        difficulty=difficulty,
    )


def evaluate_algorithmic_task(
    model: torch.nn.Module,
    dataset: AlgorithmicTaskDataset,
    device: torch.device,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Evaluate model on algorithmic tasks.
    
    Returns per-task accuracy and overall metrics.
    """
    from collections import defaultdict
    
    model.eval()
    
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            sample = dataset[i]
            
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"].to(device)
            task_type = sample["task_type"]
            
            # Get model predictions
            outputs = model(input_ids)
            logits = outputs["logits"]
            
            # Find output region (after separator, where labels != -100)
            valid_mask = labels != -100
            if not valid_mask.any():
                continue
            
            # Get predicted tokens vs ground truth from labels
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            first_valid = valid_indices[0].item()
            last_valid = valid_indices[-1].item() + 1
            
            predicted = logits[0, first_valid:last_valid].argmax(dim=-1)
            ground_truth = labels[first_valid:last_valid]
            
            # Check correctness (all tokens must match)
            is_correct = (predicted == ground_truth).all().item()
            
            results[task_type]["total"] += 1
            if is_correct:
                results[task_type]["correct"] += 1
    
    # Compute accuracies
    metrics = {}
    total_correct = 0
    total_samples = 0
    
    for task_type, counts in results.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        metrics[f"{task_type}_accuracy"] = acc
        metrics[f"{task_type}_total"] = counts["total"]
        total_correct += counts["correct"]
        total_samples += counts["total"]
    
    metrics["overall_accuracy"] = total_correct / total_samples if total_samples > 0 else 0
    metrics["total_samples"] = total_samples
    
    return metrics
