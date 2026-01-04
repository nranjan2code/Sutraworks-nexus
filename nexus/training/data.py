"""
NEXUS Data Pipeline
====================

Data loading and preprocessing utilities for NEXUS training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    max_seq_len: int = 2048
    context_ratio: float = 0.5
    mask_ratio: float = 0.15
    pad_token_id: int = 0
    mask_token_id: int = 1


class NEXUSDataset(Dataset):
    """
    Dataset for NEXUS training.
    
    Supports multiple data formats:
    - Token sequences (language modeling)
    - Continuous features (regression/embedding tasks)
    - Structured data (graphs, tables)
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        config: DataConfig,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of data samples
            config: Data configuration
            tokenizer: Optional tokenizer for text data
        """
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Handle different data types
        if "text" in sample and self.tokenizer is not None:
            return self._process_text(sample)
        elif "input_ids" in sample:
            return self._process_tokens(sample)
        elif "features" in sample:
            return self._process_features(sample)
        else:
            # Default: assume token format
            return self._process_tokens(sample)
            
    def _process_text(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process text sample."""
        text = sample["text"]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Create labels (shifted for causal LM)
        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100
        
        # Create context/target masks for world model
        seq_len = (attention_mask.sum()).item()
        context_len = int(seq_len * self.config.context_ratio)
        
        context_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        context_mask[:context_len] = True
        
        target_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        target_mask[context_len:seq_len] = True
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }
        
    def _process_tokens(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process pre-tokenized sample."""
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        
        # Pad or truncate
        if len(input_ids) > self.config.max_seq_len:
            input_ids = input_ids[:self.config.max_seq_len]
        elif len(input_ids) < self.config.max_seq_len:
            padding = torch.full(
                (self.config.max_seq_len - len(input_ids),),
                self.config.pad_token_id,
                dtype=torch.long,
            )
            input_ids = torch.cat([input_ids, padding])
            
        # Create attention mask
        attention_mask = (input_ids != self.config.pad_token_id).long()
        
        # Labels
        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100
        
        # Context/target masks
        seq_len = attention_mask.sum().item()
        context_len = int(seq_len * self.config.context_ratio)
        
        context_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        context_mask[:context_len] = True
        
        target_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        target_mask[context_len:seq_len] = True
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }
        
    def _process_features(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process continuous feature sample."""
        features = torch.tensor(sample["features"], dtype=torch.float32)
        
        # Ensure correct shape
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Add sequence dimension
            
        seq_len, d_model = features.shape
        
        # Pad if needed
        if seq_len < self.config.max_seq_len:
            padding = torch.zeros(
                self.config.max_seq_len - seq_len,
                d_model,
                dtype=torch.float32,
            )
            features = torch.cat([features, padding], dim=0)
        elif seq_len > self.config.max_seq_len:
            features = features[:self.config.max_seq_len]
            seq_len = self.config.max_seq_len
            
        # Attention mask
        attention_mask = torch.zeros(self.config.max_seq_len, dtype=torch.long)
        attention_mask[:seq_len] = 1
        
        # Context/target masks
        context_len = int(seq_len * self.config.context_ratio)
        
        context_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        context_mask[:context_len] = True
        
        target_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        target_mask[context_len:seq_len] = True
        
        return {
            "features": features,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }


class SyntheticNEXUSDataset(Dataset):
    """
    Synthetic dataset for testing NEXUS training.
    
    Generates:
    - Random token sequences
    - Synthetic reasoning tasks
    - Causal structure learning data
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        config: DataConfig = None,
        vocab_size: int = 50000,
        task_type: str = "language_modeling",
    ):
        self.num_samples = num_samples
        self.config = config or DataConfig()
        self.vocab_size = vocab_size
        self.task_type = task_type
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a synthetic sample."""
        
        if self.task_type == "language_modeling":
            return self._generate_lm_sample()
        elif self.task_type == "reasoning":
            return self._generate_reasoning_sample()
        elif self.task_type == "causal":
            return self._generate_causal_sample()
        else:
            return self._generate_lm_sample()
            
    def _generate_lm_sample(self) -> Dict[str, torch.Tensor]:
        """Generate synthetic language modeling sample."""
        # Random sequence length
        seq_len = random.randint(
            self.config.max_seq_len // 4,
            self.config.max_seq_len,
        )
        
        # Random tokens (avoiding special tokens 0, 1)
        input_ids = torch.randint(
            2, self.vocab_size,
            (seq_len,),
            dtype=torch.long,
        )
        
        # Pad to max length
        if seq_len < self.config.max_seq_len:
            padding = torch.full(
                (self.config.max_seq_len - seq_len,),
                self.config.pad_token_id,
                dtype=torch.long,
            )
            input_ids = torch.cat([input_ids, padding])
            
        attention_mask = (input_ids != self.config.pad_token_id).long()
        
        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100
        
        # Context/target masks
        context_len = int(seq_len * self.config.context_ratio)
        
        context_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        context_mask[:context_len] = True
        
        target_mask = torch.zeros(self.config.max_seq_len, dtype=torch.bool)
        target_mask[context_len:seq_len] = True
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }
        
    def _generate_reasoning_sample(self) -> Dict[str, torch.Tensor]:
        """Generate synthetic reasoning sample."""
        sample = self._generate_lm_sample()
        
        # Add reasoning target (mean of input embeddings as simple target)
        reasoning_target = torch.randn(512)  # d_model dimension
        sample["reasoning_target"] = reasoning_target
        
        return sample
        
    def _generate_causal_sample(self) -> Dict[str, torch.Tensor]:
        """Generate synthetic causal learning sample."""
        sample = self._generate_lm_sample()
        
        # Add causal structure (simplified)
        n_vars = 32
        causal_graph = torch.zeros(n_vars, n_vars)
        
        # Create random DAG
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if random.random() < 0.2:  # 20% edge probability
                    causal_graph[i, j] = 1
                    
        sample["causal_graph"] = causal_graph
        
        return sample


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for NEXUS batches."""
    keys = batch[0].keys()
    
    collated = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        else:
            collated[key] = [sample[key] for sample in batch]
            
    return collated
