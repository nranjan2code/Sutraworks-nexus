"""NEXUS Training Pipeline."""

from nexus.training.trainer import NEXUSTrainer, TrainingConfig
from nexus.training.data import NEXUSDataset, DataConfig, SyntheticNEXUSDataset, create_dataloader
from nexus.training.losses import NEXUSLoss, JEPALoss, ContrastiveLoss, CausalLoss
from nexus.training.continual import ContinualConfig, ContinualLearner

__all__ = [
    # Trainer
    "NEXUSTrainer",
    "TrainingConfig",
    # Data
    "NEXUSDataset",
    "DataConfig",
    "SyntheticNEXUSDataset",
    "create_dataloader",
    # Losses
    "NEXUSLoss",
    "JEPALoss",
    "ContrastiveLoss",
    "CausalLoss",
    # Continual Learning
    "ContinualConfig",
    "ContinualLearner",
]
