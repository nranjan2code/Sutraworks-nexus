"""
Hierarchical World Model
=========================

Implements a Joint Embedding Predictive Architecture (JEPA) inspired world model
that learns abstract representations of the world for prediction and planning.

Key innovations:
- Predicts in abstract representation space (not pixel/token space)
- Hierarchical multi-scale temporal abstractions
- Contrastive energy-based learning
- Enables planning through imagination

Unlike generative models that predict raw outputs, this model predicts abstract
representations, avoiding hallucination of irrelevant details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class WorldModelConfig:
    """Configuration for Hierarchical World Model."""
    
    d_model: int = 256          # Model dimension
    d_latent: int = 128         # Latent state dimension
    n_levels: int = 3           # Number of hierarchy levels
    context_ratio: float = 0.5  # Ratio of context to total sequence
    predictor_depth: int = 4    # Depth of predictor network
    n_heads: int = 8            # Number of attention heads
    dropout: float = 0.1
    ema_decay: float = 0.996    # EMA decay for target encoder
    temperature: float = 0.07   # Contrastive temperature


class ContextEncoder(nn.Module):
    """
    Encodes observable context into abstract representations.
    
    Uses a Vision Transformer-like architecture adapted for general sequences.
    Only processes visible/context portions of input.
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Position encoding - support up to 8192 positions
        self.pos_embed = nn.Parameter(
            torch.randn(1, 8192, config.d_model) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.predictor_depth
        )
        
        self.norm = nn.LayerNorm(config.d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode context into representations.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional context mask
            
        Returns:
            Context representations (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class TargetEncoder(nn.Module):
    """
    Encodes target regions into representations.
    
    This encoder is updated via exponential moving average (EMA) of the
    context encoder, providing stable targets for representation learning.
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        # Same architecture as context encoder
        self.pos_embed = nn.Parameter(
            torch.randn(1, 8192, config.d_model) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.predictor_depth
        )
        
        self.norm = nn.LayerNorm(config.d_model)
        
        # Target encoder doesn't require gradients (EMA updated)
        for param in self.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode targets (no gradients)."""
        batch, seq_len, _ = x.shape
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x


class Predictor(nn.Module):
    """
    Predicts target representations from context representations.
    
    This is the "world model" that learns to predict what abstract features
    should be present in unobserved regions, conditioned on observed context.
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Position tokens for target locations
        self.target_pos_embed = nn.Parameter(
            torch.randn(1, 8192, config.d_model) * 0.02
        )
        
        # Cross-attention predictor
        self.layers = nn.ModuleList()
        for _ in range(config.predictor_depth):
            self.layers.append(
                PredictorBlock(config)
            )
            
        self.norm = nn.LayerNorm(config.d_model)
        
        # Project to latent space for prediction
        self.proj = nn.Linear(config.d_model, config.d_latent)
        
    def forward(
        self,
        context: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict representations for target positions.
        
        Args:
            context: Context encoder output (batch, ctx_len, d_model)
            target_positions: Positions to predict (batch, n_targets)
            
        Returns:
            Predicted representations (batch, n_targets, d_latent)
        """
        batch, n_targets = target_positions.shape
        
        # Initialize target queries from positional embeddings
        # Gather embeddings at target positions
        target_queries = self.target_pos_embed[:, :n_targets, :].expand(batch, -1, -1)
        
        # Cross-attend to context
        x = target_queries
        for layer in self.layers:
            x = layer(x, context)
            
        x = self.norm(x)
        x = self.proj(x)
        
        return x


class PredictorBlock(nn.Module):
    """Single block of the predictor with self and cross attention."""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        # Self-attention among predictions
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        
        # Cross-attention to context
        self.cross_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.norm3 = nn.LayerNorm(config.d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through predictor block."""
        # Self attention
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        
        # Cross attention
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, context, context)[0]
        
        # FFN
        x_norm = self.norm3(x)
        x = x + self.ffn(x_norm)
        
        return x


class HierarchicalWorldModel(nn.Module):
    """
    Hierarchical World Model with multi-scale temporal abstraction.
    
    This is a core component of NEXUS that learns to:
    1. Encode observations into abstract representations
    2. Predict future/unobserved representations from context
    3. Model the world at multiple temporal scales
    
    Key differences from LLMs:
    - Predicts in representation space, not token space
    - Uses contrastive learning, not autoregressive generation
    - Learns hierarchical temporal abstractions
    - Enables planning through "imagination"
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        
        # Context encoder (learned)
        self.context_encoder = ContextEncoder(config)
        
        # Target encoder (EMA of context encoder)
        self.target_encoder = TargetEncoder(config)
        
        # Predictor network
        self.predictor = Predictor(config)
        
        # Multi-scale temporal abstraction layers
        self.temporal_abstractions = nn.ModuleList([
            TemporalAbstraction(config, level=i)
            for i in range(config.n_levels)
        ])
        
        # Projection for target representations
        self.target_proj = nn.Linear(config.d_model, config.d_latent)
        
        # Initialize target encoder from context encoder
        self._init_target_encoder()
        
    def _init_target_encoder(self):
        """Initialize target encoder as copy of context encoder."""
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            target_param.data.copy_(context_param.data)
            
    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder via EMA."""
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            target_param.data.mul_(self.config.ema_decay)
            target_param.data.add_(
                (1 - self.config.ema_decay) * context_param.data
            )
            
    def forward(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x: Input sequence (batch, seq_len, d_model)
            context_mask: Boolean mask for context positions
            target_mask: Boolean mask for target positions
            
        Returns:
            Dictionary with predictions and targets for loss computation
        """
        batch, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Extract context and target portions
        # context_mask: True where context is visible
        context_x = x * context_mask.unsqueeze(-1).float()
        
        # Encode context (with gradient)
        context_repr = self.context_encoder(context_x)
        
        # Encode targets (no gradient, EMA updated)
        with torch.no_grad():
            target_repr = self.target_encoder(x)
            target_repr = self.target_proj(target_repr)
            # Extract only target positions
            target_repr = target_repr * target_mask.unsqueeze(-1).float()
        
        # Get target positions (indices where target_mask is True)
        # Simplified: predict all positions, loss computed only on targets
        target_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        target_positions = target_positions.unsqueeze(0).expand(batch, -1)
        
        # Predict target representations from context
        predicted_repr = self.predictor(context_repr, target_positions)
        
        # Apply hierarchical temporal abstractions
        multi_scale_repr = []
        for temporal_layer in self.temporal_abstractions:
            abstracted = temporal_layer(context_repr)
            multi_scale_repr.append(abstracted)
        
        return {
            "predicted": predicted_repr,
            "target": target_repr,
            "context": context_repr,
            "multi_scale": multi_scale_repr,
            "target_mask": target_mask,
        }
        
    def predict(
        self,
        context: torch.Tensor,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """
        Predict future representations (inference mode).
        
        This enables "imagination" - predicting what future states
        might look like without actually observing them.
        """
        context_repr = self.context_encoder(context)
        
        predictions = []
        current_context = context_repr
        
        for _ in range(n_steps):
            # Predict next representations
            batch = current_context.shape[0]
            target_pos = torch.arange(
                current_context.shape[1],
                current_context.shape[1] + 1,
                device=context.device
            ).unsqueeze(0).expand(batch, -1)
            
            pred = self.predictor(current_context, target_pos)
            predictions.append(pred)
            
            # Update context with prediction (autoregressive imagination)
            # In practice, would need to unproject pred back to d_model
            
        return torch.cat(predictions, dim=1)


class TemporalAbstraction(nn.Module):
    """
    Temporal abstraction layer for multi-scale world modeling.
    
    Different levels capture different temporal scales:
    - Level 0: Fine-grained, step-by-step transitions
    - Level 1: Medium-scale patterns and routines
    - Level 2: High-level goals and extended sequences
    """
    
    def __init__(self, config: WorldModelConfig, level: int):
        super().__init__()
        self.level = level
        
        # Pooling window increases with level
        self.pool_size = 2 ** (level + 1)
        
        # Abstraction network
        self.abstract = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_latent),
        )
        
        # Temporal convolution at this scale
        self.temporal_conv = nn.Conv1d(
            config.d_latent,
            config.d_latent,
            kernel_size=3,
            padding=1,
            groups=config.d_latent // 4,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract input to this temporal scale.
        
        Args:
            x: Input (batch, seq_len, d_model)
            
        Returns:
            Abstracted representation at this temporal scale
        """
        batch, seq_len, d_model = x.shape
        
        # Temporal pooling
        if seq_len >= self.pool_size:
            # Average pooling over time
            x_pooled = F.avg_pool1d(
                x.transpose(1, 2),
                kernel_size=self.pool_size,
                stride=self.pool_size,
            ).transpose(1, 2)
        else:
            x_pooled = x.mean(dim=1, keepdim=True)
            
        # Abstract
        x_abstract = self.abstract(x_pooled)
        
        # Temporal convolution
        x_conv = self.temporal_conv(
            x_abstract.transpose(1, 2)
        ).transpose(1, 2)
        
        return x_conv


def jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    JEPA-style representation prediction loss.
    
    Computes smooth L1 loss between predicted and target representations,
    only at masked (target) positions.
    """
    # Normalize representations
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    
    # Compute loss only at target positions
    mask = mask.unsqueeze(-1).float()
    
    # Smooth L1 loss (Huber loss)
    loss = F.smooth_l1_loss(
        predicted * mask,
        target * mask,
        reduction="sum",
    )
    
    # Normalize by number of target positions
    n_targets = mask.sum().clamp(min=1)
    loss = loss / n_targets
    
    return loss
