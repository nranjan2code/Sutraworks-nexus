"""
NEXUS Training Demo

This script demonstrates how to train a NEXUS model from scratch,
showcasing the multi-objective training process with:
- State-space sequence modeling loss
- JEPA world model loss
- Reasoning grounding loss
- Energy-based regularization
- Causal consistency loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import argparse
from pathlib import Path

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(config_size: str = "small"):
    """Create NEXUS model with specified size."""
    from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
    
    configs = {
        "tiny": NEXUSConfig(
            vocab_size=32000,
            d_model=256,
            d_latent=128,
            ssm_n_layers=4,
            n_heads=4,
            ssm_d_state=32,
            max_seq_len=2048,
        ),
        "small": NEXUSConfig(
            vocab_size=32000,
            d_model=512,
            d_latent=256,
            ssm_n_layers=6,
            n_heads=8,
            ssm_d_state=64,
            max_seq_len=4096,
        ),
        "medium": NEXUSConfig(
            vocab_size=32000,
            d_model=1024,
            d_latent=512,
            ssm_n_layers=12,
            n_heads=16,
            ssm_d_state=128,
            max_seq_len=8192,
        ),
        "large": NEXUSConfig(
            vocab_size=32000,
            d_model=2048,
            d_latent=1024,
            ssm_n_layers=24,
            n_heads=32,
            ssm_d_state=256,
            max_seq_len=16384,
        ),
    }
    
    config = configs.get(config_size, configs["small"])
    model = NEXUSCore(config)
    
    return model, config


def create_synthetic_dataset(num_samples: int, seq_len: int, vocab_size: int):
    """Create synthetic dataset for demonstration."""
    from nexus.training.data import SyntheticNEXUSDataset, DataConfig
    
    config = DataConfig(max_seq_len=seq_len)
    dataset = SyntheticNEXUSDataset(
        num_samples=num_samples,
        config=config,
        vocab_size=vocab_size,
    )
    
    return dataset


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Perform a single training step."""
    model.train()
    optimizer.zero_grad()
    
    inputs = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(inputs)
        
        # Compute multi-objective loss
        targets = {'labels': labels}
        total_loss, loss_dict = loss_fn(outputs, targets)
    
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_metrics = {}
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        metrics = train_step(model, batch, optimizer, loss_fn, scaler, use_amp)
        
        # Accumulate metrics
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        
        # Log progress
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            avg_loss = total_metrics['total_loss'] / (batch_idx + 1)
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{num_batches} | Loss: {avg_loss:.4f}")
    
    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_metrics = {}
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            targets = {'labels': labels}
            total_loss, loss_dict = loss_fn(outputs, targets)
            
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_metrics[k] = total_metrics.get(k, 0) + val
    
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_metrics


def main(args):
    """Main training function."""
    print("=" * 60)
    print("NEXUS Training Demo")
    print("=" * 60)
    
    # Create model
    print(f"\n1. Creating {args.model_size} model...")
    model, config = create_model(args.model_size)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Create datasets
    print(f"\n2. Creating synthetic datasets...")
    train_dataset = create_synthetic_dataset(
        num_samples=args.train_samples,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
    )
    val_dataset = create_synthetic_dataset(
        num_samples=args.val_samples,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create optimizer and loss
    print(f"\n3. Setting up training...")
    from nexus.training.losses import NEXUSLoss
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    loss_fn = NEXUSLoss(
        lm_weight=1.0,
        world_model_weight=0.1,
        reasoning_weight=0.1,
        energy_weight=0.05,
        causal_weight=0.1,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate / 10,
    )
    
    # Optional: mixed precision training
    scaler = None
    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("   Using mixed precision training (AMP)")
    
    # Training loop
    print(f"\n4. Starting training for {args.epochs} epochs...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, epoch, scaler, use_amp
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch results
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['total_loss']:.4f}")
        print(f"  LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        # Print component losses
        print(f"  Components:")
        for key in ['lm_loss', 'jepa_loss', 'reasoning_loss', 'energy_loss', 'causal_loss']:
            if key in train_metrics:
                print(f"    {key}: {train_metrics[key]:.4f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            if args.save_dir:
                save_path = Path(args.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_loss': val_metrics['total_loss'],
                }, save_path / 'best_model.pt')
                print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    
    if args.save_dir:
        # Save final model
        save_path = Path(args.save_dir)
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'val_loss': val_metrics['total_loss'],
        }, save_path / 'final_model.pt')
        print(f"Final model saved to {save_path / 'final_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEXUS Training Demo")
    
    # Model settings
    parser.add_argument("--model-size", type=str, default="tiny",
                        choices=["tiny", "small", "medium", "large"],
                        help="Model size configuration")
    
    # Data settings
    parser.add_argument("--train-samples", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=200,
                        help="Number of validation samples")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loader workers")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use automatic mixed precision")
    
    # Output settings
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except ImportError as e:
        print(f"Error: Could not import NEXUS modules: {e}")
        print("Make sure to install the package first: pip install -e .")
