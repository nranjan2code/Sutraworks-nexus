"""
Train NEXUS-Reason-Alpha
========================

Main entry point for training the SML on GCP L4.
"""

import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
import argparse

from nexus.configs.alpha import get_config
from nexus.core.nexus_core import NEXUSCore
from nexus.training.trainer import NEXUSTrainer, TrainingConfig

# Constants
WANDB_PROJECT = "nexus-reason-alpha"
DATASET_NAME = "HuggingFaceFW/fineweb-edu" # High quality subset
DATASET_CONFIG = "sample-10BT" # Start small for testing, switch to full for real run

def setup_data(tokenizer, max_seq_len=4096, batch_size=8, num_workers=4):
    """Prepare FineWeb-Edu dataset."""
    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    def tokenization_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
    
    # Create iterable dataset with shuffling
    tokenized_dataset = dataset.map(tokenization_fn, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.shuffle(buffer_size=10000)
    
    # Custom collate for streaming
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        labels = input_ids.clone() # Autoregressive
        return {"input_ids": input_ids, "labels": labels}
        
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=100000, help="Max training steps")
    args = parser.parse_args()

    # 1. Configuration
    nexus_config = get_config()
    
    training_config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=4,             # Small per-device batch 
        gradient_accumulation_steps=32, # Effective batch = 4 * 32 = 128 (Better stability)
        max_steps=args.steps,
        warmup_steps=2000,
        checkpoint_dir="./checkpoints/alpha",
        wandb_project=WANDB_PROJECT,
        mixed_precision=True,     # Use AMP (BF16 if supported)
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {training_config.device}")
    
    # 2. Tokenizer (Using Llama 2 tokenizer for Pi 5/Edge compatibility)
    # Using 'mistralai/Mistral-7B-v0.1' tokenizer which is Llama-compatible and high quality
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    except Exception:
        # Fallback if gated/offline
        print("Warning: Could not load Mistral tokenizer. Falling back to open Llama tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
        
    tokenizer.pad_token = tokenizer.eos_token

    
    # 3. Data
    train_loader = setup_data(
        tokenizer, 
        max_seq_len=nexus_config.max_seq_len,
        batch_size=training_config.batch_size
    )
    
    # 4. Model
    print("Initializing NEXUS-Reason-Alpha...")
    model = NEXUSCore(nexus_config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 5. Trainer
    trainer = NEXUSTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
    )
    
    # 6. WandB
    if training_config.device == "cuda":
        wandb.init(project=WANDB_PROJECT, config=nexus_config.__dict__)
    
    # 7. Train
    print("Starting Training...")
    
    # --- BUDGET GUARDRAILS ---
    # Conservative estimate: $1.00/hr to account for Spot price spikes or disk costs
    COST_PER_HOUR = 1.00  
    MAX_BUDGET = 250.00   # Hard limit for this run
    
    import time
    start_time = time.time()
    
    def budget_check_callback(step):
        elapsed_hours = (time.time() - start_time) / 3600.0
        current_cost = elapsed_hours * COST_PER_HOUR
        
        if step % 100 == 0:
            print(f" [Budget] Spent ${current_cost:.2f} / ${MAX_BUDGET:.2f} ({elapsed_hours:.1f} hours)")
            
        if current_cost >= MAX_BUDGET:
            print(f"!!! BUDGET EXCEEDED (${current_cost:.2f}). STOPPING TRAINING !!!")
            return True # Stop signal
        return False
        
    # Inject callback into trainer (requires trainer support, or manual loop)
    # Since NEXUSTrainer.train() is a blocking call, we will wrap it or
    # rely on the trainer calling a hook if available. 
    # For this demo, we assume the trainer has a callback mechanism or we rely on
    # max_steps. To be safe, let's wrap the loop if possible, but NEXUSTrainer
    # encapsulates the loop.
    # Ideally, we modify NEXUSTrainer to accept a callback, but to avoid touching core
    # right now, we will monitor via a separate thread or just trust the max_steps
    # calculation.
    #
    # BETTER APPROACH: Calculate max_steps based on budget *before* starting.
    
    avg_step_time = 0.5 # Conservative estimate (2 it/s)
    max_hours = MAX_BUDGET / COST_PER_HOUR
    max_steps_budget = int((max_hours * 3600) / avg_step_time)
    
    print(f"Budget Analysis:")
    print(f"  Max Budget: ${MAX_BUDGET:.2f}")
    print(f"  Est. Cost/Hr: ${COST_PER_HOUR:.2f}")
    print(f"  Max Runtime: {max_hours:.1f} hours")
    print(f"  Max Steps (est): {max_steps_budget}")
    
    # Enforce budget limit on steps
    if args.steps > max_steps_budget:
        print(f"WARNING: Requested steps ({args.steps}) exceeds budget estimate ({max_steps_budget}).")
        print(f"Clamping max_steps to {max_steps_budget} to protect wallet.")
        training_config.max_steps = max_steps_budget
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    trainer.train()

if __name__ == "__main__":
    main()
