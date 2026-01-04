"""
NEXUS Basic Usage Demo

This script demonstrates the basic usage of the NEXUS architecture,
showcasing its key capabilities:
1. O(n) linear-time sequence processing
2. Multi-modal input handling
3. Generation, reasoning, and imagination
4. Causal interventions

Run this script to see NEXUS in action!
"""

import torch
import numpy as np
from typing import Dict, Any

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def demo_nexus_initialization():
    """Demonstrate NEXUS model initialization."""
    print("\n" + "=" * 60)
    print("1. NEXUS Model Initialization")
    print("=" * 60)
    
    from nexus.core.nexus_core import NEXUSCore, NEXUSConfig
    
    # Create configuration with correct parameter names
    config = NEXUSConfig(
        vocab_size=32000,
        d_model=512,
        d_latent=256,
        ssm_n_layers=6,
        n_heads=8,
        ssm_d_state=64,
        max_seq_len=8192,
        dropout=0.1,
    )
    
    print(f"Configuration:")
    print(f"  - Model dim: {config.d_model}")
    print(f"  - SSM layers: {config.ssm_n_layers}")
    print(f"  - State dim (for O(n) processing): {config.ssm_d_state}")
    print(f"  - Max sequence length: {config.max_seq_len}")
    
    # Initialize model
    model = NEXUSCore(config)
    model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params:,} parameters")
    
    return model, config


def demo_sequence_processing(model: Any):
    """Demonstrate O(n) sequence processing."""
    print("\n" + "=" * 60)
    print("2. O(n) Linear-Time Sequence Processing")
    print("=" * 60)
    
    import time
    
    # Use smaller sequence lengths for CPU demo
    sequence_lengths = [128, 256, 512, 1024]
    times = []
    
    print("\nProcessing sequences of varying lengths:")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        for seq_len in sequence_lengths:
            # Create random input sequence
            x = torch.randint(0, 32000, (1, seq_len), device=device)
            
            # Warm-up
            _ = model(x)
            
            # Time the forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            outputs = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            print(f"  Seq len {seq_len:>5}: {elapsed*1000:>8.2f} ms")
    
    # Demonstrate linear scaling
    print("\nScaling analysis:")
    print("-" * 40)
    
    # Check if time roughly doubles when length doubles (linear)
    ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
    avg_ratio = np.mean(ratios)
    
    print(f"  Average time ratio when doubling sequence length: {avg_ratio:.2f}x")
    print(f"  Expected for O(n): ~2x | Expected for O(n²): ~4x")
    
    if avg_ratio < 3:
        print("  ✓ NEXUS demonstrates near-linear O(n) scaling!")
    
    return outputs


def demo_reasoning(model: Any):
    """Demonstrate neuro-symbolic reasoning."""
    print("\n" + "=" * 60)
    print("3. Neuro-Symbolic Reasoning")
    print("=" * 60)
    
    # Create a simple reasoning input (continuous embeddings)
    batch_size = 1
    seq_len = 256
    d_model = model.config.d_model
    
    # For reasoning, we need continuous vectors, not token IDs
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    model.eval()
    with torch.no_grad():
        # Get reasoning output with proof trace
        outputs = model.reason(x)
    
    print("\nReasoning output structure:")
    print("-" * 40)
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor{list(value.shape)}")
        elif isinstance(value, dict):
            print(f"  {key}: Dict with {len(value)} entries")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    print("\nKey features:")
    print("  - Symbolic grounding: Conclusions tied to knowledge base")
    print("  - Proof traces: Explainable reasoning steps")
    print("  - Soft unification: Fuzzy matching for flexible reasoning")
    
    return outputs


def demo_world_model(model: Any):
    """Demonstrate JEPA-style world modeling."""
    print("\n" + "=" * 60)
    print("4. JEPA-Style World Modeling (Imagination)")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 256
    d_model = model.config.d_model
    
    # Use continuous vectors for imagination
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    model.eval()
    with torch.no_grad():
        # Imagination: predict future abstract representations
        imagination_output = model.imagine(x, n_steps=5)
    
    print("\nImagination output structure:")
    print("-" * 40)
    if isinstance(imagination_output, torch.Tensor):
        print(f"  predictions: Tensor{list(imagination_output.shape)}")
    elif isinstance(imagination_output, dict):
        for key, value in imagination_output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor{list(value.shape)}")
            elif isinstance(value, list):
                print(f"  {key}: List of {len(value)} items")
            else:
                print(f"  {key}: {type(value).__name__}")
    
    print("\nKey features:")
    print("  - Abstract prediction: Works in representation space, not pixel/token space")
    print("  - Multi-step rollout: Can imagine sequences of future states")
    print("  - Hierarchical abstraction: Multiple timescales of prediction")
    
    return imagination_output


def demo_causal_inference(model: Any):
    """Demonstrate causal inference capabilities."""
    print("\n" + "=" * 60)
    print("5. Causal Inference & Intervention")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 256
    d_model = model.config.d_model
    
    # Use continuous vectors for causal inference
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    model.eval()
    with torch.no_grad():
        # Perform intervention
        intervention_output = model.intervene(
            x,
            intervention=(128, torch.randn(1, model.config.d_model, device=device))
        )
    
    print("\nIntervention output structure:")
    print("-" * 40)
    for key, value in intervention_output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor{list(value.shape)}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    print("\nKey features:")
    print("  - Causal discovery: Learn causal structure from data")
    print("  - Interventions: Simulate 'do' operations")
    print("  - Counterfactuals: 'What if X had been different?'")
    
    return intervention_output


def demo_generation(model: Any):
    """Demonstrate text generation."""
    print("\n" + "=" * 60)
    print("6. Text Generation")
    print("=" * 60)
    
    # Create a prompt
    prompt_len = 32
    prompt = torch.randint(0, 32000, (1, prompt_len), device=device)
    
    model.eval()
    with torch.no_grad():
        # Generate tokens
        generated = model.generate(
            prompt,
            max_new_tokens=64,
            temperature=0.8,
            top_k=50
        )
    
    print(f"\nGeneration results:")
    print("-" * 40)
    print(f"  Prompt length: {prompt_len} tokens")
    print(f"  Generated length: {generated.shape[1]} tokens")
    print(f"  New tokens: {generated.shape[1] - prompt_len}")
    
    print("\nKey features:")
    print("  - Efficient autoregressive generation using state-space backbone")
    print("  - Temperature and top-k sampling support")
    print("  - Reasoning-enhanced generation when enabled")
    
    return generated


def demo_efficiency_comparison():
    """Compare NEXUS efficiency against theoretical Transformer."""
    print("\n" + "=" * 60)
    print("7. Efficiency Comparison: NEXUS vs Transformer")
    print("=" * 60)
    
    # Theoretical comparison
    seq_lengths = [1024, 4096, 16384, 65536]
    
    print("\nTheoretical computational complexity:")
    print("-" * 60)
    print(f"{'Seq Length':>12} | {'Transformer O(n²)':>18} | {'NEXUS O(n)':>15} | {'Speedup':>10}")
    print("-" * 60)
    
    for n in seq_lengths:
        transformer_ops = n * n  # O(n²) for attention
        nexus_ops = n  # O(n) for state-space
        speedup = transformer_ops / nexus_ops
        
        print(f"{n:>12,} | {transformer_ops:>18,} | {nexus_ops:>15,} | {speedup:>9.0f}x")
    
    print("-" * 60)
    print("\nNEXUS achieves massive efficiency gains at long sequence lengths!")
    print("At 64K tokens: 64,000x theoretical speedup over quadratic attention")


def main():
    """Run all demos."""
    print("=" * 60)
    print("NEXUS - Neural EXploratory Unified Synthesis")
    print("Next-Generation AI Algorithm Demo")
    print("=" * 60)
    
    try:
        # Initialize model
        model, config = demo_nexus_initialization()
        
        # Run demos
        demo_sequence_processing(model)
        demo_reasoning(model)
        demo_world_model(model)
        demo_causal_inference(model)
        demo_generation(model)
        demo_efficiency_comparison()
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print("\nNEXUS Key Innovations:")
        print("  1. O(n) linear-time sequence processing (vs Transformer O(n²))")
        print("  2. JEPA-style abstract world modeling")
        print("  3. Neuro-symbolic reasoning with proof traces")
        print("  4. Energy-based adaptive computation")
        print("  5. Built-in causal inference and intervention")
        print("\nReady for the next generation of AI!")
        
    except ImportError as e:
        print(f"\nNote: Could not import NEXUS modules: {e}")
        print("Make sure to install the package first: pip install -e .")
        print("\nRunning theoretical comparison only...")
        demo_efficiency_comparison()


if __name__ == "__main__":
    main()
