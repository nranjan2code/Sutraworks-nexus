"""
NEXUS-Reason-Alpha (Edge Edition)
=================================

Configuration optimized for Raspberry Pi 5 (ARM Cortex-A76).

Design Philosophy: "Deep & Narrow"
- Narrower width (768) reduces matrix multiplication cost (CPU friendly).
- Deeper layers (32) maintain reasoning capacity.
- Llama-compatible tokenizer (32k) for ecosystem compatibility.
"""

from nexus.core.nexus_core import NEXUSConfig

# ~380M Parameters
# d_model=768, layers=32 -> ~226M backbone
# + Embeddings (32k * 768) = ~25M
# + Heads/Projections/SSM States = ~129M
ALPHA_CONFIG = NEXUSConfig(
    # Model Architecture (Edge Optimized)
    d_model=768,         # 1024 -> 768 (-44% FLOPs)
    d_latent=384,        # Half of d_model
    ssm_n_layers=32,     # Increased depth for reasoning
    n_heads=12,          # 768 / 12 = 64 dim per head
    ssm_d_state=64,
    ssm_d_conv=4,
    
    # Tokenizer (Llama Standard)
    vocab_size=32000,    # Llama 2/Mistral standard
    max_seq_len=4096,
    
    # World Model (JEPA)
    world_model_layers=4,
    
    # Optimization
    dropout=0.1,
    reasoning_depth=4,   # Deeper recursion allowed by narrow width
)

def get_config():
    return ALPHA_CONFIG
