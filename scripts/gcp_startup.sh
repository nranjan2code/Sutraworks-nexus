#!/bin/bash
# GCP L4 Spot Instance Startup Script for NEXUS Training
# Usage: bash gcp_startup.sh

set -e  # Exit on error

echo ">>> Starting NEXUS Environment Setup..."

# 1. System Dependencies
echo ">>> Installing System Dependencies..."
sudo apt-get update && sudo apt-get install -y \
    git \
    python3-pip \
    python3-venv \
    htop \
    tmux \
    wget

# 2. Setup Python Environment
echo ">>> Setting up Python Environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 (Optimized for L4)
echo ">>> Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install Project Dependencies
echo ">>> Installing NEXUS Dependencies..."
# Assuming requirements.txt exists, otherwise install manually
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install transformers datasets wandb tqdm tiktoken numpy scipy sentencepiece
fi

# Install local package in editable mode
pip install -e .

# 5. Connect to WandB (Optional)
# echo ">>> Login to WandB..."
# wandb login

echo ">>> Environment Setup Complete!"
echo ""
echo "To start training:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Run training:  python scripts/train_alpha.py"
