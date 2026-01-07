#!/bin/bash
# NEXUS Continuum - Installation Script
# ======================================
#
# Installs NEXUS as a systemd service on Linux

set -e

echo "========================================="
echo "NEXUS Continuum - Installation Script"
echo "========================================="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Configuration
INSTALL_DIR="/opt/nexus"
SERVICE_USER="nexus"
PYTHON_VERSION="3.10"

echo "Installation directory: $INSTALL_DIR"
echo "Service user: $SERVICE_USER"
echo ""

# Create user if doesn't exist
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating user: $SERVICE_USER"
    useradd -r -s /bin/bash -d $INSTALL_DIR -m $SERVICE_USER
else
    echo "User already exists: $SERVICE_USER"
fi

# Create installation directory
echo "Creating installation directory..."
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    curl \
    build-essential

# Create virtual environment
echo "Creating Python virtual environment..."
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate

# Copy application files
echo "Copying application files..."
# This assumes script is run from repository root
cp -r nexus/ $INSTALL_DIR/
cp -r examples/ $INSTALL_DIR/
cp requirements.txt $INSTALL_DIR/

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "Creating data directories..."
mkdir -p nexus_checkpoints
mkdir -p logs

# Set permissions
echo "Setting permissions..."
chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR

# Install systemd service
echo "Installing systemd service..."
cp deployment/nexus.service /etc/systemd/system/
systemctl daemon-reload

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Review configuration (optional)"
echo "  2. Start service:   sudo systemctl start nexus"
echo "  3. Enable on boot:  sudo systemctl enable nexus"
echo "  4. Check status:    sudo systemctl status nexus"
echo "  5. View logs:       sudo journalctl -u nexus -f"
echo "  6. Access dashboard: http://localhost:8000/dashboard"
echo ""
echo "To uninstall:"
echo "  sudo systemctl stop nexus"
echo "  sudo systemctl disable nexus"
echo "  sudo rm /etc/systemd/system/nexus.service"
echo "  sudo rm -rf /opt/nexus"
echo ""
