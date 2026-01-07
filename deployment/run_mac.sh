#!/bin/bash
# NEXUS Continuum - Mac Launch Script
# ====================================
#
# Run NEXUS as a background process on macOS
# (since macOS doesn't use systemd)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_DIR="$PROJECT_DIR/venv"
LOG_FILE="$PROJECT_DIR/nexus.log"
PID_FILE="$PROJECT_DIR/nexus.pid"

echo "========================================="
echo "NEXUS Continuum - Mac Launch Script"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install/update dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT_DIR/requirements.txt"

# Create necessary directories
mkdir -p "$PROJECT_DIR/nexus_checkpoints"
mkdir -p "$PROJECT_DIR/logs"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "NEXUS is already running (PID: $OLD_PID)"
        echo ""
        echo "To stop: kill $OLD_PID"
        echo "To restart: kill $OLD_PID && $0"
        exit 1
    else
        # PID file exists but process doesn't
        rm "$PID_FILE"
    fi
fi

# Start NEXUS
echo "Starting NEXUS Continuum..."
cd "$PROJECT_DIR"

nohup python -m uvicorn nexus.service.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

echo ""
echo "========================================="
echo "NEXUS Started Successfully!"
echo "========================================="
echo ""
echo "PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Access points:"
echo "  - Dashboard: http://localhost:8000/dashboard"
echo "  - API: http://localhost:8000/api/status"
echo ""
echo "Commands:"
echo "  - View logs: tail -f $LOG_FILE"
echo "  - Stop: kill $PID"
echo "  - Restart: kill $PID && $0"
echo ""
echo "Waiting for startup..."
sleep 3

# Check if process is still running
if ps -p $PID > /dev/null; then
    echo "✅ NEXUS is running"

    # Test health endpoint
    if command -v curl >/dev/null 2>&1; then
        echo "Testing health endpoint..."
        if curl -s http://localhost:8000/api/status > /dev/null; then
            echo "✅ Health check passed"
        else
            echo "⚠️  Health check failed (server may still be starting)"
        fi
    fi
else
    echo "❌ NEXUS failed to start. Check log file:"
    echo "   tail -n 50 $LOG_FILE"
    rm "$PID_FILE"
    exit 1
fi

echo ""
echo "View real-time logs:"
echo "  tail -f $LOG_FILE"
echo ""
