#!/bin/bash
# NEXUS Continuum - Mac Stop Script
# ==================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PID_FILE="$PROJECT_DIR/nexus.pid"

echo "========================================="
echo "Stopping NEXUS Continuum"
echo "========================================="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID file not found. NEXUS may not be running."
    echo ""
    echo "To check manually:"
    echo "  ps aux | grep 'nexus.service.server'"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "❌ Process not found (PID: $PID)"
    echo "Cleaning up stale PID file..."
    rm "$PID_FILE"
    exit 1
fi

echo "Stopping NEXUS (PID: $PID)..."

# Try graceful shutdown first
kill "$PID"

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ NEXUS stopped gracefully"
        rm "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "⚠️  Graceful shutdown timeout. Forcing..."
kill -9 "$PID" 2>/dev/null || true
rm "$PID_FILE"

echo "✅ NEXUS stopped (forced)"
echo ""
