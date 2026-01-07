# How to Start NEXUS - Quick Guide

## Prerequisites

```bash
# Check Python version (need 3.10+)
python3 --version

# Install dependencies
pip install -r requirements.txt
```

---

## Development Mode (Any OS)

**Quick start for testing:**

```bash
# From project root
python -m uvicorn nexus.service.server:app --reload
```

Then open: http://localhost:8000/dashboard

---

## Production - macOS (Your System)

### Option 1: Using Provided Script (Recommended)

```bash
# Start
./deployment/run_mac.sh

# Stop
./deployment/stop_mac.sh

# View logs
tail -f nexus.log
```

### Option 2: Manual Launch

```bash
# Start in background
nohup python -m uvicorn nexus.service.server:app \
  --host 0.0.0.0 --port 8000 > nexus.log 2>&1 &

# Save PID
echo $! > nexus.pid

# View logs
tail -f nexus.log

# Stop
kill $(cat nexus.pid)
```

### Option 3: Using Screen/Tmux

```bash
# Using screen
screen -S nexus
python -m uvicorn nexus.service.server:app --host 0.0.0.0 --port 8000
# Press Ctrl+A, then D to detach
# Reattach: screen -r nexus

# Using tmux
tmux new -s nexus
python -m uvicorn nexus.service.server:app --host 0.0.0.0 --port 8000
# Press Ctrl+B, then D to detach
# Reattach: tmux attach -t nexus
```

---

## Production - Linux

### Option 1: Systemd Service (Recommended)

```bash
# Install
sudo deployment/install.sh

# Start
sudo systemctl start nexus

# Enable on boot
sudo systemctl enable nexus

# View logs
sudo journalctl -u nexus -f

# Stop
sudo systemctl stop nexus

# Restart
sudo systemctl restart nexus

# Status
sudo systemctl status nexus
```

### Option 2: Manual Launch

```bash
# Start in background
nohup python -m uvicorn nexus.service.server:app \
  --host 0.0.0.0 --port 8000 > nexus.log 2>&1 &

# Stop
pkill -f "nexus.service.server"
```

---

## Production - Windows

### Option 1: Direct Launch

```powershell
# Start
python -m uvicorn nexus.service.server:app --host 0.0.0.0 --port 8000
```

### Option 2: As Windows Service

Install NSSM (Non-Sucking Service Manager):

```powershell
# Download from: https://nssm.cc/download

# Install service
nssm install NEXUS "C:\Python310\python.exe" "-m uvicorn nexus.service.server:app --host 0.0.0.0 --port 8000"
nssm set NEXUS AppDirectory "C:\path\to\nexus"

# Start service
nssm start NEXUS

# Stop service
nssm stop NEXUS
```

---

## Verify It's Running

```bash
# Check health
curl http://localhost:8000/api/status

# Expected response:
# {"daemon": {...}, "health": {"healthy": true}, ...}

# Open dashboard
open http://localhost:8000/dashboard  # Mac
xdg-open http://localhost:8000/dashboard  # Linux
start http://localhost:8000/dashboard  # Windows
```

---

## Monitoring

### View Status

```bash
curl http://localhost:8000/api/status | python -m json.tool
```

### Key Endpoints

- **Dashboard:** http://localhost:8000/dashboard
- **Status API:** http://localhost:8000/api/status
- **Health:** http://localhost:8000/api/status (check `.health.healthy`)
- **Metrics:** http://localhost:8000/api/metrics (Prometheus format)

### Check Logs

**macOS/Linux:**
```bash
tail -f nexus.log
```

**Systemd:**
```bash
sudo journalctl -u nexus -f
```

**Windows:**
Check console output or redirect to file

---

## Troubleshooting

### Port Already in Use

```bash
# Find what's using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
python -m uvicorn nexus.service.server:app --port 8001
```

### Dependencies Missing

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Check If Running

```bash
# Mac/Linux
ps aux | grep "nexus.service.server"

# Or check port
curl http://localhost:8000/api/status
```

### Can't Access Dashboard

1. Check server is running
2. Try: http://127.0.0.1:8000/dashboard
3. Check firewall settings
4. View logs for errors

---

## Next Steps

Once running:

1. **Monitor health:** `curl http://localhost:8000/api/status | jq '.health'`
2. **Submit requests:** Use the API or dashboard
3. **Check checkpoints:** Look in `nexus_checkpoints/` directory
4. **View thoughts:** Check status for recent model thoughts

**Full documentation:** See `docs/operations/runbook.md`

---

## Quick Reference

| Action | Command (macOS) |
|--------|----------------|
| Start | `./deployment/run_mac.sh` |
| Stop | `./deployment/stop_mac.sh` |
| Logs | `tail -f nexus.log` |
| Status | `curl http://localhost:8000/api/status` |
| Dashboard | `open http://localhost:8000/dashboard` |

| Action | Command (Linux systemd) |
|--------|------------------------|
| Start | `sudo systemctl start nexus` |
| Stop | `sudo systemctl stop nexus` |
| Logs | `sudo journalctl -u nexus -f` |
| Status | `curl http://localhost:8000/api/status` |
| Dashboard | `xdg-open http://localhost:8000/dashboard` |
