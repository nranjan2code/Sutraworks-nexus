# NEXUS Control Guide - Complete Reference

**Your complete guide to controlling NEXUS Continuum**

---

## 🎮 Control Methods

You can control NEXUS in **3 ways**:

1. **CLI Tool** (`nexusctl`) - Command line control ⭐ **Recommended**
2. **Dashboard** - Web-based GUI (http://localhost:8000/dashboard)
3. **Direct API** - HTTP REST API

---

## 🚀 Quick Start with CLI

```bash
# Make sure you're in the project directory
cd /Users/sutraworkslab/Projects/sutraworks-genNxt

# Start NEXUS
./nexusctl start

# Check status
./nexusctl status

# Open dashboard
./nexusctl dashboard

# Stop NEXUS
./nexusctl stop
```

---

## 📋 CLI Commands Reference

### Basic Operations

```bash
# START - Launch NEXUS
./nexusctl start

# STOP - Graceful shutdown (saves checkpoint)
./nexusctl stop

# RESTART - Stop and start
./nexusctl restart

# STATUS - Show detailed status
./nexusctl status

# HEALTH - Quick health check
./nexusctl health
```

### Learning Control

```bash
# PAUSE - Pause background learning (keeps serving requests)
./nexusctl pause

# RESUME - Resume background learning
./nexusctl resume
```

### Monitoring

```bash
# LOGS - View last 50 log lines
./nexusctl logs

# LOGS (FOLLOW) - Live tail of logs
./nexusctl logs -f

# SAVE - Show checkpoint information
./nexusctl save

# DASHBOARD - Open web dashboard
./nexusctl dashboard
```

---

## 🎨 Dashboard Controls

### Access Dashboard

**Method 1: Using CLI**
```bash
./nexusctl dashboard
```

**Method 2: Direct Browser**
```
http://localhost:8000/dashboard
```

### Dashboard Features

**Left Sidebar - System Status:**
- **Status Indicator**: Running/Paused state
- **Resource Mode**: Active (10% CPU) / Idle (25% CPU)
- **CPU Usage**: Real-time with visual bar
- **Violations**: Resource limit violations count

**Control Buttons:**
- **Pause** - Pause background learning
- **Resume** - Resume background learning
- **Bootstrap Training** - Enable teacher-student learning
  - Optional: Enter topic to focus on
- **Stop Training** - Disable teacher mode

**Evolution Stats:**
- **Age**: Time since NEXUS started
- **Experience**: Continuous growth factor (0-1)
- **Flow Depth**: Average iterations to equilibrium
- **Interactions**: Total processed requests

**Main Area:**
- **Interaction Panel**: Chat with NEXUS in real-time
- **Thought Stream**: See NEXUS's internal processing log

---

## 🔧 API Control (Advanced)

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### 1. Get Status
```bash
curl http://localhost:8000/api/status

# With formatting
curl http://localhost:8000/api/status | python -m json.tool
```

**Response includes:**
- Daemon status (running/paused)
- Health check results
- Request metrics (total, success rate, latency)
- Memory usage (current, peak, growth rate)
- Model stats (interactions, flow depth)
- Circuit breaker states
- Checkpoint information

#### 2. Control Commands

**Pause Learning:**
```bash
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "pause"}'
```

**Resume Learning:**
```bash
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "resume"}'
```

**Start Training Mode:**
```bash
# General training
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_start"}'

# With specific topic
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_start", "topic": "mathematics"}'
```

**Stop Training Mode:**
```bash
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_stop"}'
```

#### 3. Interact with NEXUS

```bash
curl -X POST http://localhost:8000/api/interact \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?"}'
```

**Response:**
```json
{
  "response": "Python is a high-level programming language..."
}
```

---

## 🎯 Common Workflows

### Daily Operations

**Morning - Start NEXUS:**
```bash
./nexusctl start
./nexusctl status  # Verify it's healthy
```

**During Day - Monitor:**
```bash
./nexusctl status  # Check metrics
./nexusctl health  # Quick health check
./nexusctl logs    # Review recent activity
```

**Evening - Stop (or leave running):**
```bash
# Option 1: Stop
./nexusctl stop

# Option 2: Leave running (recommended!)
# NEXUS is designed to run continuously
```

### Development Workflow

**Testing Changes:**
```bash
# Stop NEXUS
./nexusctl stop

# Make code changes
# ... edit files ...

# Restart
./nexusctl restart

# Monitor logs for errors
./nexusctl logs -f
```

### Performance Tuning

**High CPU Usage:**
```bash
# Pause background learning
./nexusctl pause

# Check status
./nexusctl status  # CPU should drop

# Resume when ready
./nexusctl resume
```

**Memory Issues:**
```bash
# Check memory
./nexusctl status  # Look at memory section

# If high growth rate (>10 MB/hour):
./nexusctl restart  # Triggers cleanup
```

### Teacher-Student Learning

**Enable Learning from Ollama:**
```bash
# From CLI - manual API call
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_start", "topic": "science"}'

# Or use dashboard:
# 1. Open dashboard: ./nexusctl dashboard
# 2. Enter topic in text field (optional)
# 3. Click "Bootstrap Training"
```

**Disable Teacher Mode:**
```bash
# From CLI
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_stop"}'

# Or click "Stop Training" in dashboard
```

---

## 📊 Understanding Status Output

```bash
./nexusctl status
```

### Daemon Section
```
Daemon:
  Running: True          # Is NEXUS active?
  Paused: False          # Is learning paused?
  Uptime: 3600s          # How long running
```

### Health Section
```
Health:
✅ Status: healthy       # Overall health
```

**Possible Statuses:**
- `healthy` - All good
- `degraded` - Some issues but operational
- `unhealthy` - Critical problems

**Common Issues:**
- High error rate
- High refusal rate
- High latency
- Low convergence rate

### Metrics Section
```
Metrics:
  Total requests: 150    # Total processed
  Responses: 120         # Successfully answered
  Refusals: 30           # Politely declined
  Success rate: 80.0%    # Response rate
```

**What's Normal:**
- Success rate: 70-90% (refusal is a feature!)
- Response time P95: < 2000ms
- Error rate: < 1%

### Memory Section
```
Memory:
  Current: 856.3 MB      # Current usage
  Peak: 892.1 MB         # Highest usage
  Growth: 2.4 MB/hour    # Leak indicator
```

**Watch For:**
- Growth > 10 MB/hour = possible leak
- Current > 4000 MB = very high usage

### Model Section
```
Model:
  Architecture: flowing  # Layer-free or layered
  Interactions: 150      # Total learning interactions
  Avg flow depth: 12.3   # Iterations to equilibrium
```

**Typical Values:**
- Flow depth: 10-20 (varies with input complexity)
- More depth = harder inputs

---

## 🔍 Monitoring & Debugging

### Real-Time Monitoring

**Watch Status:**
```bash
# Linux/Mac
watch -n 2 './nexusctl status'

# Or manually poll
while true; do
  clear
  ./nexusctl status
  sleep 2
done
```

**Follow Logs:**
```bash
./nexusctl logs -f
```

**Watch Metrics:**
```bash
# Get just metrics
curl -s http://localhost:8000/api/status | jq '.metrics'

# Watch specific metric
watch -n 1 'curl -s http://localhost:8000/api/status | jq ".metrics.requests.total"'
```

### Troubleshooting

**NEXUS Won't Start:**
```bash
# Check if already running
./nexusctl status

# Check logs for errors
./nexusctl logs | tail -50

# Check port availability
lsof -i :8000  # macOS/Linux

# Try development mode (shows errors)
python -m uvicorn nexus.service.server:app
```

**High CPU Usage:**
```bash
# Check current mode
./nexusctl status  # Look at "Mode"

# Pause learning
./nexusctl pause

# If still high, check logs
./nexusctl logs -f
```

**Can't Connect to Dashboard:**
```bash
# Verify NEXUS is running
./nexusctl status

# Check if port is accessible
curl http://localhost:8000/api/status

# Try different browser
# Or: http://127.0.0.1:8000/dashboard
```

---

## 💾 Checkpoints & Recovery

### Understanding Checkpoints

**Automatic Saves:**
- Every **5 minutes** during operation
- On **graceful shutdown** (`./nexusctl stop`)

**Location:**
```
/Users/sutraworkslab/Projects/sutraworks-genNxt/nexus_checkpoints/
```

**View Checkpoints:**
```bash
./nexusctl save  # Shows recent checkpoints

# Or directly
ls -lh nexus_checkpoints/
```

### Recovery from Checkpoint

**Automatic:**
```bash
# NEXUS auto-loads latest checkpoint on start
./nexusctl start
# Will resume from last checkpoint
```

**Manual Selection:**
Not currently supported via CLI. Edit daemon code to specify checkpoint path.

### Backup Checkpoints

```bash
# Backup current checkpoints
tar -czf nexus-backup-$(date +%Y%m%d).tar.gz nexus_checkpoints/

# Restore from backup
tar -xzf nexus-backup-YYYYMMDD.tar.gz
./nexusctl restart
```

---

## 🎓 Best Practices

### Production Operation

1. **Let It Run**
   - NEXUS is designed for continuous operation
   - Checkpoints protect against crashes
   - Memory management prevents leaks

2. **Monitor Regularly**
   ```bash
   # Daily health check
   ./nexusctl health

   # Weekly review
   ./nexusctl status
   ```

3. **Backup Weekly**
   ```bash
   tar -czf nexus-backup-$(date +%Y%m%d).tar.gz nexus_checkpoints/
   ```

### Development Best Practices

1. **Use Development Mode for Testing**
   ```bash
   python -m uvicorn nexus.service.server:app --reload
   ```

2. **Check Status After Changes**
   ```bash
   ./nexusctl restart
   ./nexusctl health
   ./nexusctl logs -f
   ```

3. **Test API Changes**
   ```bash
   curl http://localhost:8000/api/status
   ```

---

## 🆘 Emergency Procedures

### NEXUS Unresponsive

```bash
# 1. Check if running
./nexusctl status

# 2. Check process
ps aux | grep nexus

# 3. Force stop
kill $(cat nexus.pid)

# 4. Clean start
rm nexus.pid
./nexusctl start
```

### Memory Crisis

```bash
# Immediate restart (triggers cleanup)
./nexusctl restart

# Monitor memory
./nexusctl status | grep Memory

# If continues: reduce load
./nexusctl pause
```

### Lost Control

```bash
# Find process
ps aux | grep "nexus.service.server"

# Kill manually
kill -9 <PID>

# Clean state
rm nexus.pid

# Fresh start
./nexusctl start
```

---

## 📚 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│           NEXUS Control Quick Reference         │
├─────────────────────────────────────────────────┤
│ START       ./nexusctl start                    │
│ STOP        ./nexusctl stop                     │
│ RESTART     ./nexusctl restart                  │
│ STATUS      ./nexusctl status                   │
│ HEALTH      ./nexusctl health                   │
│ PAUSE       ./nexusctl pause                    │
│ RESUME      ./nexusctl resume                   │
│ LOGS        ./nexusctl logs [-f]                │
│ DASHBOARD   ./nexusctl dashboard                │
│ SAVE        ./nexusctl save                     │
├─────────────────────────────────────────────────┤
│ Dashboard   http://localhost:8000/dashboard     │
│ API         http://localhost:8000/api/status    │
│ Checkpoints nexus_checkpoints/                  │
│ Logs        nexus.log                           │
└─────────────────────────────────────────────────┘
```

---

**For detailed operations guide, see:** `docs/operations/runbook.md`

**For production deployment, see:** `PRODUCTION_READY.md`

**For getting started, see:** `START.md`
