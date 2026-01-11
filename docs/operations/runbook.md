# NEXUS Continuum - Operations Runbook

**Version:** 1.1  
**Last Updated:** 2026-01-10  
**Maintainer:** NEXUS Team

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deployment](#deployment)
3. [Security](#security)
4. [Monitoring](#monitoring)
5. [Troubleshooting](#troubleshooting)
6. [Maintenance](#maintenance)
7. [Emergency Procedures](#emergency-procedures)

---

## Quick Start

### Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
python -m uvicorn nexus.service.server:app --reload

# 3. Access dashboard
open http://localhost:8000/dashboard
```

### Production (Docker)

```bash
# 1. Build and start
docker-compose up -d

# 2. View logs
docker-compose logs -f nexus

# 3. Check status
curl http://localhost:8000/api/status
```

### Production (Systemd)

```bash
# 1. Install (one-time)
sudo deployment/install.sh

# 2. Start service
sudo systemctl start nexus

# 3. Check status
sudo systemctl status nexus

# 4. View logs
sudo journalctl -u nexus -f
```

---

## Deployment

### Docker Deployment

**1. Build Image**

```bash
docker build -t nexus-continuum:latest .
```

**2. Run Container**

```bash
docker run -d \
  --name nexus \
  -p 8000:8000 \
  -v nexus-checkpoints:/app/nexus_checkpoints \
  -v nexus-logs:/app/logs \
  --restart unless-stopped \
  nexus-continuum:latest
```

**3. With Docker Compose** (Recommended)

```bash
# Start all services
docker-compose up -d

# View all logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Systemd Deployment

**1. Installation**

```bash
# Run installer
sudo deployment/install.sh

# Verify installation
sudo systemctl status nexus
```

**2. Service Management**

```bash
# Start
sudo systemctl start nexus

# Stop
sudo systemctl stop nexus

# Restart
sudo systemctl restart nexus

# Enable on boot
sudo systemctl enable nexus

# Disable
sudo systemctl disable nexus

# View status
sudo systemctl status nexus
```

**3. Configuration**

Edit `/etc/systemd/system/nexus.service`:

```ini
[Service]
Environment="NEXUS_MODEL_SIZE=base"  # small, base, large
Environment="NEXUS_ARCHITECTURE=flowing"  # flowing, layered
Environment="LOG_LEVEL=INFO"
```

Then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart nexus
```

---

## Security

### API Key Authentication

Enable API key authentication by setting the `NEXUS_API_KEY` environment variable:

```bash
# Generate a secure API key
python -c "from nexus.service.auth import create_api_key; print(create_api_key())"

# Set the API key
export NEXUS_API_KEY="nexus_your-generated-key-here"
```

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_API_KEY` | None (public) | API key for authentication |
| `NEXUS_API_KEY_HEADER` | X-API-Key | Header name for API key |
| `NEXUS_TRUST_PROXY` | false | Trust `X-Forwarded-For` header |

**Making Authenticated Requests:**

```bash
# With API key (Required if NEXUS_API_KEY is set)
curl -H "X-API-Key: your-key" http://localhost:8000/api/interact \
  -d '{"prompt": "Hello"}' -H "Content-Type: application/json"

# Check status (Required if NEXUS_API_KEY is set)
curl -H "X-API-Key: your-key" http://localhost:8000/api/status
```

### Rate Limiting

Rate limiting is enabled by default when `slowapi` is installed.

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_RATE_LIMIT_ENABLED` | true | Enable/disable rate limiting |
| `NEXUS_RATE_LIMIT_RPM` | 60 | Requests per minute |
| `NEXUS_RATE_LIMIT_BURST` | 10 | Burst allowance |

**Rate Limit Response:**

```json
{
  "detail": "Rate limit exceeded: 60 per 1 minute"
}
```

### GPU Governance

Configure GPU resource limits:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_GPU_MEMORY_LIMIT` | 50 | Max GPU memory usage (%) |
| `NEXUS_GPU_UTILIZATION_LIMIT` | 80 | Max GPU utilization (%) |
| `NEXUS_GPU_THERMAL_WARNING` | 75 | GPU thermal warning (°C) |
| `NEXUS_GPU_THERMAL_CRITICAL` | 85 | GPU thermal critical (°C) |

---

## Monitoring

### Health Check

```bash
# Quick health check
curl http://localhost:8000/api/status | jq '.health'

# Full status
curl http://localhost:8000/api/status | jq '.'
```

Expected healthy response:

```json
{
  "health": {
    "healthy": true,
    "status": "healthy",
    "issues": []
  }
}
```

### Key Metrics

**1. Request Metrics**

```bash
curl http://localhost:8000/api/status | jq '.metrics.requests'
```

Monitor:
- `total`: Total requests processed
- `success_rate`: Should be > 0.9 (90%)
- `refusal_rate`: Expected 0.1-0.3 (10-30%)
- `rate_per_second`: Current throughput

**2. Latency Metrics**

```bash
curl http://localhost:8000/api/status | jq '.metrics.latency.request'
```

Monitor:
- `p50_ms`: Median latency (target: < 500ms)
- `p95_ms`: 95th percentile (target: < 2000ms)
- `p99_ms`: 99th percentile (target: < 5000ms)

**3. Memory Metrics**

```bash
curl http://localhost:8000/api/status | jq '.memory'
```

Monitor:
- `process.current_mb`: Current usage
- `process.peak_mb`: Peak usage
- `process.growth_rate_mb_per_hour`: Should be < 10

**4. Flow Metrics** (FlowingNEXUS)

```bash
curl http://localhost:8000/api/status | jq '.metrics.flow'
```

Monitor:
- `average_depth`: Typical: 10-20 iterations
- `convergence_rate`: Should be > 0.8 (80%)

### Logs

**Docker:**

```bash
# All logs
docker-compose logs -f nexus

# Last 100 lines
docker-compose logs --tail=100 nexus

# Grep for errors
docker-compose logs nexus | grep ERROR
```

**Systemd:**

```bash
# Follow logs
sudo journalctl -u nexus -f

# Last hour
sudo journalctl -u nexus --since "1 hour ago"

# Errors only
sudo journalctl -u nexus -p err

# Export to file
sudo journalctl -u nexus > nexus-logs.txt
```

### Prometheus Metrics

Metrics endpoint:

```bash
curl http://localhost:8000/api/metrics
```

Add to Prometheus (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: 'nexus'
    static_configs:
      - targets: ['nexus:8000']
    metrics_path: '/api/metrics'
    scrape_interval: 15s
```

### Grafana Dashboards

1. Import Prometheus datasource
2. Create dashboard with panels:
   - Request rate & latency
   - Memory & CPU usage
   - Flow depth distribution
   - Confidence scores
   - Error rate

---

## Troubleshooting

### High CPU Usage

**Symptoms:**
- CPU usage > 80%
- Slow response times
- Resource governor warnings

**Diagnosis:**

```bash
# Check current CPU
curl http://localhost:8000/api/status | jq '.resources.cpu_percent'

# Check resource mode
curl http://localhost:8000/api/status | jq '.resources.mode'
```

**Solutions:**

1. **Pause background learning:**

```bash
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "pause"}'
```

2. **Reduce flow depth (FlowingNEXUS):**

Edit config, reduce `max_flow_steps` from 50 to 30

3. **Scale down model:**

Restart with smaller model:

```bash
# Docker
docker-compose down
# Edit docker-compose.yml: NEXUS_MODEL_SIZE=small
docker-compose up -d

# Systemd
# Edit /etc/systemd/system/nexus.service
# Add: Environment="NEXUS_MODEL_SIZE=small"
sudo systemctl daemon-reload
sudo systemctl restart nexus
```

### High Memory Usage

**Symptoms:**
- Memory > 3GB
- Memory warnings in logs
- System swap usage

**Diagnosis:**

```bash
# Check memory
curl http://localhost:8000/api/status | jq '.memory.process'

# Check for leaks
curl http://localhost:8000/api/status | jq '.memory.process.growth_rate_mb_per_hour'
```

**Solutions:**

1. **Trigger aggressive cleanup:**

Memory manager runs automatically, but you can restart to force cleanup:

```bash
sudo systemctl restart nexus
# or
docker-compose restart nexus
```

2. **Reduce replay buffer:**

Edit `nexus/service/memory_manager.py`:

```python
MemoryConfig(
    max_replay_buffer_size=1024,  # Reduced from 2048
)
```

3. **Check for memory leak:**

If `growth_rate_mb_per_hour > 10`:

```bash
# Restart immediately
sudo systemctl restart nexus

# Report issue with logs
sudo journalctl -u nexus --since "1 hour ago" > leak-report.txt
```

### Circuit Breaker Open

**Symptoms:**
- Requests failing with "Circuit breaker is OPEN"
- High error rate

**Diagnosis:**

```bash
curl http://localhost:8000/api/status | jq '.circuit_breakers'
```

**Solutions:**

1. **Wait for auto-recovery** (60 seconds default)

2. **Check underlying issue:**

```bash
# View recent errors
sudo journalctl -u nexus -p err --since "5 minutes ago"
```

3. **Manual reset:**

Restart service:

```bash
sudo systemctl restart nexus
```

### Low Convergence Rate

**Symptoms:**
- `convergence_rate < 0.8`
- High flow depths
- Slow responses

**Diagnosis:**

```bash
curl http://localhost:8000/api/status | jq '.metrics.flow'
```

**Solutions:**

1. **Increase max iterations:**

Edit `FlowingConfig`:

```python
max_flow_steps=100  # Increased from 50
```

2. **Adjust convergence threshold:**

```python
convergence_threshold=2e-4  # Less strict (was 1e-4)
```

3. **Check model health:**

May indicate model degradation. Restore from earlier checkpoint:

```bash
# List checkpoints
ls -lh /app/nexus_checkpoints/

# Restore specific checkpoint
# Stop service, replace checkpoint, restart
```

### High Refusal Rate

**Symptoms:**
- `refusal_rate > 0.5` (50%)
- Most queries refused

**Diagnosis:**

```bash
curl http://localhost:8000/api/status | jq '.model.confidence_threshold'
```

**Solutions:**

1. **Normal behavior:** NEXUS refuses when uncertain - this is a feature!

2. **If too conservative:**

The threshold adapts with experience. For fresh models, this is expected.

3. **Accelerate learning:**

Enable teacher mode:

```bash
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_start", "topic": "general_knowledge"}'
```

### Checkpoint Load Failure

**Symptoms:**
- Service starts but shows "No checkpoint found"
- Or: "Failed to load checkpoint"

**Diagnosis:**

```bash
# Check checkpoint directory
ls -lh /app/nexus_checkpoints/

# Check permissions
ls -la /app/nexus_checkpoints/

# Check disk space
df -h /app/nexus_checkpoints/
```

**Solutions:**

1. **Missing checkpoints:**

Normal for first start. Service will create new ones.

2. **Corrupted checkpoint:**

```bash
# Move corrupted checkpoint
mv /app/nexus_checkpoints/checkpoint_XXXXXX.pt \
   /app/nexus_checkpoints/corrupted/

# Restart - will load previous checkpoint
sudo systemctl restart nexus
```

3. **Permission issues:**

```bash
sudo chown -R nexus:nexus /app/nexus_checkpoints/
sudo systemctl restart nexus
```

---

## Maintenance

### Regular Maintenance Schedule

**Daily:**
- Check health status
- Review error logs
- Monitor resource usage

**Weekly:**
- Review metrics trends
- Check checkpoint disk usage
- Rotate old logs

**Monthly:**
- Full system health review
- Performance optimization
- Update dependencies

### Checkpoint Management

**List Checkpoints:**

```bash
curl http://localhost:8000/api/status | jq '.checkpoints'
```

**Manual Backup:**

```bash
# Create backup
tar -czf nexus-backup-$(date +%Y%m%d).tar.gz \
  /app/nexus_checkpoints/

# Store offsite
scp nexus-backup-*.tar.gz user@backup-server:/backups/
```

**Restore from Backup:**

```bash
# Stop service
sudo systemctl stop nexus

# Extract backup
tar -xzf nexus-backup-YYYYMMDD.tar.gz -C /

# Restart service
sudo systemctl start nexus
```

### Log Rotation

**Docker:**

Configure in `docker-compose.yml`:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

**Systemd:**

Configured automatically via journald. Adjust `/etc/systemd/journald.conf`:

```ini
[Journal]
SystemMaxUse=1G
SystemMaxFileSize=100M
MaxRetentionSec=30day
```

### Updates

**Update Dependencies:**

```bash
# Backup first!
sudo systemctl stop nexus

# Update packages
cd /opt/nexus
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart
sudo systemctl start nexus
```

**Update Code:**

```bash
# Backup checkpoints
tar -czf backup.tar.gz /opt/nexus/nexus_checkpoints/

# Pull latest code
cd /opt/nexus
git pull

# Reinstall
pip install -r requirements.txt

# Restart
sudo systemctl restart nexus
```

---

## Emergency Procedures

### System Unresponsive

**1. Check if running:**

```bash
sudo systemctl status nexus
# or
docker-compose ps
```

**2. Check resource exhaustion:**

```bash
top -p $(pgrep -f nexus)
```

**3. Force restart:**

```bash
sudo systemctl restart nexus
# or
docker-compose restart nexus
```

**4. If still unresponsive, hard kill:**

```bash
sudo systemctl stop nexus
sudo killall -9 python
sudo systemctl start nexus
```

### Rapid Memory Growth

**Immediate action:**

```bash
# Restart immediately
sudo systemctl restart nexus

# Monitor closely
watch -n 5 'curl -s http://localhost:8000/api/status | jq ".memory.process.current_mb"'
```

**If continues:**

```bash
# Stop service
sudo systemctl stop nexus

# Investigate
journalctl -u nexus --since "1 hour ago" > investigation.log

# Start in safe mode (pause learning)
# Edit config to pause=true, then start
```

### Data Corruption

**Symptoms:**
- Checkpoint load failures
- Unexpected crashes
- Inconsistent behavior

**Recovery:**

```bash
# Stop service
sudo systemctl stop nexus

# Restore from known-good checkpoint
cd /opt/nexus/nexus_checkpoints
# Identify good checkpoint (by date/metadata)
# Remove corrupted checkpoints

# Restart
sudo systemctl start nexus
```

### Complete Failure - Fresh Start

**Last resort only - loses all learned knowledge:**

```bash
# Stop service
sudo systemctl stop nexus

# Backup (just in case)
tar -czf emergency-backup.tar.gz /opt/nexus/

# Clear checkpoints
rm -rf /opt/nexus/nexus_checkpoints/*

# Restart fresh
sudo systemctl start nexus
```

---

## Support & Resources

### Documentation
- Architecture: `docs/architecture/`
- API Reference: `docs/api/`
- Development: `docs/development/`

### Monitoring Dashboards
- Status: http://localhost:8000/dashboard
- Metrics: http://localhost:8000/api/status
- Prometheus: http://localhost:9090 (if configured)
- Grafana: http://localhost:3000 (if configured)

### Logs Locations
- **Docker:** `docker-compose logs nexus`
- **Systemd:** `journalctl -u nexus`
- **File logs:** `/opt/nexus/logs/` (if configured)

### Getting Help
- GitHub Issues: https://github.com/yourusername/nexus/issues
- Documentation: `docs/`
- Email: support@nexus.ai (if available)

---

**Remember:** NEXUS is designed to evolve continuously. Monitor regularly, backup frequently, and let it grow!
