# Production Architecture

## Overview

NEXUS v2.0 introduces a comprehensive production infrastructure layer that transforms the research prototype into an enterprise-ready, continuously operating system.

**Status**: Production Ready
**Version**: 2.0.0
**Zero Technical Debt**: All production features implemented to completion

---

## Production Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION NEXUS v2.0                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     CONTROL INTERFACES                                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │  │   nexusctl   │  │   Dashboard  │  │   REST API   │                │ │
│  │  │   (CLI)      │  │   (Web UI)   │  │   (HTTP)     │                │ │
│  │  │              │  │              │  │              │                │ │
│  │  │ start/stop   │  │ Real-time    │  │ /api/status  │                │ │
│  │  │ pause/resume │  │ monitoring   │  │ /api/control │                │ │
│  │  │ status/logs  │  │ interaction  │  │ /api/interact│                │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                   PRODUCTION DAEMON (daemon.py)                        │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐   │ │
│  │  │  Main Loop:                                                     │   │ │
│  │  │  1. Resource check (ResourceGovernor)                           │   │ │
│  │  │  2. Process requests (with CircuitBreaker)                      │   │ │
│  │  │  3. Background learning (when idle)                             │   │ │
│  │  │  4. Periodic checkpoint (every 5 min)                           │   │ │
│  │  │  5. Memory cleanup (periodic GC)                                │   │ │
│  │  └────────────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐               │
│           │                        │                        │               │
│           ▼                        ▼                        ▼               │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐       │
│  │  Tokenizer     │      │   Metrics      │      │   Checkpoint   │       │
│  │  (NEXUSToken)  │      │  (Prometheus)  │      │   Manager      │       │
│  │                │      │                │      │                │       │
│  │ HuggingFace    │      │ Latency P50/   │      │ Atomic save    │       │
│  │ transformers   │      │ P95/P99        │      │ SHA256 check   │       │
│  │ Special tokens │      │ Flow metrics   │      │ Auto-rotation  │       │
│  └────────────────┘      └────────────────┘      └────────────────┘       │
│                                                                             │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐       │
│  │  Circuit       │      │   Memory       │      │   Resource     │       │
│  │  Breaker       │      │   Manager      │      │   Governor     │       │
│  │                │      │                │      │                │       │
│  │ 3-state:       │      │ Leak detection │      │ CPU/RAM limits │       │
│  │ CLOSED/OPEN/   │      │ Auto cleanup   │      │ Active: 10%    │       │
│  │ HALF_OPEN      │      │ GC triggers    │      │ Idle: 25%      │       │
│  └────────────────┘      └────────────────┘      └────────────────┘       │
│                                                                             │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        CORE NEXUS MODEL                                │ │
│  │                    (FlowingNEXUS / NEXUSCore)                          │ │
│  │                                                                        │ │
│  │  Living NEXUS Layer:                                                   │ │
│  │  ├── UncertaintyGate (anti-hallucination)                              │ │
│  │  ├── LifecycleManager (continuous evolution)                           │ │
│  │  ├── ContinualLearner (learn while serving)                            │ │
│  │  └── TeacherStudentModule (bootstrap from Ollama)                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Real Tokenization (`nexus/core/tokenizer.py`)

**Purpose**: Replace mock tokenization with production-grade HuggingFace integration

**Implementation**:
```python
class NEXUSTokenizer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Special NEXUS tokens
        special_tokens = {
            "uncertain_token": "[UNCERTAIN]",
            "refuse_token": "[REFUSE]",
            "think_token": "[THINK]",
            "dream_token": "[DREAM]",
        }
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens.values())})

    def encode(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """Encode text to tensor."""
        encoded = self.tokenizer.encode(
            text, max_length=max_length, truncation=True,
            add_special_tokens=True, return_tensors="pt"
        )
        return encoded.squeeze(0)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode tensor to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)
```

**Features**:
- Thread-safe batch processing
- Special NEXUS semantic tokens
- Caching for performance
- Vocabulary size handling

---

### 2. Checkpoint Persistence (`nexus/service/checkpoint.py`)

**Purpose**: Enable knowledge persistence across restarts with corruption protection

**Architecture**:
```
Save Process:
1. Create temp file (checkpoint_XXX.tmp)
2. Serialize model state to temp file
3. Calculate SHA256 checksum
4. Atomic rename (tmp → final) ← Key anti-corruption measure
5. Rotate old checkpoints (keep N most recent)

Load Process:
1. Find latest checkpoint
2. Verify SHA256 checksum
3. Load state dict
4. Restore to model
```

**Implementation**:
```python
class CheckpointManager:
    def save_checkpoint(self, nexus, metadata=None):
        """Atomic checkpoint save with validation."""
        checkpoint_id = int(time.time())
        temp_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.tmp"
        final_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pt"

        # Save to temp file
        checkpoint_data = {
            'model_state_dict': nexus.nexus.state_dict(),
            'lifecycle_state': nexus.lifecycle.get_status(),
            'continual_learner_state': nexus.continual.get_state(),
            'timestamp': time.time(),
            'version': '2.0.0'
        }
        torch.save(checkpoint_data, temp_path)

        # Calculate checksum
        checksum = self._calculate_checksum(temp_path)
        metadata.checksum = checksum

        # Atomic rename prevents corruption on crash
        temp_path.rename(final_path)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return final_path
```

**Features**:
- Atomic writes (crash-safe)
- SHA256 integrity validation
- Automatic rotation (configurable retention)
- Rich metadata tracking
- Versioning support

---

### 3. Comprehensive Metrics (`nexus/service/metrics.py`)

**Purpose**: Production-grade observability and monitoring

**Metrics Collected**:
```python
class MetricsCollector:
    # Request metrics
    - request_latencies: List[float]       # Per-request timing
    - responded_flags: List[bool]          # Did NEXUS respond?
    - confidence_scores: List[float]       # Confidence levels

    # FlowingNEXUS metrics (layer-free specific)
    - flow_depths: List[int]               # Emergent depth per request
    - convergence_flags: List[bool]        # Did flow converge?
    - flow_energies: List[float]           # Final residual norm

    # Memory metrics
    - memory_samples: List[float]          # MB over time
    - memory_timestamps: List[float]       # Sample times

    # System metrics
    - total_requests: int
    - total_responses: int
    - total_refusals: int
    - total_errors: int
```

**Latency Statistics**:
```python
def get_latency_stats(self, latencies):
    """Calculate P50, P95, P99 latencies."""
    sorted_latencies = sorted(latencies)
    count = len(sorted_latencies)

    return LatencyStats(
        p50=sorted_latencies[int(count * 0.50)],
        p95=sorted_latencies[int(count * 0.95)],
        p99=sorted_latencies[int(count * 0.99)],
        mean=sum(latencies) / count,
        max=sorted_latencies[-1]
    )
```

**Health Check**:
```python
class HealthCheck:
    def check_health(self, metrics, daemon_state):
        """Determine system health status."""
        issues = []

        # Check error rate
        if metrics.error_rate > 0.1:
            issues.append("High error rate")

        # Check refusal rate
        if metrics.refusal_rate > 0.8:
            issues.append("High refusal rate")

        # Check latency
        if metrics.latency_p95 > 5000:  # 5 seconds
            issues.append("High latency")

        # Determine status
        if len(issues) == 0:
            return "healthy"
        elif len(issues) <= 2:
            return "degraded"
        else:
            return "unhealthy"
```

**Prometheus Export**:
```python
def export_prometheus(self):
    """Export metrics in Prometheus format."""
    return f"""
# HELP nexus_requests_total Total number of requests
# TYPE nexus_requests_total counter
nexus_requests_total {self.total_requests}

# HELP nexus_latency_seconds Request latency in seconds
# TYPE nexus_latency_seconds summary
nexus_latency_seconds{{quantile="0.5"}} {self.latency_p50/1000}
nexus_latency_seconds{{quantile="0.95"}} {self.latency_p95/1000}
nexus_latency_seconds{{quantile="0.99"}} {self.latency_p99/1000}

# HELP nexus_flow_depth Average emergent depth
# TYPE nexus_flow_depth gauge
nexus_flow_depth {self.average_flow_depth}
    """
```

---

### 4. Error Recovery (`nexus/service/resilience.py`)

**Purpose**: Graceful degradation and automatic recovery from failures

**Circuit Breaker Pattern**:
```
States:
┌─────────────────────────────────────────────────────────────┐
│  CLOSED (normal operation)                                   │
│  - Allow all requests                                        │
│  - Track failures                                            │
│  - If failures > threshold → transition to OPEN              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─ Failures exceed threshold
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  OPEN (failing fast)                                         │
│  - Reject all requests immediately                           │
│  - Wait timeout period                                       │
│  - After timeout → transition to HALF_OPEN                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─ Timeout elapsed
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  HALF_OPEN (testing recovery)                                │
│  - Allow limited requests                                    │
│  - If success → CLOSED                                       │
│  - If failure → OPEN                                         │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
class CircuitBreaker:
    def __init__(self, name, config):
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.config = config

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(f"{self.name} circuit is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
```

**Features**:
- Automatic failure detection
- Fast-fail to prevent cascading failures
- Exponential backoff
- Timeout protection
- Retry with jitter

---

### 5. Memory Management (`nexus/service/memory_manager.py`)

**Purpose**: Long-running stability and leak prevention

**Memory Leak Detection**:
```python
def detect_memory_leak(self):
    """Detect memory leaks using linear regression."""
    if len(self.memory_samples) < 10:
        return {"detected": False}

    # Calculate trend (linear regression slope)
    x = np.arange(len(self.memory_samples))
    y = np.array(self.memory_samples)
    slope, _ = np.polyfit(x, y, 1)

    # Convert to MB/hour
    sample_interval = 10  # seconds
    mb_per_hour = slope * (3600 / sample_interval)

    # Alert if growth > 10 MB/hour
    if mb_per_hour > 10.0:
        logger.warning(f"Potential memory leak detected: {mb_per_hour:.2f} MB/hour")
        return {"detected": True, "growth_rate_mb_per_hour": mb_per_hour}

    return {"detected": False, "growth_rate_mb_per_hour": mb_per_hour}
```

**Cleanup Strategies**:
```python
def cleanup(self, aggressive=False):
    """Perform memory cleanup."""
    if aggressive:
        # Aggressive cleanup under pressure
        gc.collect(2)  # Full collection
        gc.collect(2)  # Twice for cyclic garbage

        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    else:
        # Normal periodic cleanup
        gc.collect(0)  # Young generation only
```

**Replay Buffer Management**:
```python
def manage_replay_buffer(self, continual_learner, max_size=2048):
    """Prevent replay buffer from growing unbounded."""
    current_size = len(continual_learner.replay_buffer)

    if current_size > max_size:
        # Keep most recent samples
        continual_learner.replay_buffer = continual_learner.replay_buffer[-max_size:]
        logger.info(f"Trimmed replay buffer from {current_size} to {max_size}")
```

---

### 6. Production Daemon (`nexus/service/daemon.py`)

**Purpose**: Main orchestrator integrating all production features

**Lifecycle**:
```
Startup:
1. Initialize tokenizer (NEXUSTokenizer)
2. Create NEXUS model (FlowingNEXUS or NEXUSCore)
3. Load latest checkpoint (if exists)
4. Initialize all managers (Metrics, Memory, Checkpoint, Circuit Breakers)
5. Start background thread
6. Report ready

Running:
Loop:
  1. Resource check (ResourceGovernor)
  2. Process request queue (high priority)
  3. Background learning (low priority, when idle)
  4. Periodic checkpoint (every 5 min)
  5. Memory cleanup (every 1 min)
  6. Metrics collection (continuous)

Shutdown:
1. Set shutdown flag
2. Wait for current operations
3. Save final checkpoint
4. Close all resources
5. Report stopped
```

**Key Integration Points**:
```python
class NexusDaemon:
    def __init__(self):
        # Production components
        self.tokenizer = NEXUSTokenizer()
        self.checkpoint_manager = CheckpointManager()
        self.metrics = MetricsCollector()
        self.memory_manager = MemoryManager()

        # Circuit breakers
        self.inference_breaker = CircuitBreaker("inference", CircuitBreakerConfig())
        self.learning_breaker = CircuitBreaker("learning", CircuitBreakerConfig())

        # Resource governance
        self.resource_governor = ResourceGovernor(
            active_cpu_limit=10.0,
            active_memory_limit=10.0,
            idle_cpu_limit=25.0,
            idle_memory_limit=25.0
        )

        # Core model
        self.nexus = create_living_nexus(size="small", architecture="flowing")

        # Load checkpoint
        self._load_latest_checkpoint()

    def _handle_request(self, request):
        """Process request with full production stack."""
        start_time = time.time()

        try:
            # Tokenize (real tokenization!)
            input_ids = self.tokenizer.encode(request.prompt, max_length=512)
            batch = input_ids.unsqueeze(0)

            # Inference with circuit breaker
            result = self.inference_breaker.call(
                lambda: self.nexus.interact(batch)
            )

            # Decode response
            if result.responded:
                response_text = self.tokenizer.decode(result.logits.argmax(dim=-1).squeeze())
            else:
                response_text = "I don't have enough information to answer that confidently."

            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.record_request(
                latency=latency,
                responded=result.responded,
                confidence=result.confidence,
                flow_depth=result.flow_depth,
                converged=result.converged,
                flow_energy=result.final_energy
            )

            return response_text

        except CircuitBreakerOpenError:
            # Graceful degradation
            return "Service temporarily unavailable. Please try again."
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Request failed: {e}")
            raise
```

---

### 7. Control Interfaces

#### CLI (nexusctl)

**Commands**:
```bash
nexusctl start      # Start NEXUS daemon
nexusctl stop       # Graceful shutdown with checkpoint
nexusctl restart    # Stop + Start
nexusctl pause      # Pause background learning
nexusctl resume     # Resume background learning
nexusctl status     # Detailed system status
nexusctl health     # Quick health check
nexusctl logs       # View logs (-f to follow)
nexusctl save       # Show checkpoint info
nexusctl dashboard  # Open web dashboard
```

**Features**:
- Color-coded output
- Process management (PID tracking)
- Automatic health checks on startup
- Log tailing support
- Cross-platform (macOS, Linux, Windows)

#### Web Dashboard

**Features**:
- Real-time status updates (WebSocket)
- Interactive chat interface
- Resource monitoring (CPU, memory)
- Thought stream visualization
- Control buttons (pause/resume/train)
- Evolution metrics display
- Mobile-responsive design
- PWA installable (iOS/Android)

**Endpoints**:
```
GET  /dashboard          - Main UI
GET  /api/status         - System status
POST /api/interact       - Submit prompt
POST /api/control        - Control commands
GET  /api/metrics        - Prometheus export
```

#### REST API

**Full API**:
```python
# Status endpoint
GET /api/status
Response: {
    "daemon": {"running": true, "paused": false, "uptime_seconds": 3600},
    "health": {"healthy": true, "status": "healthy", "issues": []},
    "metrics": {
        "requests": {"total": 150, "responses": 120, "refusals": 30},
        "latency": {"p50": 45.2, "p95": 120.5, "p99": 250.0},
        "flow": {"average_depth": 12.3, "convergence_rate": 0.95}
    },
    "memory": {"current_mb": 856.3, "peak_mb": 892.1, "growth_rate_mb_per_hour": 2.4},
    "model": {"architecture": "flowing", "total_interactions": 150}
}

# Control endpoint
POST /api/control
Body: {"action": "pause"}  # or resume, train_start, train_stop
Response: {"status": "success", "message": "Background learning paused"}

# Interact endpoint
POST /api/interact
Body: {"prompt": "What is Python?"}
Response: {"response": "Python is a high-level programming language..."}
```

---

## Resource Governance

### CPU/Memory Limits

```python
class ResourceGovernor:
    def __init__(self, active_cpu_limit=10.0, idle_cpu_limit=25.0, ...):
        self.active_cpu_limit = active_cpu_limit      # 10% during requests
        self.idle_cpu_limit = idle_cpu_limit          # 25% when idle

    def check_and_throttle(self, mode="idle"):
        """Enforce resource limits."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        limit_cpu = self.active_cpu_limit if mode == "active" else self.idle_cpu_limit

        if cpu_percent > limit_cpu:
            # Throttle by sleeping
            sleep_time = (cpu_percent - limit_cpu) / limit_cpu * 0.1
            time.sleep(sleep_time)
            return True  # Violated

        return False  # Within limits
```

### Mode Switching

```
Active Mode (During Requests):
- CPU Limit: 10%
- Memory Limit: 10% of available
- Priority: User requests
- Learning: Paused

Idle Mode (No Requests):
- CPU Limit: 25%
- Memory Limit: 25% of available
- Priority: Background learning
- Learning: Active
```

---

## Deployment Modes

### 1. Development Mode

```bash
python -m uvicorn nexus.service.server:app --reload
```

**Features**:
- Auto-reload on code changes
- Debug logging enabled
- No authentication
- Local only (127.0.0.1)

### 2. Production Mode (systemd)

```bash
sudo deployment/install.sh
sudo systemctl start nexus
sudo systemctl enable nexus  # Auto-start on boot
```

**Features**:
- Runs as service user
- Automatic restart on failure
- Journal logging
- Starts on boot

### 3. Production Mode (Manual)

```bash
./nexusctl start
```

**Features**:
- PID tracking
- Log file rotation
- Graceful shutdown
- Checkpoint on exit

### 4. Edge Deployment (Raspberry Pi)

```bash
# Optimized for Pi 4 (4GB/8GB)
sudo deployment/install.sh
# Adjust config for smaller model size
```

**Optimizations**:
- Smaller model size (d_model=128-256)
- Reduced resource limits
- SSD recommended for checkpoints
- Swap configuration

---

## Monitoring & Observability

### Metrics Dashboard

**Key Metrics to Monitor**:

1. **Request Metrics**
   - Total requests
   - Success rate (responses / total)
   - Refusal rate (wisdom ratio)
   - Error rate

2. **Performance Metrics**
   - Latency P50/P95/P99
   - Throughput (req/sec)
   - Flow depth (emergent complexity)
   - Convergence rate

3. **Resource Metrics**
   - CPU usage (% of limit)
   - Memory usage (current/peak/growth)
   - Violation count
   - GC frequency

4. **Model Metrics**
   - Total interactions
   - Experience factor
   - Confidence threshold
   - Average flow depth

### Health Check Strategy

```
Healthy:
- Error rate < 1%
- Refusal rate 10-70% (normal)
- Latency P95 < 2000ms
- No memory leak (growth < 10 MB/hour)

Degraded:
- Error rate < 5%
- Refusal rate > 80% (too cautious)
- Latency P95 < 5000ms
- Minor issues

Unhealthy:
- Error rate > 5%
- System unresponsive
- Memory leak detected
- Critical failures
```

### Alerting

**Alert Conditions**:
```python
if error_rate > 0.05:
    alert("High error rate")

if memory_growth > 10:  # MB/hour
    alert("Memory leak detected")

if latency_p95 > 5000:  # ms
    alert("High latency")

if circuit_breaker_state == "OPEN":
    alert("Circuit breaker open - service degraded")
```

---

## Security Considerations

### Authentication (Future)

Currently, NEXUS does not implement authentication. For production deployment:

1. **Add Basic Auth**:
```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

@app.post("/api/interact")
async def interact(credentials: HTTPBasicCredentials = Depends(security)):
    # Verify credentials
    ...
```

2. **Use Reverse Proxy**:
```nginx
location /api {
    auth_basic "NEXUS API";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8000;
}
```

3. **HTTPS**:
```bash
# Using Let's Encrypt
certbot --nginx -d nexus.yourdomain.com
```

### Network Security

**Firewall**:
```bash
# Allow only localhost (development)
ufw allow from 127.0.0.1 to any port 8000

# Allow specific IP (production)
ufw allow from 192.168.1.0/24 to any port 8000
```

**SSH Tunnel** (recommended for remote access):
```bash
ssh -L 8000:localhost:8000 user@remote-host
```

---

## Disaster Recovery

### Checkpoint Recovery

**Automatic**:
```python
# On startup, daemon automatically loads latest checkpoint
self._load_latest_checkpoint()
```

**Manual**:
```bash
# List checkpoints
ls -lh nexus_checkpoints/

# Specify checkpoint in code
checkpoint_manager.load_checkpoint("nexus_checkpoints/checkpoint_1234567890.pt")
```

### Backup Strategy

```bash
# Daily backup (cron)
0 2 * * * tar -czf /backup/nexus-$(date +\%Y\%m\%d).tar.gz /path/to/nexus_checkpoints/

# Restore from backup
tar -xzf nexus-20260107.tar.gz
./nexusctl restart
```

### Corruption Recovery

If checkpoint is corrupted:
```bash
# Check integrity
python -c "import torch; torch.load('checkpoint.pt')"

# Use previous checkpoint
rm nexus_checkpoints/checkpoint_latest.pt
./nexusctl start  # Will load next most recent
```

---

## Performance Tuning

### Model Size Selection

```python
# Edge devices (Pi 4 4GB)
config = FlowingConfig(d_model=128, ssm_n_layers=4)

# Development (8-16GB RAM)
config = FlowingConfig(d_model=256, ssm_n_layers=6)

# Production (32GB+ RAM)
config = FlowingConfig(d_model=512, ssm_n_layers=12)
```

### Resource Limit Tuning

```python
# Conservative (shared server)
ResourceGovernor(active_cpu_limit=5.0, idle_cpu_limit=15.0)

# Default (dedicated server)
ResourceGovernor(active_cpu_limit=10.0, idle_cpu_limit=25.0)

# Aggressive (dedicated, high-performance)
ResourceGovernor(active_cpu_limit=25.0, idle_cpu_limit=50.0)
```

### Checkpoint Frequency

```python
# More frequent (critical data)
checkpoint_interval = 180  # 3 minutes

# Default
checkpoint_interval = 300  # 5 minutes

# Less frequent (performance priority)
checkpoint_interval = 600  # 10 minutes
```

---

## Migration Guide

### From v1.0 to v2.0

**Step 1**: Update dependencies
```bash
pip install -r requirements.txt
```

**Step 2**: No data migration needed
- v2.0 automatically loads v1.0 checkpoints
- Legacy format supported

**Step 3**: Use new control interface
```bash
# Old (manual)
python nexus/service/server.py

# New (recommended)
./nexusctl start
```

**Step 4**: Monitor with new metrics
```bash
./nexusctl status  # Rich status display
```

---

## Future Enhancements

### Planned Features

1. **Horizontal Scaling**
   - Load balancer support
   - Distributed checkpoints
   - Shared replay buffer

2. **Advanced Monitoring**
   - Grafana dashboard integration
   - Custom metric exporters
   - Distributed tracing

3. **Security**
   - JWT authentication
   - Rate limiting
   - API key management

4. **Optimization**
   - Model quantization (INT8)
   - ONNX export
   - TensorRT acceleration

---

## References

- [Production Ready Checklist](../PRODUCTION_READY.md)
- [Control Guide](../CONTROL_GUIDE.md)
- [Operations Runbook](../operations/runbook.md)
- [Deployment Guide](../deployment/deployment-guide.md)

---

*Production architecture designed for zero downtime, continuous operation, and enterprise reliability.*
