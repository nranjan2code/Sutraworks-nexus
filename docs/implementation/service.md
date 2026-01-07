# Nexus Continuum Service

The **Nexus Continuum** service transforms the NEXUS model from a static artifact into a living, always-on system.

## 🧠 Philosophy

> "AI should not be a tool you pick up and put down. It should be a presence that grows with you."

The service implements this by running a background **Daemon** that:
1.  **Prioritizes You**: Instantly wakes up for user interactions (Active Mode).
2.  **Dreams While Idle**: Switches to "Idle Mode" (low resource usage) to replay memories and learn.
3.  **Respects Your Hardware**: Strictly governs CPU/RAM usage.

## 🏗️ Architecture

### 1. The Daemon (`nexus.service.daemon`)
- **Role**: Central orchestrator.
- **Loop**:
    - Check resources.
    - Perform memory management and cleanup.
    - Check request queue (High Priority).
    - If empty, run background learning (Low Priority).
    - Periodic checkpointing (every 5 minutes).
- **Concurrency**: Runs in a dedicated thread; fully non-blocking for API server.

### 2. Resource Governor (`nexus.service.resource`)
- **Role**: Hardware constraint enforcer.
- **Constraints**:
    - **Active**: Max 10% CPU / RAM.
    - **Idle**: Max 25% CPU / RAM.
- **Mechanism**: Monitors `psutil` stats. If limits exceeded, injects sleep cycles (throttling).

### 3. Memory Manager (`nexus.service.memory_manager`)
- **Role**: Ensures stable long-running operation.
- **Features**:
    - **Periodic Garbage Collection**: Runs every 10 minutes.
    - **Cache Cleanup**: Clears PyTorch CUDA/MPS caches every 5 minutes.
    - **Memory Leak Detection**: Monitors growth rate; warns if > 10 MB/hour.
    - **Replay Buffer Management**: Trims to 75% at 90% capacity.
    - **History Trimming**: Caps thought/metric histories to prevent unbounded growth.
- **Thresholds**:
    - **Warning**: 2 GB memory usage.
    - **Critical**: 4 GB memory usage (triggers aggressive cleanup).

### 4. Teacher-Student Module (`nexus.training.teacher`)
- **Role**: Bootstrap knowledge from stronger models.
- **Teacher**: Local **Ollama** instance (e.g., Llama 2, Mistral).
- **Directed Learning**:
    - Users can specify a **Focus Topic** in the dashboard or via API.
    - If a topic is set, the Teacher generates synthetic examples specifically about that topic.
    - If no topic is set, random academic topics are used.
- **Process**:
    1.  Daemon enters Idle Mode.
    2.  If `Bootstrap Training` is enabled, queries Teacher for a synthetic (Question, Answer) pair.
    3.  NEXUS trains on this pair to "distill" the teacher's knowledge.

### 5. Gradient Control (FlowingNEXUS)
- **Role**: Ensures memory stability for layer-free architecture.
- **Behavior**: 
    - FlowingNEXUS automatically **disables gradients** during inference.
    - Prevents memory leaks from computation graph accumulation.
    - Enables stable 24/7 operation without memory growth.
- **Note**: Learning is self-supervised via energy minimization, not backpropagation.

## 📊 Dashboard

The service serves a real-time dashboard at `http://localhost:8000/dashboard`.

### Features
- **Live Thought Stream**: Watch the model "thinking" or "dreaming".
- **Resource Bars**: Visual confirmation of CPU/RAM limits.
- **Evolution Metrics**: Track `Experience Factor` and `Flow Depth` in real-time.
- **Memory Status**: Current usage, peak, and growth rate.
- **Controls**: Pause, Resume, or Trigger Training with optional topic.

## 🚀 Usage

### Requirements
- `fastapi`, `uvicorn`, `psutil` (included in `requirements.txt`)
- Optional: [Ollama](https://ollama.com) for bootstrap training.

### Running
```bash
python -m uvicorn nexus.service.server:app --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard` | GET | Web-based monitoring dashboard |
| `/api/status` | GET | System health, metrics, and memory stats |
| `/api/metrics` | GET | Prometheus-format metrics |
| `/api/interact` | POST | Send a prompt to NEXUS |
| `/api/control` | POST | Control daemon behavior |

### Control Actions

```bash
# Pause background learning
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "pause"}'

# Resume background learning
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "resume"}'

# Start training (general)
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_start"}'

# Start training (with topic)
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_start", "topic": "quantum physics"}'

# Stop training
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action": "train_stop"}'
```

### Status Response Structure

```json
{
  "daemon": {"running": true, "paused": false, "uptime_seconds": 3600},
  "health": {"healthy": true, "status": "healthy", "issues": []},
  "model": {"architecture": "flowing", "interactions": 150},
  "metrics": {"total_requests": 150, "success_rate": 0.85},
  "memory": {"process": {"current_mb": 856, "growth_rate_mb_per_hour": 2.4}},
  "resources": {"mode": "idle", "cpu_percent": 15},
  "circuit_breakers": {"inference": "CLOSED", "learning": "CLOSED"},
  "checkpoints": {"total_size_mb": 45.2, "count": 5}
}
```
