# Nexus Continuum Service

The **Nexus Continuum** service transforms the NEXUS model from a static artifact into a living, always-on system.

## 🧠 Philosophy

> "AI should not be a tool you pick up and put down. It should be a presence that grows with you."

The service implements this by running a background **Daemon** that:
1.  **Prioritizes You**: Instantly wakes up for user interactions (Active Mode).
2.  **Dreams While Idle**: Switchs to "Idle Mode" (low resource usage) to replay memories and learn.
3.  **Respects Your Hardware**: Strictly governs CPU/RAM usage.

## 🏗️ Architecture

### 1. The Daemon (`nexus.service.daemon`)
- **Role**: Central orchestrator.
- **Loop**:
    - Check resources.
    - Check request queue (High Priority).
    - If empty, run background learning (Low Priority).
- **Concurrency**: Runs in a dedicated thread; fully non-blocking for API server.

### 2. Resource Governor (`nexus.service.resource`)
- **Role**: Hardware constraint enforcer.
- **Constraints**:
    - **Active**: Max 10% CPU / RAM.
    - **Idle**: Max 25% CPU / RAM.
- **Mechanism**: Monitors `psutil` stats. If limits exceeded, injects sleep cycles (throttling).

### 3. Teacher-Student Module (`nexus.training.teacher`)
- **Role**: Bootstrap knowledge from stronger models.
- **Teacher**: Local **Ollama** instance (e.g., Llama 2, Mistral).
- **Directed Learning**:
    - Users can specify a **Focus Topic** in the dashboard.
    - If a topic is set, the Teacher generates synthetic examples specifically about that topic.
    - If no topic is set, random academic topics are used.
- **Process**:
    1.  Daemon enters Idle Mode.
    2.  If `Bootstrap Training` is enabled, queries Teacher for a synthetic (Question, Answer) pair.
    3.  NEXUS trains on this pair to "distill" the teacher's knowledge.

## 📊 Dashboard

The service serves a real-time dashboard at `http://localhost:8000/dashboard`.

### Features
- **Live Thought Stream**: Watch the model "thinking" or "dreaming".
- **Resource Bars**: Visual confirmation of CPU/RAM limits.
- **Evolution Metrics**: Track `Experience Factor` and `Flow Depth` in real-time.
- **Controls**: Pause, Resume, or Trigger Training.

## 🚀 Usage

### Requirements
- `fastapi`, `uvicorn`, `psutil` (included in `requirements.txt`)
- Optional: [Ollama](https://ollama.com) for bootstrap training.

### Running
```bash
python nexus/service/server.py
```

### API Endpoints
- `GET /api/status`: System health and metrics.
- `POST /api/interact`: Send a prompt.
- `POST /api/control`: Pause/Resume/Train.
