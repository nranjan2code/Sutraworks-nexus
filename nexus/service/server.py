"""
Nexus Service API
=================

FastAPI server that exposes the Nexus Continuum to the world.
Serves:
1. REST API for control and interaction.
2. Web Dashboard for visualization.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
import logging
import os
import uvicorn
import math
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from nexus.service.daemon import NexusDaemon

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.server")

# Global Daemon Instance
daemon = NexusDaemon()


class SafeJSONResponse(JSONResponse):
    """JSONResponse that replaces NaN/Inf with None to allow valid JSON serialization."""

    def render(self, content: Any) -> bytes:
        def clean_float(v):
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
            return v

        # We do a recursive clean for common structures
        # For a truly robust solution, we might want to use a custom json.JSONEncoder
        def clean_structure(obj):
            if isinstance(obj, dict):
                return {k: clean_structure(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_structure(v) for v in obj]
            elif isinstance(obj, float):
                return clean_float(obj)
            return obj

        return super().render(clean_structure(content))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Service Startup: Launching Daemon...")
    daemon.startup()
    yield
    # Shutdown
    logger.info("Service Shutdown: Stopping Daemon...")
    daemon.shutdown()


app = FastAPI(title="Nexus Continuum", lifespan=lifespan, default_response_class=SafeJSONResponse)


# Models
class InteractRequest(BaseModel):
    prompt: str


class ControlRequest(BaseModel):
    action: str  # pause, resume, reset, train_start
    topic: Optional[str] = None


class ConfigRequest(BaseModel):
    ollama_host: str
    ollama_model: str


# API Endpoints
@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return {
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama2"),
    }


@app.post("/api/config")
async def set_config(req: ConfigRequest):
    """Update configuration and reload daemon."""
    # Update env vars for this process
    os.environ["OLLAMA_HOST"] = req.ollama_host
    os.environ["OLLAMA_MODEL"] = req.ollama_model

    # Persist to .env file (simple append/replace for now)
    # A robust solution would use a proper parser, but we'll do simple write
    with open(".env", "w") as f:
        f.write(f"OLLAMA_HOST={req.ollama_host}\n")
        f.write(f"OLLAMA_MODEL={req.ollama_model}\n")

    # Reload Daemon Component
    daemon.reload_teacher()

    return {"status": "updated", "config": req.dict()}


@app.get("/api/ollama/tags")
async def get_ollama_tags():
    """Proxy to discover available Ollama models."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        # Use httpx or requests (requests is sync but simple for now)
        import requests

        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return {"models": []}
    except Exception as e:
        logger.error(f"Failed to fetch Ollama tags: {e}")
        return {"error": str(e), "models": []}


@app.get("/api/status")
async def get_status():
    """Get full system status for dashboard."""
    return daemon.get_status()


@app.post("/api/interact")
async def interact(req: InteractRequest):
    """Send a prompt to Nexus."""
    if not daemon.running:
        raise HTTPException(status_code=503, detail="Daemon is not running")

    start_time = os.times()
    # This calls the daemon which queues the request
    # Note: daemon.submit_request is currently blocking in our simplistic impl
    # In a real async app we'd await a future.
    try:
        response = daemon.submit_request(req.prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control")
async def control(req: ControlRequest):
    """Control the daemon state."""
    if req.action == "pause":
        daemon.pause()
        return {"status": "paused"}
    elif req.action == "resume":
        daemon.resume()
        return {"status": "resumed"}
    elif req.action == "train_start":
        daemon.set_training_mode(True, topic=req.topic)
        return {"status": "training_started", "topic": req.topic}
    elif req.action == "train_stop":
        daemon.set_training_mode(False)
        return {"status": "training_stopped"}
    else:
        raise HTTPException(status_code=400, detail="Unknown action")


# Serve Dashboard
# We will assume dashboard.html is in the same directory for simplicity
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open(os.path.join(CURRENT_DIR, "dashboard.html"), "r") as f:
        return f.read()


@app.get("/")
async def root():
    return FileResponse(os.path.join(CURRENT_DIR, "dashboard.html"))


if __name__ == "__main__":
    uvicorn.run("nexus.service.server:app", host="0.0.0.0", port=8000, reload=True)
