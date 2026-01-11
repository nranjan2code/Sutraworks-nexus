"""
Nexus Service API
=================

FastAPI server that exposes the Nexus Continuum to the world.
Serves:
1. REST API for control and interaction.
2. Web Dashboard for visualization.

Security:
- API key authentication (via NEXUS_API_KEY env var)
- Rate limiting (60 requests/minute default)
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
import os
import uvicorn
import math
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from nexus.service.daemon import NexusDaemon
from nexus.service.auth import get_auth_manager, get_client_ip, AuthManager
from nexus.service.hardware import detect_hardware, HardwareCapabilities

# Use centralized logging
from nexus.service.logging_config import get_logger

logger = get_logger("server")

# Global Daemon Instance
daemon = NexusDaemon()

# Authentication Manager
auth_manager = get_auth_manager()

# Rate Limiting (optional - only if slowapi is installed)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_client_ip)
    RATE_LIMITING_ENABLED = True
    logger.info(f"Rate limiting enabled: {auth_manager.get_rate_limit_string()}")
except ImportError:
    limiter = None
    RATE_LIMITING_ENABLED = False
    logger.warning("slowapi not installed - rate limiting disabled")


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
    logger.info("Service Startup: Detecting hardware...")
    hardware = detect_hardware()
    logger.info(f"Hardware: {hardware.summary()}")
    logger.info("Service Startup: Launching Daemon...")
    daemon.startup()
    yield
    # Shutdown
    logger.info("Service Shutdown: Stopping Daemon...")
    daemon.shutdown()


app = FastAPI(
    title="Nexus Continuum",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
    description="Production-ready AI system with dynamic hardware utilization",
    version="2.1.0",
)

# Add rate limiting middleware if available
if RATE_LIMITING_ENABLED:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
    status = daemon.get_status()
    # Add hardware info
    status["hardware"] = detect_hardware().to_dict()
    return status


@app.get("/api/hardware")
async def get_hardware():
    """Get detected hardware capabilities."""
    return detect_hardware().to_dict()


# Rate-limited interact endpoint
if RATE_LIMITING_ENABLED:

    @app.post("/api/interact")
    @limiter.limit(auth_manager.get_rate_limit_string())
    async def interact(request: Request, req: InteractRequest):
        """Send a prompt to Nexus (rate limited)."""
        if not daemon.running:
            raise HTTPException(status_code=503, detail="Daemon is not running")
        try:
            response = daemon.submit_request(req.prompt)
            return {"response": response}
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

else:

    @app.post("/api/interact")
    async def interact(req: InteractRequest):
        """Send a prompt to Nexus."""
        if not daemon.running:
            raise HTTPException(status_code=503, detail="Daemon is not running")
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
