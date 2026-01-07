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
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os
import uvicorn
from contextlib import asynccontextmanager

from nexus.service.daemon import NexusDaemon

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.server")

# Global Daemon Instance
daemon = NexusDaemon()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Service Startup: Launching Daemon...")
    daemon.startup()
    yield
    # Shutdown
    logger.info("Service Shutdown: Stopping Daemon...")
    daemon.shutdown()


app = FastAPI(title="Nexus Continuum", lifespan=lifespan)


# Models
class InteractRequest(BaseModel):
    prompt: str


class ControlRequest(BaseModel):
    action: str  # pause, resume, reset, train_start
    topic: Optional[str] = None


# API Endpoints
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
    response = daemon.submit_request(req.prompt)
    return {"response": response}


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
