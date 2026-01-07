# NEXUS Continuum - Production Dockerfile
# ========================================
#
# Multi-stage build for optimized image size
# Supports CPU and GPU deployment

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ======================================
# Stage 1: Dependencies
# ======================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ======================================
# Stage 2: Application
# ======================================
FROM base as application

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY nexus/ ./nexus/
COPY examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/nexus_checkpoints \
    && mkdir -p /app/logs

# Create non-root user for security
RUN useradd -m -u 1000 nexus && \
    chown -R nexus:nexus /app

USER nexus

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "nexus.service.server:app", "--host", "0.0.0.0", "--port", "8000"]
