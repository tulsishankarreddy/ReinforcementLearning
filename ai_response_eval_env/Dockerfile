# Self-contained Dockerfile for HF Spaces deployment.
# Uses python:3.12-slim + pip (no dependence on ghcr.io/meta-pytorch/openenv-base
# which may not be pullable from HF's build environment).

FROM python:3.12-slim AS base

WORKDIR /app

# System deps: curl for the healthcheck, build-essential in case any wheel needs compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the environment code
COPY . /app

# Make the package importable both as `ai_response_eval_env.*` and as bare `server.app`
ENV PYTHONPATH="/app:/app/.."

# HF Spaces convention: app must listen on $PORT (defaults to 7860)
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Run the FastAPI server. server/app.py exposes `app`.
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
