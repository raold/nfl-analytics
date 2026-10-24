FROM python:3.11-slim as base

# Install system dependencies only if not present
RUN apt-get update && \
    # Check if redis-server is already installed
    (which redis-server > /dev/null 2>&1 || apt-get install -y redis-server) && \
    # Check if build tools are present
    (which gcc > /dev/null 2>&1 || apt-get install -y build-essential) && \
    # Check if curl is present
    (which curl > /dev/null 2>&1 || apt-get install -y curl) && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install the locked dependency set (pyproject.toml + uv.lock) plus extras: --extra compute
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.13.5 && \
    uv export --frozen --no-hashes --no-emit-project --extra compute -o /tmp/requirements.lock && \
    uv pip install --system --no-cache -r /tmp/requirements.lock

FROM base as worker

# Copy application code
COPY nfl_analytics/ ./nfl_analytics/
COPY run_compute.py .
COPY *.py .

# Copy configuration templates
COPY containers/config/ ./config/

# Create data directories
RUN mkdir -p /data/redis /data/results /data/checkpoints /data/logs

# Set environment variables
ENV PYTHONPATH="/app:/app/nfl_analytics:/app/nfl_analytics/compute"
ENV REDIS_HOST="localhost"
ENV REDIS_PORT="6379"
ENV MACHINE_ID=""
ENV HARDWARE_PROFILE=""

# Copy startup script
COPY containers/start.sh .
RUN chmod +x start.sh

# Expose Redis port (for debugging)
EXPOSE 6379

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD redis-cli ping || exit 1

# Start Redis and worker
CMD ["./start.sh"]