# NFL Analytics - Multi-stage Production Dockerfile
# Optimized for Python 3.11+ with minimal image size

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 nfl && \
    mkdir -p /app /data /models /logs && \
    chown -R nfl:nfl /app /data /models /logs

WORKDIR /app

# Copy requirements and install Python dependencies
COPY --chown=nfl:nfl requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=nfl:nfl . .

# Switch to non-root user
USER nfl

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "nfl_analytics.main"]


# ============================================
# Training stage - includes ML/DL dependencies
# ============================================
FROM base as training

USER root

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    pytorch-lightning==2.1.0 \
    optuna==3.4.0 \
    tensorboard==2.15.0

USER nfl

CMD ["python", "-m", "nfl_analytics.models.train_all"]


# ============================================
# Dashboard stage - Streamlit UI
# ============================================
FROM base as dashboard

USER root

RUN pip install --no-cache-dir \
    streamlit==1.28.0 \
    plotly==5.18.0

USER nfl

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]


# ============================================
# API stage - FastAPI service
# ============================================
FROM base as api

USER root

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0

USER nfl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]


# ============================================
# Backtesting stage - Offline evaluation
# ============================================
FROM base as backtest

USER nfl

CMD ["python", "-m", "nfl_analytics.backtests.comprehensive_ensemble_backtest"]
