# MCP Server Dockerfile for NFL Analytics
# Multi-stage build for efficient containerization

FROM python:3.12-slim as builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files for dependency installation
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system \
    mcp[cli]>=1.2.0 \
    psycopg2-binary>=2.9.0 \
    python-dotenv>=1.0.0 \
    requests>=2.32.0 \
    beautifulsoup4>=4.12.0

# Final stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy MCP server code
COPY mcp_server/ ./mcp_server/

# Create necessary directories
RUN mkdir -p logs/mcp cache/mcp

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_LOG_LEVEL=INFO

# Expose MCP server port (stdio mode doesn't need expose, but useful for documentation)
EXPOSE 3000

# Health check (optional - checks if the server module can be imported)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mcp_server" || exit 1

# Run MCP server
CMD ["python", "-m", "mcp_server.server"]
