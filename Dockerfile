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

# Copy requirements and check/install Python dependencies
COPY requirements.txt .

# Smart dependency installation - only install what's missing
RUN python -c "
import subprocess
import sys

def check_and_install_packages():
    required_packages = []

    # Check each package before installing
    packages_to_check = [
        ('redis', 'redis>=4.5.0'),
        ('aioredis', 'aioredis>=2.0.0'),
        ('psutil', 'psutil>=5.9.0'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn')
    ]

    for import_name, package_spec in packages_to_check:
        try:
            __import__(import_name)
            print(f'✅ {import_name} already installed')
        except ImportError:
            print(f'📦 Need to install {package_spec}')
            required_packages.append(package_spec)

    # Install requirements.txt packages
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-r', 'requirements.txt'], check=True)
        print('✅ requirements.txt installed')
    except subprocess.CalledProcessError:
        print('⚠️ Some requirements.txt packages may have failed')

    # Install any additional missing packages
    if required_packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + required_packages, check=True)
        print(f'✅ Installed missing packages: {required_packages}')
    else:
        print('✅ All additional packages already present')

check_and_install_packages()
"

FROM base as worker

# Copy application code
COPY py/ ./py/
COPY run_compute.py .
COPY *.py .

# Copy configuration templates
COPY containers/config/ ./config/

# Create data directories
RUN mkdir -p /data/redis /data/results /data/checkpoints /data/logs

# Set environment variables
ENV PYTHONPATH="/app:/app/py:/app/py/compute"
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