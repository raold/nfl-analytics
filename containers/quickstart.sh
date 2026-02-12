#!/bin/bash
#
# Quick start script for NFL Analytics Docker + Redis system
# Checks all dependencies and services before starting

set -e

echo "🚀 NFL Analytics Quick Start"
echo "============================"
echo

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check service is running
service_running() {
    pgrep -x "$1" >/dev/null 2>&1
}

# 1. Check Docker
echo "🐳 Checking Docker..."
if command_exists docker; then
    echo "✅ Docker is installed"
    if docker ps >/dev/null 2>&1; then
        echo "✅ Docker daemon is running"
    else
        echo "❌ Docker daemon is not running"
        echo "   Start Docker Desktop or run: sudo systemctl start docker"
        exit 1
    fi
else
    echo "❌ Docker is not installed"
    echo "   Install from: https://www.docker.com/get-started"
    exit 1
fi

# 2. Check Docker Compose
echo
echo "📦 Checking Docker Compose..."
if command_exists docker-compose; then
    echo "✅ Docker Compose is installed"
else
    echo "❌ Docker Compose is not installed"
    echo "   Install with: pip install docker-compose"
    exit 1
fi

# 3. Check Python dependencies
echo
echo "🐍 Checking Python dependencies..."
cd "$(dirname "$0")/.."
if python3 containers/check_dependencies.py; then
    echo "✅ Python dependencies OK"
else
    echo "⚠️ Some Python dependencies missing"
    read -p "Install missing dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    fi
fi

# 4. Check Redis (for local testing)
echo
echo "🔧 Checking Redis..."
if command_exists redis-server; then
    echo "✅ Redis is installed"
    if redis-cli ping >/dev/null 2>&1; then
        echo "✅ Redis is running"
    else
        echo "⚠️ Redis is not running"
        read -p "Start Redis? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            redis-server --daemonize yes
            echo "✅ Redis started"
        fi
    fi
else
    echo "⚠️ Redis not installed (Docker will provide it)"
fi

# 5. Check Google Drive sync directory
echo
echo "☁️ Checking Google Drive sync..."
SYNC_DIR="${GOOGLE_DRIVE_DIR:-$HOME/Google Drive/nfl-analytics}"
if [ -d "$SYNC_DIR" ]; then
    echo "✅ Google Drive sync directory found: $SYNC_DIR"
else
    echo "⚠️ Google Drive sync directory not found"
    echo "   Expected: $SYNC_DIR"
    echo "   Set GOOGLE_DRIVE_DIR environment variable if using different path"
    read -p "Continue without Google Drive sync? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 6. Environment setup
echo
echo "⚙️ Setting up environment..."
cd containers
if [ ! -f .env ]; then
    if [ -f .env.template ]; then
        cp .env.template .env
        echo "✅ Created .env from template"
        echo "   Edit containers/.env to customize settings"
    else
        echo "⚠️ No .env file found"
    fi
else
    echo "✅ .env file exists"
fi

# 7. Build Docker image
echo
echo "🔨 Building Docker image..."
read -p "Build Docker image? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose build
    echo "✅ Docker image built"
else
    echo "⏭️ Skipping Docker build"
fi

# 8. Initialize task queue
echo
echo "📊 Initializing task queue..."
read -p "Initialize task queue with standard tasks? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose run --rm nfl-analytics python run_compute.py --redis-mode --init
    echo "✅ Task queue initialized"
else
    echo "⏭️ Skipping queue initialization"
fi

# 9. Ready to start
echo
echo "========================================="
echo "✅ System is ready!"
echo "========================================="
echo
echo "Start the distributed compute system with:"
echo "  cd containers"
echo "  docker-compose up"
echo
echo "Or run in background:"
echo "  docker-compose up -d"
echo
echo "Check status:"
echo "  docker-compose exec nfl-analytics python run_compute.py --redis-mode --status"
echo
echo "Stop system:"
echo "  docker-compose down"
echo
echo "For more information, see containers/README.md"