#!/bin/bash

set -e

echo "🫀 Starting NFL Analytics Distributed Computing System"
echo "=================================================="

# Check for cached machine capabilities first
MACHINE_CACHE="/data/machine_profile.json"
HARDWARE_CACHE="/data/hardware_profile.cache"

# Detect machine capabilities if not set or cached
if [ "$MACHINE_ID" = "auto" ] || [ -z "$MACHINE_ID" ]; then
    if [ -f "$MACHINE_CACHE" ] && [ "$(find $MACHINE_CACHE -mtime -7 2>/dev/null)" ]; then
        # Use cached machine ID if less than 7 days old
        export MACHINE_ID=$(python -c "
import json
try:
    with open('$MACHINE_CACHE', 'r') as f:
        data = json.load(f)
        print(data.get('machine_id', ''))
except:
    print('')
")
        echo "📋 Using cached machine ID: $MACHINE_ID"
    fi

    # Generate new machine ID if cache miss
    if [ -z "$MACHINE_ID" ] || [ "$MACHINE_ID" = "auto" ]; then
        echo "🔍 Detecting machine ID..."
        export MACHINE_ID=$(python -c "
import hashlib
import platform
import subprocess
import socket
import json

# Create unique machine ID based on hostname + MAC address
hostname = socket.gethostname()
try:
    mac = subprocess.check_output(['cat', '/sys/class/net/eth0/address']).decode().strip()
except:
    try:
        mac = subprocess.check_output(['ifconfig']).decode()[:100]  # First 100 chars as fallback
    except:
        mac = platform.node()

machine_id = hashlib.md5(f'{hostname}:{mac}'.encode()).hexdigest()[:12]
print(machine_id)

# Cache the result
try:
    cache_data = {'machine_id': machine_id, 'hostname': hostname, 'platform': platform.platform()}
    with open('$MACHINE_CACHE', 'w') as f:
        json.dump(cache_data, f, indent=2)
except:
    pass
")
        echo "✅ Generated machine ID: $MACHINE_ID"
    fi
fi

if [ "$HARDWARE_PROFILE" = "auto" ] || [ -z "$HARDWARE_PROFILE" ]; then
    if [ -f "$HARDWARE_CACHE" ] && [ "$(find $HARDWARE_CACHE -mtime -1 2>/dev/null)" ]; then
        # Use cached hardware profile if less than 1 day old
        export HARDWARE_PROFILE=$(cat "$HARDWARE_CACHE" 2>/dev/null || echo "")
        echo "📋 Using cached hardware profile: $HARDWARE_PROFILE"
    fi

    # Detect hardware profile if cache miss
    if [ -z "$HARDWARE_PROFILE" ] || [ "$HARDWARE_PROFILE" = "auto" ]; then
        echo "🔍 Detecting hardware capabilities..."
        export HARDWARE_PROFILE=$(python -c "
import platform
import subprocess
import sys

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if '4090' in gpu_name or 'RTX 40' in gpu_name:
            print('gpu_high')
        elif 'RTX' in gpu_name or 'GTX' in gpu_name:
            print('gpu_standard')
        else:
            print('gpu_basic')
    else:
        raise ImportError('No CUDA')
except:
    # No GPU or PyTorch, check CPU
    if 'arm64' in platform.machine().lower() or 'M1' in platform.processor() or 'M2' in platform.processor() or 'M3' in platform.processor() or 'M4' in platform.processor():
        print('cpu_arm_high')  # Apple Silicon
    else:
        print('cpu_x86')
")
        # Cache the result
        echo "$HARDWARE_PROFILE" > "$HARDWARE_CACHE" 2>/dev/null || true
        echo "✅ Detected hardware profile: $HARDWARE_PROFILE"
    fi
fi

echo "Machine ID: $MACHINE_ID"
echo "Hardware Profile: $HARDWARE_PROFILE"

# Configure Redis for analytical workloads
echo "📊 Configuring Redis for analytics..."
cat > /tmp/redis.conf << EOF
# Redis configuration for NFL analytics
port 6379
bind 127.0.0.1
maxmemory 8gb
maxmemory-policy allkeys-lru

# Persistence configuration
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Performance optimizations
tcp-keepalive 60
timeout 0
tcp-backlog 511

# Logging
loglevel notice
logfile /data/logs/redis.log

# Working directory
dir /data/redis
EOF

# Check if Redis is already running before starting
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is already running"
else
    echo "🔧 Starting Redis server..."
    redis-server /tmp/redis.conf --daemonize yes

    # Wait for Redis to be ready
    echo "⏳ Waiting for Redis to be ready..."
    for i in {1..30}; do
        if redis-cli ping > /dev/null 2>&1; then
            echo "✅ Redis is ready"
            break
        fi
        sleep 1
    done

    if ! redis-cli ping > /dev/null 2>&1; then
        echo "❌ Redis failed to start"
        exit 1
    fi
fi

# Load existing Redis data if available
if [ -f "/data/redis/dump.rdb" ]; then
    echo "📥 Loading existing Redis data..."
    redis-cli FLUSHALL
    redis-cli DEBUG RELOAD
fi

# Set up sync management
echo "🔄 Setting up Google Drive sync..."
cat > /tmp/sync_manager.py << 'EOF'
import time
import subprocess
import os
import redis
import json
from datetime import datetime
from pathlib import Path

def sync_redis_data():
    """Sync Redis data to Google Drive"""
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Create periodic snapshot
    try:
        r.bgsave()
        print(f"✅ Redis snapshot created at {datetime.now()}")

        # Update sync metadata
        sync_info = {
            "machine_id": os.environ.get("MACHINE_ID"),
            "last_sync": datetime.now().isoformat(),
            "redis_info": r.info()
        }

        with open("/data/sync_metadata.json", "w") as f:
            json.dump(sync_info, f, indent=2)

    except Exception as e:
        print(f"❌ Sync error: {e}")

if __name__ == "__main__":
    while True:
        sync_redis_data()
        time.sleep(300)  # Sync every 5 minutes
EOF

# Start sync manager in background
echo "🔄 Starting sync manager..."
python /tmp/sync_manager.py &
SYNC_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down NFL Analytics system..."
    kill $SYNC_PID 2>/dev/null || true
    redis-cli SHUTDOWN SAVE 2>/dev/null || true
    echo "👋 Shutdown complete"
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the main compute worker
echo "🚀 Starting NFL Analytics compute worker..."
echo "   Machine ID: $MACHINE_ID"
echo "   Hardware: $HARDWARE_PROFILE"
echo "   Redis: $(redis-cli ping)"

cd /app

# Start the compute system with Redis backend
python run_compute.py --redis-mode --machine-id "$MACHINE_ID" --hardware-profile "$HARDWARE_PROFILE"

# If we get here, the main process exited
cleanup