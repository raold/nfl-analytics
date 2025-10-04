# NFL Analytics Docker + Redis Distributed Computing

This directory contains the Docker-based distributed computing infrastructure for the NFL Analytics system, featuring Redis task queues and Google Drive synchronization.

## Architecture Overview

```
┌─ Machine A (Desktop 4090) ─────────────────┐    ┌─ Google Drive Cloud ─┐
│ Docker Container                           │◄──►│                     │
│ ├─ Redis Server (localhost)                │    │ ├─ /data/redis/     │
│ ├─ NFL Analytics Worker                    │    │ ├─ /data/results/   │
│ ├─ Hardware Detection & Routing            │    │ ├─ /data/checkpoints/│
│ └─ Sync Manager (background)               │    │ └─ /data/logs/      │
└────────────────────────────────────────────┘    └─────────────────────┘
                                                            ▲
┌─ Machine B (Laptop M4) ────────────────────┐              │
│ Identical Container                        │◄─────────────┘
│ Different Hardware Profile & Tasks        │
└────────────────────────────────────────────┘
```

## Features

- **🫀 Redis-based Task Coordination**: Atomic task claiming, priority queues, hardware-aware routing
- **🖥️ Hardware-Aware Distribution**: GPU vs CPU task routing, capability detection
- **🔄 Google Drive Sync**: Automatic synchronization of Redis data and results
- **🐳 Docker Containerization**: Consistent environment across machines
- **📊 Production Monitoring**: Health checks, metrics, performance tracking
- **🔧 Backward Compatibility**: Fallback to SQLite for development

## Quick Start

### 1. Prerequisites

```bash
# Install Docker
# macOS: Docker Desktop
# Linux: sudo apt install docker.io docker-compose

# Install Redis (for development/testing)
# macOS: brew install redis
# Linux: sudo apt install redis-server

# Ensure Google Drive is synced to /data directory
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your settings
MACHINE_ID=auto                    # auto-detect or set manually
HARDWARE_PROFILE=auto              # auto-detect hardware
NFL_ANALYTICS_ENV=development      # or production
```

### 3. Build and Run

```bash
# Build container
docker-compose build

# Initialize Redis queue
docker-compose run nfl-analytics python run_compute.py --redis-mode --init

# Start distributed worker
docker-compose up nfl-analytics

# Or run in background
docker-compose up -d nfl-analytics
```

### 4. Monitor Progress

```bash
# Check status
docker-compose exec nfl-analytics python run_compute.py --redis-mode --status

# View logs
docker-compose logs -f nfl-analytics

# Performance dashboard
docker-compose exec nfl-analytics python run_compute.py --dashboard
```

## Usage Examples

### Desktop with RTX 4090
```bash
# Set environment for high-end GPU
echo "MACHINE_ID=desktop-4090" >> .env
echo "HARDWARE_PROFILE=gpu_high" >> .env

# Start worker
docker-compose up nfl-analytics
```

### Laptop with M4 CPU
```bash
# Set environment for Apple Silicon
echo "MACHINE_ID=laptop-m4" >> .env
echo "HARDWARE_PROFILE=cpu_arm_high" >> .env

# Start worker
docker-compose up nfl-analytics
```

### Development Mode
```bash
# Use SQLite backend for development
docker-compose run nfl-analytics python run_compute.py --init
docker-compose run nfl-analytics python run_compute.py
```

## Task Queue Types

The system automatically routes tasks based on hardware capabilities:

- **`gpu:high`**: RTX 4090, high-end GPUs (>16GB VRAM)
- **`gpu:standard`**: Standard GPUs (8-16GB VRAM)
- **`cpu:parallel`**: Multi-core CPU tasks (feature engineering)
- **`cpu:sequential`**: Single-thread tasks (analysis)
- **`data:pipeline`**: ETL and data ingestion
- **`analysis:standard`**: Research and reporting

## Google Drive Sync Strategy

### Directory Structure
```
/data/                          # Google Drive synced directory
├── redis/                      # Redis persistence files
│   ├── dump.rdb               # Redis snapshot
│   └── appendonly.aof         # Redis append-only file
├── machine_dumps/              # Machine-specific snapshots
│   ├── desktop-4090_*.rdb     # Desktop snapshots
│   └── laptop-m4_*.rdb        # Laptop snapshots
├── results/                    # Task results
│   └── year=2024/month=01/     # Partitioned by time
├── checkpoints/                # Model checkpoints
└── logs/                       # System logs
```

### Sync Process
1. **Every 5 minutes**: Redis BGSAVE creates atomic snapshot
2. **Conflict Detection**: Multiple machine snapshots checked
3. **Conflict Resolution**: Automatic merging with timestamp priority
4. **Sync Metadata**: Machine heartbeats and status tracking

## Monitoring & Debugging

### Health Checks
```bash
# Container health
docker-compose ps

# Redis status
docker-compose exec nfl-analytics redis-cli ping

# Queue status
docker-compose exec nfl-analytics python run_compute.py --redis-mode --status

# Worker logs
docker-compose logs nfl-analytics | grep "Task"
```

### Performance Monitoring
```bash
# Redis memory usage
docker-compose exec nfl-analytics redis-cli info memory

# Task throughput
docker-compose exec nfl-analytics python -c "
from redis_task_queue import RedisTaskQueue
queue = RedisTaskQueue()
print(queue.get_queue_status())
"

# Machine capabilities
docker-compose exec nfl-analytics python -c "
from run_compute import detect_hardware_profile
profile = detect_hardware_profile()
print(f'GPU: {profile.gpu_name} ({profile.gpu_memory/(1024**3):.1f}GB)')
"
```

### Troubleshooting

**Redis Connection Issues:**
```bash
# Check Redis is running
docker-compose exec nfl-analytics redis-cli ping

# Restart Redis service
docker-compose restart nfl-analytics
```

**Sync Conflicts:**
```bash
# Check sync metadata
cat /data/sync_metadata.json

# View machine dumps
ls -la /data/machine_dumps/

# Force sync reset
rm /data/machine_dumps/* && docker-compose restart nfl-analytics
```

**Task Processing Issues:**
```bash
# Check task queues
docker-compose exec nfl-analytics redis-cli ZRANGE gpu:high 0 -1 WITHSCORES

# View failed tasks
docker-compose exec nfl-analytics redis-cli SMEMBERS tasks_by_status:failed

# Reset queues (careful!)
docker-compose exec nfl-analytics redis-cli FLUSHALL
```

## Configuration Options

### Environment Variables
```bash
# Core settings
MACHINE_ID=auto                    # Machine identifier
HARDWARE_PROFILE=auto              # Hardware capabilities
NFL_ANALYTICS_ENV=development      # Environment

# Redis settings
REDIS_HOST=localhost               # Redis server
REDIS_PORT=6379                    # Redis port
REDIS_MAX_MEMORY=8gb              # Memory limit

# Sync settings
GOOGLE_DRIVE_SYNC_INTERVAL=300    # Sync every 5 minutes
SYNC_ENABLED=true                 # Enable Google Drive sync

# Performance tuning
MAX_CONCURRENT_TASKS=4            # Parallel tasks
TASK_TIMEOUT_HOURS=6              # Task timeout
ENABLE_CACHING=true               # Result caching
```

### Hardware Profiles
- **`gpu_high`**: RTX 4090, A100 (>16GB VRAM)
- **`gpu_standard`**: RTX 3080, 4070 (8-16GB VRAM)
- **`cpu_arm_high`**: Apple M1/M2/M3/M4 (8+ cores)
- **`cpu_x86`**: Intel/AMD (4+ cores)

## Development

### Testing
```bash
# Test Redis infrastructure
cd containers
python test_redis_setup.py

# Test specific components
python -m pytest ../py/compute/test_redis_task_queue.py
```

### Adding New Task Types
```python
# In redis_task_queue.py
def _determine_queue(self, task: TaskDefinition) -> str:
    if task.task_type == "new_task_type":
        return QueueType.CPU_PARALLEL.value
    # ... existing logic
```

### Custom Hardware Detection
```python
# In redis_worker.py
def _detect_capabilities(self) -> List[str]:
    capabilities = []
    # Add custom capability detection
    return capabilities
```

## Production Deployment

### Multi-Machine Setup
1. **Shared Google Drive**: Ensure `/data` is synced across all machines
2. **Unique Machine IDs**: Set distinct `MACHINE_ID` for each machine
3. **Hardware Profiles**: Configure appropriate profiles for each machine
4. **Monitoring**: Set up centralized logging and alerting

### Security Considerations
- Redis runs on localhost only (not exposed)
- Google Drive provides access control
- Container isolation limits attack surface
- Secrets managed via environment variables

### Scaling Guidelines
- **2-3 machines**: Current architecture works perfectly
- **5+ machines**: Consider Redis clustering
- **10+ machines**: Move to managed Redis service
- **Production**: Add monitoring, alerting, backup strategies

## Support

For issues or questions:
1. Check logs: `docker-compose logs nfl-analytics`
2. Test infrastructure: `python test_redis_setup.py`
3. Verify Google Drive sync is working
4. Check hardware detection is correct

The system is designed to be resilient and self-healing, but proper monitoring ensures optimal performance.