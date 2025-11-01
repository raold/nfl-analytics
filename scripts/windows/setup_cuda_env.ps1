# PowerShell script to set up CUDA environment on Windows 11 with RTX 4090
# Run as: .\scripts\windows\setup_cuda_env.ps1

Write-Host "🚀 Setting up CUDA environment for RTX 4090" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check if running as Administrator (optional but recommended)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "⚠️  Not running as Administrator. Some features may be limited." -ForegroundColor Yellow
}

# Check NVIDIA Driver
Write-Host "`n📊 Checking NVIDIA Driver..." -ForegroundColor Green
try {
    $nvidiaInfo = nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    Write-Host "✅ GPU Detected: $nvidiaInfo" -ForegroundColor Green
} catch {
    Write-Host "❌ nvidia-smi not found! Please install NVIDIA drivers" -ForegroundColor Red
    Write-Host "   Download from: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    exit 1
}

# Check Python installation
Write-Host "`n🐍 Checking Python installation..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.10+" -ForegroundColor Red
    Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check CUDA availability in PyTorch
Write-Host "`n🔥 Checking PyTorch CUDA support..." -ForegroundColor Green
$cudaCheck = @"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available!')
    exit(1)
"@

python -c $cudaCheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ CUDA not available in PyTorch!" -ForegroundColor Red
    Write-Host "   Install PyTorch with CUDA:" -ForegroundColor Yellow
    Write-Host "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Cyan
    exit 1
}

# Check and install uv if needed
Write-Host "`n📦 Checking uv installation..." -ForegroundColor Green
try {
    $uvVersion = uv --version 2>&1
    Write-Host "✅ uv found: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  uv not found. Installing..." -ForegroundColor Yellow
    pip install uv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ uv installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install uv" -ForegroundColor Red
    }
}

# Sync project dependencies
Write-Host "`n📚 Syncing project dependencies..." -ForegroundColor Green
if (Test-Path "pyproject.toml") {
    uv sync
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies synced" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  No pyproject.toml found, using requirements.txt" -ForegroundColor Yellow
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    }
}

# Create necessary directories
Write-Host "`n📁 Creating project directories..." -ForegroundColor Green
$directories = @(
    "models/bayesian",
    "models/gnn/ablation",
    "models/cvae/checkpoints",
    "models/rl/dqn_checkpoints",
    "models/rl/ppo_checkpoints",
    "experiments/optimization",
    "experiments/evaluation",
    "reports/backtest",
    "logs/training"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   Created: $dir" -ForegroundColor DarkGray
    }
}

# Set environment variables for optimal GPU performance
Write-Host "`n⚙️  Setting environment variables..." -ForegroundColor Green
[System.Environment]::SetEnvironmentVariable("CUDA_LAUNCH_BLOCKING", "0", "Process")
[System.Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512", "Process")
[System.Environment]::SetEnvironmentVariable("TF_FORCE_GPU_ALLOW_GROWTH", "true", "Process")

# Display GPU information
Write-Host "`n🎮 GPU Information:" -ForegroundColor Cyan
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version,compute_cap --format=table

# Test a simple CUDA operation
Write-Host "`n🧪 Testing CUDA with tensor operations..." -ForegroundColor Green
$testCuda = @"
import torch
import time

# Create large tensors on GPU
size = 10000
a = torch.randn(size, size, device='cuda')
b = torch.randn(size, size, device='cuda')

# Warm up
_ = torch.matmul(a, b)
torch.cuda.synchronize()

# Benchmark
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f'Matrix multiplication ({size}x{size}): {elapsed*1000:.2f}ms')
print(f'TFLOPS: {2*size**3 / elapsed / 1e12:.2f}')
"@

python -c $testCuda

# Summary
Write-Host "`n" -NoNewline
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ CUDA Environment Setup Complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ready to train on RTX 4090!" -ForegroundColor Green
Write-Host ""
Write-Host "Quick Start Commands:" -ForegroundColor Yellow
Write-Host "  python scripts\gpu_training_orchestrator.py --phase 1" -ForegroundColor White
Write-Host "  python scripts\gpu_monitor.py" -ForegroundColor White
Write-Host ""