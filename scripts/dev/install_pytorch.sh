#!/usr/bin/env bash
# Install PyTorch with appropriate backend support
# Usage: bash scripts/install_pytorch.sh [mps|cuda|cpu]

set -e

BACKEND="${1:-auto}"

detect_backend() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo "mps"
        else
            echo "cpu"
        fi
    elif command -v nvidia-smi &> /dev/null; then
        # CUDA available
        echo "cuda"
    else
        echo "cpu"
    fi
}

if [[ "$BACKEND" == "auto" ]]; then
    BACKEND=$(detect_backend)
    echo "Auto-detected backend: $BACKEND"
fi

case "$BACKEND" in
    mps)
        echo "Installing PyTorch with MPS (Apple Silicon) support..."
        .venv/bin/pip install torch torchvision torchaudio
        ;;
    cuda)
        echo "Installing PyTorch with CUDA 12.1 support..."
        .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    cpu)
        echo "Installing PyTorch (CPU only)..."
        .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "Error: Unknown backend '$BACKEND'. Use: mps, cuda, or cpu"
        exit 1
        ;;
esac

echo ""
echo "PyTorch installation complete. Verifying..."
.venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else 'MPS: N/A')"
