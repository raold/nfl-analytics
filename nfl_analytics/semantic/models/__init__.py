"""
LLM Models for semantic analysis.

Contains fine-tuned models for NFL-specific NLP tasks:
    - NER model (Mistral-7B with LoRA via Ollama)
    - Sentiment model (Mistral or RoBERTa)
    - Embedding model (nomic-embed-text via Ollama)

Supports multiple backends:
    1. Ollama (primary - local GPU, fast, free)
    2. LM Studio (alternative local option)
    3. Gemini API (fallback)
    4. Claude API (complex reasoning)
"""

from .config import ModelConfig, get_model_config, MODELS, API_CONFIG
from .ollama_client import OllamaClient, get_ollama_status, ensure_models_available
from .inference import SemanticInferenceClient, InferenceBackend, InferenceResult

# MLX is optional (Apple Silicon only)
try:
    from .mlx_client import MLXClient, get_mlx_status, check_mlx_available

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    MLXClient = None
    get_mlx_status = lambda: "MLX not installed"
    check_mlx_available = lambda: False

__all__ = [
    "ModelConfig",
    "get_model_config",
    "MODELS",
    "API_CONFIG",
    "OllamaClient",
    "get_ollama_status",
    "ensure_models_available",
    "SemanticInferenceClient",
    "InferenceBackend",
    "InferenceResult",
    "MLXClient",
    "get_mlx_status",
    "check_mlx_available",
    "MLX_AVAILABLE",
]
