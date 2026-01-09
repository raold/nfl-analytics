"""
Model configuration for local GPU inference.

Optimized for RTX 3090/4090 (24GB VRAM).
Uses 4-bit quantization for Mistral-7B to fit alongside other models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Base paths
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "semantic"
CACHE_DIR = Path.home() / ".cache" / "nfl_semantic"


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    model_id: str  # HuggingFace model ID or local path
    task: Literal["ner", "sentiment", "embedding", "generation"]
    quantization: Literal["none", "4bit", "8bit"] = "none"
    max_length: int = 512
    device: str = "cuda"
    dtype: str = "float16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    def __post_init__(self):
        if self.quantization == "4bit":
            self.load_in_4bit = True
        elif self.quantization == "8bit":
            self.load_in_8bit = True


@dataclass
class InferenceConfig:
    """Configuration for inference server."""

    host: str = "127.0.0.1"
    port: int = 8100
    max_batch_size: int = 8
    timeout_seconds: float = 30.0
    use_vllm: bool = True  # Use vLLM for generation models
    use_redis_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes


# Pre-configured models for NFL semantic stack
MODELS = {
    # NER Model - Fine-tuned Mistral for entity extraction
    "ner": ModelConfig(
        name="nfl-ner",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",  # Base, will be LoRA fine-tuned
        task="ner",
        quantization="4bit",  # ~7GB VRAM
        max_length=1024,
    ),
    # Sentiment Model - Fine-tuned RoBERTa
    "sentiment": ModelConfig(
        name="nfl-sentiment",
        model_id="roberta-large",  # Base, will be fine-tuned
        task="sentiment",
        quantization="none",  # ~1.4GB VRAM
        max_length=512,
    ),
    # Embedding Model - For RAG similarity search
    "embedding": ModelConfig(
        name="nfl-embedding",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        task="embedding",
        quantization="none",  # <500MB
        max_length=256,
    ),
    # Generation Model - For summaries and expert reasoning
    "generation": ModelConfig(
        name="nfl-generation",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="generation",
        quantization="4bit",  # Shared with NER if using same base
        max_length=2048,
    ),
}

# Total VRAM estimate: ~9GB (leaving room for batching on 24GB GPU)


# API fallback configuration (when local models unavailable or for validation)
@dataclass
class APIConfig:
    """Configuration for external API fallbacks."""

    gemini_model: str = "gemini-2.5-flash"
    claude_model: str = "claude-3-haiku-20240307"
    max_monthly_budget_usd: float = 100.0
    rate_limit_rpm: int = 60  # Requests per minute


API_CONFIG = APIConfig()


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]


def get_vram_estimate() -> dict[str, float]:
    """Estimate VRAM usage for all models in GB."""
    estimates = {
        "ner": 7.0,  # 4-bit Mistral-7B
        "sentiment": 1.4,  # RoBERTa-large fp16
        "embedding": 0.5,  # MiniLM
        "generation": 0.0,  # Shared with NER
    }
    estimates["total"] = sum(estimates.values())
    estimates["available_for_batching"] = 24.0 - estimates["total"]
    return estimates
