"""
Model loader for semantic stack.

Handles loading models with:
- 4-bit/8-bit quantization for VRAM efficiency
- Model caching to avoid reloading
- Graceful fallback to CPU if GPU unavailable
"""

import logging
from functools import lru_cache
from typing import Any

import torch

from .config import ModelConfig, get_model_config, CACHE_DIR

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {device_name} ({vram_gb:.1f}GB VRAM)")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS")
        return "mps"
    else:
        logger.warning("No GPU available, using CPU (will be slow)")
        return "cpu"


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str = "embedding") -> Any:
    """
    Load sentence transformer model for embeddings.

    Returns:
        SentenceTransformer model
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    config = get_model_config(model_name)
    device = get_device()

    logger.info(f"Loading embedding model: {config.model_id}")
    model = SentenceTransformer(
        config.model_id,
        device=device,
        cache_folder=str(CACHE_DIR),
    )
    return model


@lru_cache(maxsize=2)
def load_sentiment_model(model_name: str = "sentiment") -> tuple[Any, Any]:
    """
    Load RoBERTa model for sentiment analysis.

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    config = get_model_config(model_name)
    device = get_device()

    logger.info(f"Loading sentiment model: {config.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        cache_dir=str(CACHE_DIR),
    )

    # Check for fine-tuned model first
    fine_tuned_path = CACHE_DIR / "fine_tuned" / "sentiment"
    if fine_tuned_path.exists():
        logger.info(f"Loading fine-tuned model from {fine_tuned_path}")
        model_path = str(fine_tuned_path)
    else:
        logger.info("No fine-tuned model found, using base model")
        model_path = config.model_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        cache_dir=str(CACHE_DIR),
        num_labels=3,  # negative, neutral, positive
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    return model, tokenizer


@lru_cache(maxsize=1)
def load_generation_model(model_name: str = "generation") -> tuple[Any, Any]:
    """
    Load Mistral model for text generation (NER, summaries, reasoning).

    Uses 4-bit quantization to fit in VRAM alongside other models.

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        raise ImportError("Install transformers and bitsandbytes: pip install transformers bitsandbytes")

    config = get_model_config(model_name)
    device = get_device()

    logger.info(f"Loading generation model: {config.model_id} (4-bit quantized)")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        cache_dir=str(CACHE_DIR),
    )

    # 4-bit quantization config
    if config.load_in_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Check for LoRA fine-tuned adapter
    lora_path = CACHE_DIR / "fine_tuned" / "ner_lora"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        cache_dir=str(CACHE_DIR),
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
    )

    # Load LoRA adapter if available
    if lora_path.exists():
        try:
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter from {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path))
        except ImportError:
            logger.warning("peft not installed, skipping LoRA adapter")

    model.eval()
    return model, tokenizer


def check_model_availability() -> dict[str, bool]:
    """Check which models are available and loadable."""
    availability = {}

    # Check embedding model
    try:
        from sentence_transformers import SentenceTransformer

        availability["embedding"] = True
    except ImportError:
        availability["embedding"] = False

    # Check transformers
    try:
        from transformers import AutoModelForCausalLM

        availability["generation"] = True
        availability["sentiment"] = True
    except ImportError:
        availability["generation"] = False
        availability["sentiment"] = False

    # Check quantization support
    try:
        import bitsandbytes

        availability["4bit_quantization"] = True
    except ImportError:
        availability["4bit_quantization"] = False

    # Check GPU
    availability["cuda"] = torch.cuda.is_available()
    availability["mps"] = torch.backends.mps.is_available()

    return availability


def get_model_status() -> str:
    """Get human-readable model availability status."""
    avail = check_model_availability()
    lines = ["Model Availability Status:", "-" * 30]

    for key, available in avail.items():
        status = "OK" if available else "MISSING"
        lines.append(f"  {key}: {status}")

    return "\n".join(lines)
