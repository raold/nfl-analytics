"""
MLX client for Apple Silicon optimized inference.

MLX is Apple's machine learning framework optimized for M-series chips.
It provides better performance and memory efficiency than Ollama on Mac.

Setup:
    pip install mlx-lm

Usage:
    client = MLXClient()
    response = client.generate("Extract entities from: Patrick Mahomes injured")

Recommended Models (from mlx-community on HuggingFace):
    - mlx-community/Qwen2.5-7B-Instruct-4bit  (best for NER/sentiment)
    - mlx-community/Mistral-7B-Instruct-v0.3-4bit
    - mlx-community/Llama-3.2-3B-Instruct-4bit (smaller, faster)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Default MLX models from HuggingFace mlx-community
RECOMMENDED_MLX_MODELS = {
    "ner": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "sentiment": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "generation": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "fast": "mlx-community/Llama-3.2-3B-Instruct-4bit",
}


@dataclass
class MLXResponse:
    """Response from MLX generation."""

    text: str
    model: str
    tokens_generated: int
    tokens_per_second: float
    total_time_ms: float


def check_mlx_available() -> bool:
    """Check if MLX is available on this system."""
    try:
        import mlx.core as mx

        # Check for Apple Silicon
        return mx.metal.is_available()
    except ImportError:
        return False
    except Exception:
        return False


class MLXClient:
    """
    Client for MLX-based LLM inference on Apple Silicon.

    Provides similar interface to OllamaClient for easy switching.
    """

    def __init__(
        self,
        default_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        max_tokens: int = 1024,
    ):
        """
        Initialize MLX client.

        Args:
            default_model: HuggingFace model ID (mlx-community models recommended)
            max_tokens: Default max tokens for generation
        """
        self.default_model = default_model
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._loaded_model_name = None

        # Check MLX availability
        if not check_mlx_available():
            logger.warning("MLX not available - requires Apple Silicon Mac with macOS 14+")

    def is_available(self) -> bool:
        """Check if MLX is available."""
        return check_mlx_available()

    def _load_model(self, model: str) -> tuple[Any, Any]:
        """
        Load model and tokenizer, with caching.

        Args:
            model: HuggingFace model ID

        Returns:
            Tuple of (model, tokenizer)
        """
        if self._loaded_model_name == model and self._model is not None:
            return self._model, self._tokenizer

        try:
            from mlx_lm import load

            logger.info(f"Loading MLX model: {model}")
            start = time.time()

            self._model, self._tokenizer = load(model)
            self._loaded_model_name = model

            load_time = time.time() - start
            logger.info(f"Model loaded in {load_time:.1f}s")

            return self._model, self._tokenizer

        except ImportError:
            raise ImportError(
                "MLX LM not installed. Install with: pip install mlx-lm"
            )

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> MLXResponse:
        """
        Generate text using MLX.

        Args:
            prompt: Input prompt
            model: Model to use (default: Qwen2.5-7B-Instruct-4bit)
            system: System prompt (will be prepended)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            MLXResponse with generated text and metrics
        """
        if not self.is_available():
            raise RuntimeError("MLX not available on this system")

        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        try:
            from mlx_lm import generate

            # Load model
            mlx_model, tokenizer = self._load_model(model)

            # Build prompt with system message if provided
            if system:
                # Format for instruction-tuned models
                full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                full_prompt = prompt

            # Generate
            start = time.time()

            response = generate(
                mlx_model,
                tokenizer,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

            elapsed = time.time() - start

            # Count tokens (approximate)
            tokens_generated = len(tokenizer.encode(response))
            tps = tokens_generated / elapsed if elapsed > 0 else 0

            return MLXResponse(
                text=response.strip(),
                model=model,
                tokens_generated=tokens_generated,
                tokens_per_second=tps,
                total_time_ms=elapsed * 1000,
            )

        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream text generation using MLX.

        Args:
            prompt: Input prompt
            model: Model to use
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Yields:
            Generated text tokens
        """
        if not self.is_available():
            raise RuntimeError("MLX not available on this system")

        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens

        try:
            from mlx_lm import generate_step

            mlx_model, tokenizer = self._load_model(model)

            # Build prompt
            if system:
                full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                full_prompt = prompt

            # Tokenize
            input_ids = tokenizer.encode(full_prompt, return_tensors="mlx")

            # Generate with streaming
            for token in generate_step(mlx_model, input_ids, temp=temperature):
                text = tokenizer.decode([token])
                yield text

        except ImportError:
            raise ImportError("MLX LM streaming requires mlx-lm >= 0.19")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
    ) -> MLXResponse:
        """
        Chat completion using MLX.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            model: Model to use
            temperature: Sampling temperature

        Returns:
            MLXResponse
        """
        # Build prompt from messages
        prompt_parts = []
        system_msg = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_msg = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        return self.generate(
            prompt=prompt,
            model=model,
            system=system_msg,
            temperature=temperature,
        )

    def get_status(self) -> dict[str, Any]:
        """Get MLX status information."""
        status = {
            "available": self.is_available(),
            "loaded_model": self._loaded_model_name,
            "default_model": self.default_model,
        }

        if self.is_available():
            try:
                import mlx.core as mx

                status["metal_available"] = mx.metal.is_available()
                status["device"] = "Apple Silicon (Metal)"
            except Exception:
                pass

        return status


def get_mlx_status() -> str:
    """Get human-readable MLX status."""
    if not check_mlx_available():
        return "MLX: NOT AVAILABLE\nRequires Apple Silicon Mac with macOS 14+"

    try:
        import mlx.core as mx

        lines = [
            "MLX Status: AVAILABLE",
            f"Metal GPU: {mx.metal.is_available()}",
            "-" * 30,
            "Recommended models:",
        ]

        for task, model in RECOMMENDED_MLX_MODELS.items():
            lines.append(f"  {task}: {model}")

        lines.append("")
        lines.append("Download with: mlx_lm.convert --hf-path <model>")

        return "\n".join(lines)

    except Exception as e:
        return f"MLX: ERROR - {e}"


def download_model(model_id: str) -> bool:
    """
    Download and convert a model for MLX.

    Args:
        model_id: HuggingFace model ID

    Returns:
        True if successful
    """
    try:
        from mlx_lm import load

        logger.info(f"Downloading model: {model_id}")
        # Loading will automatically download
        load(model_id)
        logger.info(f"Model ready: {model_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


# Quick test
if __name__ == "__main__":
    print(get_mlx_status())

    if check_mlx_available():
        client = MLXClient()
        response = client.generate(
            "What is 2+2?",
            max_tokens=50,
        )
        print(f"\nResponse: {response.text}")
        print(f"Speed: {response.tokens_per_second:.1f} tok/s")
