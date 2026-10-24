"""
Ollama client for local LLM inference.

Ollama provides easy-to-use local model serving with:
- Simple REST API
- Automatic model management
- Optimized inference for consumer GPUs
- Support for Mistral, Llama, and other models

Usage:
    # Ensure Ollama is running: ollama serve
    # Pull required model: ollama pull mistral:7b-instruct-v0.2-q4_K_M

    client = OllamaClient()
    response = client.generate("Extract entities from: Patrick Mahomes injured his ankle")
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)

# Default Ollama endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

# Recommended models for NFL semantic tasks (Dec 2025)
RECOMMENDED_MODELS = {
    "ner": "qwen3:8b",  # Best for structured output/NER
    "generation": "qwen3:8b",
    "sentiment": "qwen3:8b",
    "reasoning": "qwen3:8b",  # For complex reasoning
}

# Fallback to smaller models if VRAM constrained
FALLBACK_MODELS = {
    "ner": "qwen2.5:3b-instruct",
    "generation": "phi4:latest",
    "sentiment": "qwen2.5:3b-instruct",
}


@dataclass
class OllamaResponse:
    """Response from Ollama API."""

    model: str
    response: str
    done: bool
    total_duration_ns: int | None = None
    eval_count: int | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if self.total_duration_ns and self.eval_count:
            return self.eval_count / (self.total_duration_ns / 1e9)
        return None


class OllamaClient:
    """Client for Ollama local LLM inference."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 60.0,
        default_model: str = "qwen3:8b",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model
        self._client = httpx.Client(timeout=timeout)

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.ConnectError:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        logger.info(f"Pulling model: {model}")
        try:
            response = self._client.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=600.0,  # 10 min for large models
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stream: bool = False,
        use_chat: bool = True,  # Use chat endpoint for instruct models
    ) -> OllamaResponse | Iterator[OllamaResponse]:
        """
        Generate text using Ollama.

        Args:
            prompt: The input prompt
            model: Model to use (default: qwen3:8b)
            system: System prompt for instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses
            use_chat: Use chat endpoint (required for instruct models like qwen3)

        Returns:
            OllamaResponse or iterator of responses if streaming
        """
        model = model or self.default_model

        # Use chat endpoint for instruction-tuned models (qwen3, etc.)
        if use_chat:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.chat(messages, model=model, temperature=temperature)

        # Legacy generate endpoint (for base models)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        if stream:
            return self._generate_stream(payload)
        else:
            return self._generate_sync(payload)

    def _generate_sync(self, payload: dict[str, Any]) -> OllamaResponse:
        """Synchronous generation."""
        response = self._client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return OllamaResponse(
            model=data.get("model", ""),
            response=data.get("response", ""),
            done=data.get("done", True),
            total_duration_ns=data.get("total_duration"),
            eval_count=data.get("eval_count"),
        )

    def _generate_stream(self, payload: dict[str, Any]) -> Iterator[OllamaResponse]:
        """Streaming generation."""
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    yield OllamaResponse(
                        model=data.get("model", ""),
                        response=data.get("response", ""),
                        done=data.get("done", False),
                        total_duration_ns=data.get("total_duration"),
                        eval_count=data.get("eval_count"),
                    )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
    ) -> OllamaResponse:
        """
        Chat completion using Ollama.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            model: Model to use
            temperature: Sampling temperature

        Returns:
            OllamaResponse
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return OllamaResponse(
            model=data.get("model", ""),
            response=data.get("message", {}).get("content", ""),
            done=True,
            total_duration_ns=data.get("total_duration"),
            eval_count=data.get("eval_count"),
        )

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        """
        Generate embeddings using Ollama.

        Args:
            text: Text to embed
            model: Embedding model (default: nomic-embed-text)

        Returns:
            List of floats (embedding vector)
        """
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def ensure_models_available(client: OllamaClient | None = None) -> dict[str, bool]:
    """
    Ensure required models are available, pulling if needed.

    Returns:
        Dict of model -> available status
    """
    if client is None:
        client = OllamaClient()

    if not client.is_available():
        logger.error("Ollama server not running. Start with: ollama serve")
        return {m: False for m in RECOMMENDED_MODELS.values()}

    available = client.list_models()
    status = {}

    for task, model in RECOMMENDED_MODELS.items():
        if model in available:
            status[model] = True
            logger.info(f" {task}: {model} available")
        else:
            logger.info(f" {task}: {model} not found, pulling...")
            status[model] = client.pull_model(model)

    return status


def get_ollama_status() -> str:
    """Get human-readable Ollama status."""
    client = OllamaClient()

    if not client.is_available():
        return "Ollama: NOT RUNNING\nStart with: ollama serve"

    models = client.list_models()
    lines = [
        "Ollama Status: RUNNING",
        f"Available models: {len(models)}",
        "-" * 30,
    ]

    for model in models[:10]:  # Show first 10
        lines.append(f"  - {model}")

    if len(models) > 10:
        lines.append(f"  ... and {len(models) - 10} more")

    return "\n".join(lines)
