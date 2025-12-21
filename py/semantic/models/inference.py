"""
Unified inference client for semantic stack.

Provides a single interface that:
1. Tries Ollama first (local GPU, fast, free)
2. Falls back to Gemini API if Ollama unavailable
3. Uses Claude API for complex reasoning validation

Usage:
    client = SemanticInferenceClient()

    # NER extraction
    entities = client.extract_entities("Patrick Mahomes questionable with ankle injury")

    # Sentiment analysis
    sentiment = client.analyze_sentiment("Chiefs looking strong heading into playoffs")

    # Expert reasoning
    analysis = client.expert_reasoning("Given the injury report...", context=rag_docs)
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .ollama_client import OllamaClient, RECOMMENDED_MODELS

logger = logging.getLogger(__name__)


class InferenceBackend(Enum):
    """Available inference backends."""

    MLX = "mlx"  # Apple Silicon optimized (fastest on Mac)
    OLLAMA = "ollama"  # Cross-platform local
    GEMINI = "gemini"  # API fallback
    CLAUDE = "claude"  # API fallback (complex reasoning)


@dataclass
class InferenceResult:
    """Result from any inference backend."""

    text: str
    backend: InferenceBackend
    model: str
    tokens_used: int | None = None
    latency_ms: float | None = None
    cached: bool = False


# System prompts for different tasks
SYSTEM_PROMPTS = {
    "ner": """You are an NFL entity extraction expert. Extract entities from the text in JSON format.
Categories: PLAYER, TEAM, INJURY, DATE, GAME, COACH, POSITION
Output format: {"entities": [{"text": "...", "type": "...", "confidence": 0.0-1.0}]}
Be precise and only extract clearly mentioned entities.""",
    "sentiment": """You are an NFL sentiment analyst. Analyze the sentiment of news about NFL teams and players.
Output format: {"sentiment": "positive|neutral|negative", "confidence": 0.0-1.0, "reasoning": "brief explanation"}
Consider: injuries, performance, team dynamics, public perception.""",
    "summary": """You are an NFL news summarizer. Provide concise, factual summaries focused on betting-relevant information.
Focus on: injuries, lineup changes, coaching decisions, weather, motivation factors.
Be objective and stick to facts.""",
    "reasoning": """You are an expert NFL analyst helping with sports betting decisions.
Analyze the provided information and give actionable insights.
Consider: historical patterns, matchup dynamics, market inefficiencies.
Be specific and quantify your confidence levels.""",
}


class SemanticInferenceClient:
    """
    Unified inference client with fallback chain.

    Priority (on Mac):
    1. MLX (Apple Silicon optimized, fastest)
    2. Ollama (cross-platform local)
    3. Gemini API (fast, cheap)
    4. Claude API (best quality, more expensive)

    Priority (on other platforms):
    1. Ollama (local, fast, free)
    2. Gemini API
    3. Claude API
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_timeout: float = 240.0,  # 4 min timeout for qwen3 models
        prefer_local: bool = True,
        prefer_mlx: bool = True,  # Prefer MLX on Mac
        gemini_api_key: str | None = None,
    ):
        self.prefer_local = prefer_local
        self.prefer_mlx = prefer_mlx
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        # Initialize MLX client (Apple Silicon only)
        self._mlx = None
        self._mlx_available = False
        if prefer_mlx:
            try:
                from .mlx_client import MLXClient, check_mlx_available

                if check_mlx_available():
                    self._mlx = MLXClient()
                    self._mlx_available = True
                    logger.info("MLX backend: AVAILABLE (Apple Silicon optimized)")
            except ImportError:
                logger.info("MLX not installed - pip install mlx-lm")

        # Initialize Ollama client with longer timeout for qwen3
        self._ollama = OllamaClient(base_url=ollama_url, timeout=ollama_timeout)
        self._ollama_available = self._ollama.is_available()

        if self._ollama_available:
            logger.info("Ollama backend: AVAILABLE")
        elif not self._mlx_available:
            logger.warning("No local backends available - will use API fallbacks")

    @property
    def primary_backend(self) -> InferenceBackend:
        """Get the current primary backend."""
        if self.prefer_local:
            if self._mlx_available and self.prefer_mlx:
                return InferenceBackend.MLX
            if self._ollama_available:
                return InferenceBackend.OLLAMA
        if self.gemini_api_key:
            return InferenceBackend.GEMINI
        return InferenceBackend.CLAUDE

    def extract_entities(self, text: str) -> dict[str, Any]:
        """
        Extract NFL entities from text.

        Args:
            text: News text, injury report, etc.

        Returns:
            Dict with entities list
        """
        prompt = f"Extract entities from the following NFL text:\n\n{text}"
        result = self._generate(
            prompt=prompt,
            system=SYSTEM_PROMPTS["ner"],
            task="ner",
            temperature=0.1,
        )

        # Parse JSON response
        try:
            import json

            # Find JSON in response
            response = result.text
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception as e:
            logger.warning(f"Failed to parse NER response: {e}")

        return {"entities": [], "raw_response": result.text}

    def analyze_sentiment(self, text: str) -> dict[str, Any]:
        """
        Analyze sentiment of NFL-related text.

        Args:
            text: News article, tweet, etc.

        Returns:
            Dict with sentiment, confidence, reasoning
        """
        prompt = f"Analyze the sentiment of this NFL-related text:\n\n{text}"
        result = self._generate(
            prompt=prompt,
            system=SYSTEM_PROMPTS["sentiment"],
            task="sentiment",
            temperature=0.1,
        )

        # Parse JSON response
        try:
            import json

            response = result.text
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception as e:
            logger.warning(f"Failed to parse sentiment response: {e}")

        return {"sentiment": "neutral", "confidence": 0.5, "raw_response": result.text}

    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize NFL news for betting relevance.

        Args:
            text: Full article or report
            max_length: Target summary length

        Returns:
            Concise summary string
        """
        prompt = f"Summarize this NFL news in {max_length} words or less, focusing on betting-relevant factors:\n\n{text}"
        result = self._generate(
            prompt=prompt,
            system=SYSTEM_PROMPTS["summary"],
            task="generation",
            temperature=0.3,
        )
        return result.text.strip()

    def expert_reasoning(
        self,
        question: str,
        context: list[str] | None = None,
        use_best_model: bool = True,
    ) -> dict[str, Any]:
        """
        Get expert-level reasoning about a betting scenario.

        Args:
            question: The analysis question
            context: RAG-retrieved context documents
            use_best_model: Use highest quality model (may use API)

        Returns:
            Dict with analysis, confidence, factors
        """
        # Build context-aware prompt
        if context:
            context_str = "\n\n".join([f"[Context {i + 1}]: {c}" for i, c in enumerate(context)])
            prompt = f"Based on the following context:\n\n{context_str}\n\nQuestion: {question}"
        else:
            prompt = question

        # For expert reasoning, may want to use API for best quality
        if use_best_model and self.gemini_api_key:
            result = self._generate_gemini(
                prompt=prompt,
                system=SYSTEM_PROMPTS["reasoning"],
                temperature=0.3,
            )
        else:
            result = self._generate(
                prompt=prompt,
                system=SYSTEM_PROMPTS["reasoning"],
                task="reasoning",
                temperature=0.3,
            )

        return {
            "analysis": result.text,
            "backend": result.backend.value,
            "model": result.model,
        }

    def _generate(
        self,
        prompt: str,
        system: str,
        task: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> InferenceResult:
        """Internal generation with backend selection."""
        import time

        start = time.time()

        # Try MLX first (fastest on Mac)
        if self._mlx_available and self.prefer_mlx:
            try:
                from .mlx_client import RECOMMENDED_MLX_MODELS

                model = RECOMMENDED_MLX_MODELS.get(task, "mlx-community/Qwen2.5-7B-Instruct-4bit")
                response = self._mlx.generate(
                    prompt=prompt,
                    model=model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return InferenceResult(
                    text=response.text,
                    backend=InferenceBackend.MLX,
                    model=model,
                    tokens_used=response.tokens_generated,
                    latency_ms=response.total_time_ms,
                )
            except Exception as e:
                logger.warning(f"MLX failed, trying Ollama: {e}")

        # Try Ollama
        if self._ollama_available:
            try:
                model = RECOMMENDED_MODELS.get(task, "qwen2.5:7b-instruct")
                response = self._ollama.generate(
                    prompt=prompt,
                    model=model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = (time.time() - start) * 1000

                return InferenceResult(
                    text=response.response,
                    backend=InferenceBackend.OLLAMA,
                    model=model,
                    tokens_used=response.eval_count,
                    latency_ms=latency,
                )
            except Exception as e:
                logger.warning(f"Ollama failed, falling back to API: {e}")

        # Fallback to Gemini
        if self.gemini_api_key:
            return self._generate_gemini(prompt, system, temperature)

        raise RuntimeError("No inference backend available")

    def _generate_gemini(
        self,
        prompt: str,
        system: str,
        temperature: float = 0.3,
    ) -> InferenceResult:
        """Generate using Gemini API."""
        import time

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                system_instruction=system,
            )

            start = time.time()
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            latency = (time.time() - start) * 1000

            return InferenceResult(
                text=response.text,
                backend=InferenceBackend.GEMINI,
                model="gemini-2.5-flash",
                latency_ms=latency,
            )
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def get_status(self) -> dict[str, Any]:
        """Get status of all backends."""
        status = {
            "mlx": {
                "available": self._mlx_available,
                "model": self._mlx.default_model if self._mlx else None,
                "device": "Apple Silicon (Metal)" if self._mlx_available else None,
            },
            "ollama": {
                "available": self._ollama_available,
                "models": self._ollama.list_models() if self._ollama_available else [],
            },
            "gemini": {
                "available": bool(self.gemini_api_key),
                "model": "gemini-2.5-flash",
            },
            "primary_backend": self.primary_backend.value,
        }
        return status


# Convenience function for quick testing
def test_inference():
    """Test the inference client."""
    client = SemanticInferenceClient()

    print("Status:", client.get_status())
    print()

    # Test NER
    text = "Patrick Mahomes is questionable for Sunday's game against the Raiders with an ankle injury."
    print(f"Input: {text}")
    print(f"Entities: {client.extract_entities(text)}")
    print()

    # Test sentiment
    text = "Chiefs looking dominant after crushing victory, defense finally clicking"
    print(f"Input: {text}")
    print(f"Sentiment: {client.analyze_sentiment(text)}")


if __name__ == "__main__":
    test_inference()
