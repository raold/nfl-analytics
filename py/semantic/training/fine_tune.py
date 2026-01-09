"""
Fine-tuning utilities for NFL NER and sentiment models.

Supports multiple fine-tuning approaches:
1. Ollama custom model (fastest, uses Modelfile)
2. MLX fine-tuning (Apple Silicon optimized, uses LoRA)
3. HuggingFace/PEFT fine-tuning (most flexible, GPU required)

Usage:
    # Option 1: Create Ollama custom model (no actual fine-tuning, just system prompt)
    python -m py.semantic.training.fine_tune --method ollama --data data/semantic_training

    # Option 2: MLX LoRA fine-tuning (requires Apple Silicon)
    python -m py.semantic.training.fine_tune --method mlx --data data/semantic_training

    # Option 3: Validate fine-tuned model
    python -m py.semantic.training.fine_tune --validate --model nfl-ner
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_ollama_model(
    training_dir: str,
    model_name: str = "nfl-ner",
    base_model: str = "qwen3:8b",
) -> bool:
    """
    Create Ollama custom model with NFL-specific system prompt.

    Note: This doesn't do actual fine-tuning - it creates a model variant
    with a specialized system prompt. For true fine-tuning, use MLX or PEFT.

    Args:
        training_dir: Directory containing Modelfile
        model_name: Name for the custom model
        base_model: Base model to customize

    Returns:
        True if successful
    """
    training_path = Path(training_dir)
    modelfile_path = training_path / "Modelfile"

    if not modelfile_path.exists():
        logger.error(f"Modelfile not found at {modelfile_path}")
        return False

    # Update Modelfile to use specified base model
    with open(modelfile_path) as f:
        content = f.read()

    if f"FROM {base_model}" not in content:
        content = content.replace("FROM qwen3:8b", f"FROM {base_model}")
        with open(modelfile_path, "w") as f:
            f.write(content)

    # Create model using Ollama
    logger.info(f"Creating Ollama model '{model_name}' from {modelfile_path}")

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            logger.info(f"Successfully created model: {model_name}")
            logger.info(f"Test with: ollama run {model_name}")
            return True
        else:
            logger.error(f"Failed to create model: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Model creation timed out")
        return False
    except FileNotFoundError:
        logger.error("Ollama not found. Install from https://ollama.ai")
        return False


def create_mlx_lora_config(
    training_dir: str,
    output_dir: str = "models/nfl-ner-lora",
    base_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
    epochs: int = 3,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
) -> dict:
    """
    Create MLX LoRA fine-tuning configuration.

    Args:
        training_dir: Directory with training data
        output_dir: Output directory for LoRA adapters
        base_model: Base model from HuggingFace
        epochs: Number of training epochs
        learning_rate: Learning rate
        lora_rank: LoRA rank (higher = more capacity, slower)

    Returns:
        Configuration dict
    """
    config = {
        "model": base_model,
        "train": str(Path(training_dir) / "train.jsonl"),
        "valid": str(Path(training_dir) / "ner_training_validation.jsonl"),
        "adapter_path": output_dir,
        "iters": epochs * 1000,  # Approximate iterations
        "batch_size": 4,
        "learning_rate": learning_rate,
        "lora_layers": 16,
        "lora_rank": lora_rank,
        "seed": 42,
    }

    # Save config
    config_path = Path(output_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"MLX LoRA config saved to {config_path}")

    # Generate training command
    train_cmd = f"""
# MLX LoRA Fine-tuning Command
# Requires: pip install mlx-lm

python -m mlx_lm.lora \\
    --model {base_model} \\
    --train \\
    --data {training_dir} \\
    --adapter-path {output_dir} \\
    --batch-size 4 \\
    --iters {config['iters']} \\
    --learning-rate {learning_rate} \\
    --lora-layers 16

# After training, test with:
python -m mlx_lm.generate \\
    --model {base_model} \\
    --adapter-path {output_dir} \\
    --prompt "Extract NFL entities from: Patrick Mahomes questionable with ankle injury"
"""

    script_path = Path(output_dir) / "train.sh"
    with open(script_path, "w") as f:
        f.write(train_cmd)

    logger.info(f"Training script saved to {script_path}")

    return config


def validate_model(
    model_name: str,
    test_cases: list[dict] | None = None,
    use_ollama: bool = True,
) -> dict[str, Any]:
    """
    Validate a fine-tuned model on test cases.

    Args:
        model_name: Name of the model to validate
        test_cases: Optional list of test cases with 'text' and 'expected_entities'
        use_ollama: Whether to use Ollama for inference

    Returns:
        Validation results
    """
    if test_cases is None:
        # Default test cases
        test_cases = [
            {
                "text": "Patrick Mahomes is questionable for Sunday with an ankle injury.",
                "expected_entities": ["Patrick Mahomes", "ankle", "questionable"],
            },
            {
                "text": "The Chiefs list Travis Kelce as doubtful with a knee issue.",
                "expected_entities": ["Chiefs", "Travis Kelce", "doubtful", "knee"],
            },
            {
                "text": "Ravens RB Derrick Henry practiced in full on Wednesday.",
                "expected_entities": ["Ravens", "Derrick Henry", "RB"],
            },
        ]

    results = {
        "model": model_name,
        "total_cases": len(test_cases),
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    if use_ollama:
        try:
            from py.semantic.models.ollama_client import OllamaClient

            client = OllamaClient(timeout=120.0, default_model=model_name)

            if not client.is_available():
                logger.error("Ollama not running")
                return results

            for case in test_cases:
                text = case["text"]
                expected = case["expected_entities"]

                # Get model response
                response = client.generate(
                    prompt=f"Extract NFL entities from: {text}",
                    temperature=0.1,
                )

                response_text = response.response.lower()

                # Check if expected entities are found
                found = []
                missing = []
                for entity in expected:
                    if entity.lower() in response_text:
                        found.append(entity)
                    else:
                        missing.append(entity)

                passed = len(missing) == 0
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

                results["details"].append(
                    {
                        "text": text,
                        "expected": expected,
                        "found": found,
                        "missing": missing,
                        "passed": passed,
                        "response": response.response[:200],
                    }
                )

        except Exception as e:
            logger.error(f"Validation error: {e}")

    results["accuracy"] = (
        results["passed"] / results["total_cases"] if results["total_cases"] > 0 else 0.0
    )

    return results


def print_validation_report(results: dict) -> None:
    """Print validation results in a readable format."""
    print("\n" + "=" * 60)
    print(f"MODEL VALIDATION REPORT: {results['model']}")
    print("=" * 60)
    print(f"Total cases: {results['total_cases']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print("-" * 60)

    for i, detail in enumerate(results["details"], 1):
        status = "PASS" if detail["passed"] else "FAIL"
        print(f"\n{i}. [{status}] {detail['text'][:50]}...")
        print(f"   Expected: {detail['expected']}")
        print(f"   Found: {detail['found']}")
        if detail["missing"]:
            print(f"   Missing: {detail['missing']}")


def main():
    """Command-line interface for fine-tuning utilities."""
    parser = argparse.ArgumentParser(description="Fine-tune NFL NER/sentiment models")
    parser.add_argument(
        "--method",
        choices=["ollama", "mlx", "validate"],
        default="validate",
        help="Fine-tuning method or validation",
    )
    parser.add_argument(
        "--data",
        default="data/semantic_training",
        help="Training data directory",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Model name (for Ollama) or path",
    )
    parser.add_argument(
        "--output",
        default="models/nfl-ner-lora",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the model after creation",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.method == "ollama":
        success = create_ollama_model(
            training_dir=args.data,
            model_name="nfl-ner",
            base_model=args.model,
        )
        if success and args.validate:
            results = validate_model("nfl-ner")
            print_validation_report(results)

    elif args.method == "mlx":
        config = create_mlx_lora_config(
            training_dir=args.data,
            output_dir=args.output,
        )
        print(f"\nMLX config created. To train:")
        print(f"  bash {args.output}/train.sh")

    elif args.method == "validate":
        results = validate_model(args.model)
        print_validation_report(results)


if __name__ == "__main__":
    main()
