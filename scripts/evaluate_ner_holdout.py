"""
Evaluate NER model on holdout validation data.

Tests the fine-tuned NER model's performance on the 500-sample validation split
from data/semantic_training/ner_training_validation.jsonl.

Computes:
- Entity-level precision, recall, F1 (exact match)
- Per-entity-type performance (PLAYER, TEAM, INJURY, STATUS, POSITION)
- Partial match metrics (token overlap)
- Example predictions (correct vs incorrect)

Usage:
    # Test with base model (qwen3:8b)
    python scripts/evaluate_ner_holdout.py

    # Test with fine-tuned model (if created)
    python scripts/evaluate_ner_holdout.py --model nfl-ner

    # Limit evaluation samples for quick testing
    python scripts/evaluate_ner_holdout.py --max-samples 50

    # Output detailed predictions
    python scripts/evaluate_ner_holdout.py --verbose
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nfl_analytics.semantic.models.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def parse_entities_from_response(response: str) -> list[dict[str, Any]]:
    """
    Parse entity list from model response.

    Expected format:
    {"entities": [{"text": "...", "type": "...", "start": N, "end": N}]}

    Args:
        response: Model response string

    Returns:
        List of entity dicts
    """
    # Clean response - remove ANSI codes and extra whitespace
    response = response.strip()

    try:
        # Try to parse as JSON
        data = json.loads(response)
        if "entities" in data:
            return data["entities"]

        # Fallback: extract entities list if it's nested
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return value

        return []
    except json.JSONDecodeError:
        # Model template starts with {"entities": [, so completion might just be the array contents
        # Try wrapping in the full structure
        if not response.startswith("{"):
            # Response might be just the entity array or array items
            try:
                # Try parsing as array
                if response.startswith("["):
                    return json.loads(response)
                else:
                    # Try wrapping as entities array
                    wrapped = f'{{"entities": [{response}]}}'
                    data = json.loads(wrapped)
                    return data.get("entities", [])
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract JSON from text
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(response[start:end])
                if "entities" in data:
                    return data["entities"]
            except json.JSONDecodeError:
                pass

        # Final fallback: try to extract just the entities array
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                # Complete the JSON structure
                wrapped = f'{{"entities": {response[start:end]}}}'
                data = json.loads(wrapped)
                return data.get("entities", [])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse entities from response: {response[:100]}")
        return []


def normalize_entity(entity: dict) -> tuple[str, str]:
    """
    Normalize entity to (text, type) tuple for comparison.

    Args:
        entity: Entity dict with 'text' and 'type' fields

    Returns:
        Tuple of (normalized_text, normalized_type)
    """
    text = entity.get("text", "").strip().lower()
    entity_type = entity.get("type", "").strip().upper()
    return (text, entity_type)


def compute_exact_match_metrics(
    predicted_entities: list[dict],
    expected_entities: list[dict],
) -> dict[str, float]:
    """
    Compute precision, recall, F1 for exact entity matches.

    Args:
        predicted_entities: List of predicted entity dicts
        expected_entities: List of ground truth entity dicts

    Returns:
        Dict with precision, recall, f1, true_positives, false_positives, false_negatives
    """
    predicted_set = set(normalize_entity(e) for e in predicted_entities)
    expected_set = set(normalize_entity(e) for e in expected_entities)

    true_positives = len(predicted_set & expected_set)
    false_positives = len(predicted_set - expected_set)
    false_negatives = len(expected_set - predicted_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_partial_match_metrics(
    predicted_entities: list[dict],
    expected_entities: list[dict],
) -> dict[str, float]:
    """
    Compute metrics allowing partial text matches (token overlap).

    Args:
        predicted_entities: List of predicted entity dicts
        expected_entities: List of ground truth entity dicts

    Returns:
        Dict with partial precision, recall, f1
    """
    def tokenize(text: str) -> set[str]:
        return set(text.lower().split())

    partial_matches = 0
    for pred in predicted_entities:
        pred_tokens = tokenize(pred.get("text", ""))
        pred_type = pred.get("type", "").upper()

        for exp in expected_entities:
            exp_tokens = tokenize(exp.get("text", ""))
            exp_type = exp.get("type", "").upper()

            # Match if types match and tokens overlap > 50%
            if pred_type == exp_type and pred_tokens and exp_tokens:
                overlap = len(pred_tokens & exp_tokens)
                max_len = max(len(pred_tokens), len(exp_tokens))
                if overlap / max_len >= 0.5:
                    partial_matches += 1
                    break

    precision = partial_matches / len(predicted_entities) if predicted_entities else 0.0
    recall = partial_matches / len(expected_entities) if expected_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "partial_precision": precision,
        "partial_recall": recall,
        "partial_f1": f1,
    }


def evaluate_on_validation_set(
    model_name: str,
    validation_file: str,
    max_samples: int | None = None,
    verbose: bool = False,
    timeout: float = 240.0,
) -> dict[str, Any]:
    """
    Evaluate NER model on validation data.

    Args:
        model_name: Name of Ollama model to test
        validation_file: Path to validation JSONL file
        max_samples: Optional limit on number of samples to evaluate
        verbose: Whether to print detailed predictions
        timeout: Timeout in seconds for each inference

    Returns:
        Evaluation results dict
    """
    # Initialize client
    client = OllamaClient(timeout=timeout, default_model=model_name)

    if not client.is_available():
        raise RuntimeError("Ollama not running. Start with: ollama serve")

    # Check model availability
    available_models = client.list_models()
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not found. Available: {available_models}")

    # Load validation data
    validation_path = Path(validation_file)
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_file}")

    with open(validation_path) as f:
        validation_data = [json.loads(line) for line in f if line.strip()]

    if max_samples:
        validation_data = validation_data[:max_samples]

    logger.info(f"Loaded {len(validation_data)} validation samples")

    # Track metrics
    overall_metrics = {
        "model": model_name,
        "total_samples": len(validation_data),
        "successful_parses": 0,
        "failed_parses": 0,
    }

    per_type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_exact_metrics = []
    all_partial_metrics = []
    examples = {"correct": [], "incorrect": []}

    # Evaluate each sample
    for i, sample in enumerate(validation_data, 1):
        prompt = sample.get("prompt", "")
        expected_response = sample.get("response", "")

        # Parse expected entities
        try:
            expected_entities = parse_entities_from_response(expected_response)
        except Exception as e:
            logger.error(f"Failed to parse expected entities for sample {i}: {e}")
            continue

        # Get model prediction
        try:
            response = client.generate(prompt=prompt, temperature=0.1)
            predicted_entities = parse_entities_from_response(response.response)

            if predicted_entities:
                overall_metrics["successful_parses"] += 1
            else:
                overall_metrics["failed_parses"] += 1

        except Exception as e:
            logger.error(f"Inference failed for sample {i}: {e}")
            overall_metrics["failed_parses"] += 1
            continue

        # Compute exact match metrics
        exact_metrics = compute_exact_match_metrics(predicted_entities, expected_entities)
        all_exact_metrics.append(exact_metrics)

        # Compute partial match metrics
        partial_metrics = compute_partial_match_metrics(predicted_entities, expected_entities)
        all_partial_metrics.append(partial_metrics)

        # Track per-type metrics
        predicted_set = set(normalize_entity(e) for e in predicted_entities)
        expected_set = set(normalize_entity(e) for e in expected_entities)

        for entity in expected_entities:
            entity_type = entity.get("type", "UNKNOWN").upper()
            if normalize_entity(entity) in predicted_set:
                per_type_metrics[entity_type]["tp"] += 1
            else:
                per_type_metrics[entity_type]["fn"] += 1

        for entity in predicted_entities:
            entity_type = entity.get("type", "UNKNOWN").upper()
            if normalize_entity(entity) not in expected_set:
                per_type_metrics[entity_type]["fp"] += 1

        # Save examples
        is_correct = exact_metrics["f1"] == 1.0
        example = {
            "prompt": prompt,
            "expected": expected_entities,
            "predicted": predicted_entities,
            "exact_f1": exact_metrics["f1"],
            "partial_f1": partial_metrics["partial_f1"],
        }

        if is_correct and len(examples["correct"]) < 5:
            examples["correct"].append(example)
        elif not is_correct and len(examples["incorrect"]) < 5:
            examples["incorrect"].append(example)

        if verbose and i <= 10:
            print(f"\nSample {i}:")
            print(f"  Prompt: {prompt[:80]}...")
            print(f"  Expected: {expected_entities}")
            print(f"  Predicted: {predicted_entities}")
            print(f"  Exact F1: {exact_metrics['f1']:.3f}, Partial F1: {partial_metrics['partial_f1']:.3f}")

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(validation_data)} samples")

    # Aggregate metrics
    overall_metrics["exact_match"] = {
        "precision": sum(m["precision"] for m in all_exact_metrics) / len(all_exact_metrics) if all_exact_metrics else 0,
        "recall": sum(m["recall"] for m in all_exact_metrics) / len(all_exact_metrics) if all_exact_metrics else 0,
        "f1": sum(m["f1"] for m in all_exact_metrics) / len(all_exact_metrics) if all_exact_metrics else 0,
    }

    overall_metrics["partial_match"] = {
        "precision": sum(m["partial_precision"] for m in all_partial_metrics) / len(all_partial_metrics) if all_partial_metrics else 0,
        "recall": sum(m["partial_recall"] for m in all_partial_metrics) / len(all_partial_metrics) if all_partial_metrics else 0,
        "f1": sum(m["partial_f1"] for m in all_partial_metrics) / len(all_partial_metrics) if all_partial_metrics else 0,
    }

    # Compute per-type metrics
    per_type_summary = {}
    for entity_type, counts in per_type_metrics.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type_summary[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,  # Total expected entities of this type
        }

    overall_metrics["per_type"] = per_type_summary
    overall_metrics["examples"] = examples

    return overall_metrics


def print_evaluation_report(results: dict) -> None:
    """Print evaluation results in readable format."""
    print("\n" + "=" * 70)
    print(f"NER MODEL EVALUATION REPORT: {results['model']}")
    print("=" * 70)
    print(f"Total samples: {results['total_samples']}")
    print(f"Successful parses: {results['successful_parses']}")
    print(f"Failed parses: {results['failed_parses']}")
    print(f"Parse success rate: {results['successful_parses'] / results['total_samples']:.1%}")

    print("\n" + "-" * 70)
    print("OVERALL METRICS (Exact Match)")
    print("-" * 70)
    exact = results["exact_match"]
    print(f"Precision: {exact['precision']:.3f}")
    print(f"Recall:    {exact['recall']:.3f}")
    print(f"F1 Score:  {exact['f1']:.3f}")

    print("\n" + "-" * 70)
    print("OVERALL METRICS (Partial Match - Token Overlap)")
    print("-" * 70)
    partial = results["partial_match"]
    print(f"Precision: {partial['precision']:.3f}")
    print(f"Recall:    {partial['recall']:.3f}")
    print(f"F1 Score:  {partial['f1']:.3f}")

    print("\n" + "-" * 70)
    print("PER-ENTITY-TYPE PERFORMANCE (Exact Match)")
    print("-" * 70)
    print(f"{'Type':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)

    per_type = results["per_type"]
    for entity_type in sorted(per_type.keys()):
        metrics = per_type[entity_type]
        print(
            f"{entity_type:<15} "
            f"{metrics['precision']:<12.3f} "
            f"{metrics['recall']:<12.3f} "
            f"{metrics['f1']:<12.3f} "
            f"{metrics['support']:<10}"
        )

    print("\n" + "-" * 70)
    print("EXAMPLE PREDICTIONS")
    print("-" * 70)

    examples = results["examples"]

    print("\nCORRECT PREDICTIONS:")
    for i, ex in enumerate(examples["correct"][:3], 1):
        print(f"\n{i}. Prompt: {ex['prompt'][:80]}...")
        print(f"   Predicted: {ex['predicted']}")
        print(f"   Expected:  {ex['expected']}")

    print("\nINCORRECT PREDICTIONS:")
    for i, ex in enumerate(examples["incorrect"][:3], 1):
        print(f"\n{i}. Prompt: {ex['prompt'][:80]}...")
        print(f"   Predicted: {ex['predicted']}")
        print(f"   Expected:  {ex['expected']}")
        print(f"   Exact F1: {ex['exact_f1']:.3f}, Partial F1: {ex['partial_f1']:.3f}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluate NER model on holdout data")
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Ollama model name (default: qwen3:8b)",
    )
    parser.add_argument(
        "--validation-file",
        default="data/semantic_training/ner_training_validation.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        default="results/ner_holdout_evaluation.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed predictions",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Timeout in seconds for each inference",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Run evaluation
        logger.info(f"Starting evaluation of model: {args.model}")
        results = evaluate_on_validation_set(
            model_name=args.model,
            validation_file=args.validation_file,
            max_samples=args.max_samples,
            verbose=args.verbose,
            timeout=args.timeout,
        )

        # Add metadata
        results["evaluation_timestamp"] = datetime.now().isoformat()
        results["validation_file"] = args.validation_file

        # Print report
        print_evaluation_report(results)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
