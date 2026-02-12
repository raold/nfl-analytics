#!/usr/bin/env python3
"""
Batch LLM Processing for Injury Severity Scoring.

Processes all unscored injuries in semantic.injury_snapshots using Ollama (qwen3:8b).
Assigns a severity score from 0.0 (minor) to 1.0 (severe) based on injury type,
location, and game-day status.

Uses status-based baseline with LLM adjustment for injury-specific factors.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import RealDictCursor

from py.semantic.models.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Status-based baseline scores (NFL game impact)
STATUS_BASELINE = {
    "Out": 0.95,
    "Injured Reserve": 1.0,  # Season-ending
    "Doubtful": 0.85,
    "Questionable": 0.50,
    "Probable": 0.20,
    "Active": 0.05,  # Minor/managed
    "Suspension": 0.90,  # Similar to Out
    "Reserve-Ret": 0.0,  # Retired
}

# Injury type severity modifiers (added to baseline)
INJURY_SEVERITY = {
    # Season-ending (high modifier)
    "ACL": 0.10,
    "Achilles": 0.10,
    "Torn": 0.10,

    # Significant (medium modifier)
    "Knee": 0.05,
    "Ankle": 0.03,
    "Shoulder": 0.03,
    "Hamstring": 0.03,
    "Concussion": 0.05,
    "Ribs": 0.03,

    # Moderate
    "Foot": 0.02,
    "Calf": 0.02,
    "Back": 0.02,
    "Neck": 0.02,
    "Hip": 0.02,
    "Quadriceps": 0.02,
    "Groin": 0.02,

    # Minor (negative modifier)
    "Rest": -0.10,
    "Personal": -0.05,
    "Illness": -0.05,
    "Unknown": 0.0,
    "Not Injury Related": -0.10,
}


def get_unprocessed_injuries(limit: int = 100) -> list[dict]:
    """Get injuries without severity scores."""
    query = """
    SELECT id, team, status, injury_type, injury_location, athlete_id
    FROM semantic.injury_snapshots
    WHERE severity_score IS NULL
    ORDER BY
        CASE status
            WHEN 'Out' THEN 1
            WHEN 'Doubtful' THEN 2
            WHEN 'Questionable' THEN 3
            ELSE 4
        END,
        id
    LIMIT %s
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (limit,))
            return [dict(row) for row in cur.fetchall()]


def compute_rule_based_severity(injury: dict) -> float:
    """
    Compute severity using rule-based approach (fast, no LLM).

    Returns value 0.0-1.0 based on status and injury type.
    """
    status = injury.get("status", "Unknown")
    injury_type = injury.get("injury_type") or "Unknown"

    # Get baseline from status
    baseline = STATUS_BASELINE.get(status, 0.30)

    # Add modifier from injury type
    modifier = 0.0
    for injury_key, mod_value in INJURY_SEVERITY.items():
        if injury_key.lower() in injury_type.lower():
            modifier = max(modifier, mod_value)  # Use highest applicable
            break

    # Clamp to 0.0-1.0
    severity = max(0.0, min(1.0, baseline + modifier))

    return round(severity, 2)


def compute_llm_severity(client: OllamaClient, injury: dict) -> float | None:
    """
    Use LLM to assess injury severity with more nuance.

    Falls back to rule-based if LLM fails.
    """
    injury_type = injury.get("injury_type") or "Unknown"
    injury_location = injury.get("injury_location") or "Unknown"
    status = injury.get("status", "Unknown")
    team = injury.get("team", "UNK")

    # For trivial cases, skip LLM
    if status in ("Active", "Reserve-Ret") and injury_type in ("Unknown", "Rest", ""):
        return compute_rule_based_severity(injury)

    prompt = f"""Rate NFL injury severity (0.0-1.0 scale).

Injury: {injury_type} ({injury_location})
Status: {status}
Team: {team}

Scale:
- 1.0: Season-ending (ACL tear, Achilles)
- 0.8-0.9: Multi-week absence (broken bone, high ankle sprain)
- 0.5-0.7: 1-2 games missed (hamstring, concussion protocol)
- 0.2-0.4: Game-time decision (minor sprain, illness)
- 0.0-0.2: Active/managed (rest, minor)

Return ONLY a number between 0.0 and 1.0."""

    try:
        response = client.generate(prompt, temperature=0.1)
        text = response.response.strip()

        # Extract number from response
        # Handle formats: "0.75", "0.75.", "The severity is 0.75", etc.
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            score = float(match.group(1))
            # Normalize if percentage (85 -> 0.85)
            if score > 1.0 and score <= 100:
                score = score / 100.0
            return max(0.0, min(1.0, round(score, 2)))

        logger.warning(f"Could not parse LLM response: {text}")
        return None

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


def update_severity_score(injury_id: int, severity: float) -> bool:
    """Update severity_score and mark as processed."""
    query = """
    UPDATE semantic.injury_snapshots
    SET severity_score = %s, processed = true
    WHERE id = %s
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (severity, injury_id))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to update injury {injury_id}: {e}")
        return False


def batch_update_severities(updates: list[tuple[int, float]]) -> int:
    """Batch update severity scores for efficiency."""
    if not updates:
        return 0

    query = """
    UPDATE semantic.injury_snapshots
    SET severity_score = data.severity, processed = true
    FROM (VALUES %s) AS data(id, severity)
    WHERE injury_snapshots.id = data.id
    """

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                # Build values string
                values = ", ".join(f"({id_}, {sev})" for id_, sev in updates)
                full_query = query.replace("%s", values)
                cur.execute(full_query)
                conn.commit()
                return len(updates)
    except Exception as e:
        logger.error(f"Batch update failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Process injury severity scores")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of injuries to process per batch (default: 50)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Maximum total injuries to process (0=all)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for nuanced scoring (slower but more accurate)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute scores but don't update database",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("INJURY SEVERITY SCORING")
    print("=" * 60)
    print(f"Mode: {'LLM (qwen3:8b)' if args.use_llm else 'Rule-based (fast)'}")
    print(f"Batch size: {args.batch_size}")
    if args.max_total:
        print(f"Max total: {args.max_total}")
    print()

    # Initialize Ollama client if using LLM
    client = None
    if args.use_llm:
        client = OllamaClient(timeout=120.0)
        if not client.is_available():
            print("ERROR: Ollama not available. Run 'ollama serve' first.")
            print("Falling back to rule-based scoring...")
            args.use_llm = False
        else:
            print("Ollama connected successfully")

    total_processed = 0
    total_updated = 0
    start_time = time.time()

    while True:
        # Determine how many to fetch
        remaining = args.max_total - total_processed if args.max_total else args.batch_size
        fetch_size = min(args.batch_size, remaining) if args.max_total else args.batch_size

        if fetch_size <= 0:
            break

        injuries = get_unprocessed_injuries(limit=fetch_size)
        if not injuries:
            print("\nNo more unprocessed injuries.")
            break

        print(f"\nProcessing batch of {len(injuries)} injuries...")

        updates = []
        for injury in injuries:
            # Compute severity
            if args.use_llm:
                severity = compute_llm_severity(client, injury)
                if severity is None:
                    severity = compute_rule_based_severity(injury)
            else:
                severity = compute_rule_based_severity(injury)

            updates.append((injury["id"], severity))
            total_processed += 1

            if total_processed % 100 == 0:
                print(f"  Processed {total_processed} injuries...")

        # Batch update
        if not args.dry_run:
            updated = batch_update_severities(updates)
            total_updated += updated
        else:
            total_updated += len(updates)
            print(f"  [DRY RUN] Would update {len(updates)} records")

        # Rate limit for LLM
        if args.use_llm and len(injuries) == fetch_size:
            time.sleep(0.5)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total processed: {total_processed}")
    print(f"Total updated: {total_updated}")
    print(f"Time elapsed: {elapsed:.1f}s")
    if total_processed > 0:
        print(f"Rate: {total_processed / elapsed:.1f} injuries/sec")
    print()

    # Show sample of scored injuries
    if not args.dry_run and total_updated > 0:
        print("Sample of scored injuries:")
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT team, status, injury_type, severity_score
                    FROM semantic.injury_snapshots
                    WHERE severity_score IS NOT NULL
                    ORDER BY severity_score DESC
                    LIMIT 10
                """)
                for row in cur.fetchall():
                    print(f"  {row['team']} | {row['status']:15} | {row['injury_type']:15} | {row['severity_score']:.2f}")

    # Verify final state
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(severity_score) as scored,
                    AVG(severity_score) as avg_severity
                FROM semantic.injury_snapshots
            """)
            row = cur.fetchone()
            print(f"\nDatabase state:")
            print(f"  Total injuries: {row[0]}")
            print(f"  With severity: {row[1]}")
            if row[2]:
                print(f"  Avg severity: {row[2]:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
