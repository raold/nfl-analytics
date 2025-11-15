# API Advanced Examples

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

Advanced integration patterns and optimization techniques for the NFL Analytics API. Includes batch prediction optimization, streaming responses, webhook integration, real-time odds, multi-season backtesting, and rate limit optimization.

## Table of Contents

1. [Batch Prediction Optimization](#batch-prediction-optimization)
2. [Streaming Responses](#streaming-responses)
3. [Webhook Integration](#webhook-integration)
4. [Real-Time Odds Integration](#real-time-odds-integration)
5. [Multi-Season Backtesting](#multi-season-backtesting)
6. [Rate Limit Optimization](#rate-limit-optimization)
7. [Parallel Request Handling](#parallel-request-handling)
8. [Production-Grade Integration](#production-grade-integration)

---

## Batch Prediction Optimization

### Overview

Batch predictions offer 10-50x speedup over individual requests by eliminating connection overhead and enabling server-side parallelization.

**Performance Comparison:**
- **Individual requests:** 100 predictions × 500ms = 50 seconds
- **Batch request:** 100 predictions ÷ 10 batches = 5 seconds (10x speedup)

### Basic Batch Request

```python
import requests
import os

API_KEY = os.getenv("NFL_API_KEY")
BASE_URL = "https://api.nfl-analytics.com"

def predict_batch_basic(predictions):
    """Simple batch prediction"""
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        json={"predictions": predictions},
        timeout=30
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Batch failed: {response.json()['message']}")

# Usage
predictions = [
    {"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"},
    {"game_id": "2024_10_KC_LV", "player_id": "00-0034857", "stat_type": "rushing_yards"},
    {"game_id": "2024_10_KC_LV", "player_id": "00-0036355", "stat_type": "receiving_yards"}
]

results = predict_batch_basic(predictions)
print(f"Completed: {results['successful_predictions']}/{results['total_predictions']}")
```

### Chunked Batch Processing (50x Speedup)

Process large datasets by chunking into 100-prediction batches:

```python
import time
from typing import List, Dict

def predict_batch_chunked(predictions: List[Dict], chunk_size: int = 100) -> List[Dict]:
    """
    Process large prediction sets with automatic chunking.

    Args:
        predictions: List of prediction requests
        chunk_size: Maximum batch size (default: 100)

    Returns:
        List of all predictions (successful and failed)
    """
    all_results = []

    # Split into chunks
    chunks = [predictions[i:i + chunk_size] for i in range(0, len(predictions), chunk_size)]

    print(f"Processing {len(predictions)} predictions in {len(chunks)} batches...")

    for i, chunk in enumerate(chunks):
        try:
            response = requests.post(
                f"{BASE_URL}/predict/batch",
                headers={"X-API-Key": API_KEY},
                json={"predictions": chunk},
                timeout=30
            )

            if response.status_code == 200:
                batch_result = response.json()
                all_results.extend(batch_result['predictions'])

                success_rate = (
                    batch_result['successful_predictions'] /
                    batch_result['total_predictions'] * 100
                )
                print(f"  Batch {i+1}/{len(chunks)}: {success_rate:.1f}% success")
            else:
                print(f"  Batch {i+1}/{len(chunks)}: Failed - {response.status_code}")

        except Exception as e:
            print(f"  Batch {i+1}/{len(chunks)}: Error - {e}")

        # Rate limit protection: Sleep between batches
        if i < len(chunks) - 1:
            time.sleep(1)

    return all_results

# Usage: Predict entire week of games
week_10_predictions = []
for game_id in ["2024_10_KC_LV", "2024_10_BUF_IND", "2024_10_SF_DAL"]:
    for player in get_players_for_game(game_id):
        week_10_predictions.append({
            "game_id": game_id,
            "player_id": player['id'],
            "stat_type": player['position'] + '_yards'
        })

results = predict_batch_chunked(week_10_predictions, chunk_size=100)
print(f"\nTotal predictions: {len(results)}")
```

### Parallel Batch Processing (Additional 2-5x Speedup)

Use `concurrent.futures` for parallel batch requests:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def predict_batch_parallel(predictions: List[Dict], chunk_size: int = 100, max_workers: int = 5) -> List[Dict]:
    """
    Parallel batch prediction processing.

    WARNING: Consumes rate limit faster. Use only if hourly limit permits.

    Args:
        predictions: List of prediction requests
        chunk_size: Batch size (default: 100)
        max_workers: Parallel workers (default: 5)

    Returns:
        Combined results from all batches
    """
    chunks = [predictions[i:i + chunk_size] for i in range(0, len(predictions), chunk_size)]
    all_results = []

    def process_batch(chunk_idx: int, chunk: List[Dict]) -> Dict:
        """Process single batch"""
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            headers={"X-API-Key": API_KEY},
            json={"predictions": chunk},
            timeout=30
        )
        return {
            "idx": chunk_idx,
            "status": response.status_code,
            "data": response.json() if response.status_code == 200 else None
        }

    # Submit all batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, i, chunk): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result['status'] == 200 and result['data']:
                    all_results.extend(result['data']['predictions'])
                    print(f"✓ Batch {result['idx'] + 1}/{len(chunks)} completed")
                else:
                    print(f"✗ Batch {result['idx'] + 1}/{len(chunks)} failed")
            except Exception as e:
                print(f"Error processing batch: {e}")

    return all_results

# Usage
results = predict_batch_parallel(week_10_predictions, max_workers=5)
```

**Performance Summary:**

| Method | 1,000 Predictions | 10,000 Predictions |
|--------|-------------------|---------------------|
| Individual (500ms/req) | 8.3 minutes | 83 minutes |
| Batch Sequential | 50 seconds | 8.3 minutes |
| Batch Parallel (5 workers) | 10 seconds | 100 seconds |

---

## Streaming Responses

### Server-Sent Events (SSE) for Large Batches

**Note:** Streaming is not currently implemented but can be added for async batch processing.

**Proposed Future API:**

```python
import sseclient  # pip install sseclient-py
import requests

def predict_batch_streaming(predictions: List[Dict]):
    """
    Stream batch predictions as they complete (proposed feature).

    Allows processing results incrementally without waiting for entire batch.
    """
    response = requests.post(
        f"{BASE_URL}/predict/batch/stream",  # Future endpoint
        headers={
            "X-API-Key": API_KEY,
            "Accept": "text/event-stream"
        },
        json={"predictions": predictions},
        stream=True
    )

    client = sseclient.SSEClient(response)

    for event in client.events():
        if event.event == "prediction":
            prediction = json.loads(event.data)
            # Process prediction immediately
            save_to_database(prediction)
            print(f"Received: {prediction['player_id']} = {prediction['prediction']:.1f}")
        elif event.event == "complete":
            summary = json.loads(event.data)
            print(f"\nBatch complete: {summary['successful_predictions']}/{summary['total_predictions']}")
            break
```

---

## Webhook Integration

### Overview

Webhooks enable asynchronous predictions where the API calls your endpoint when predictions are ready. Useful for very large batch jobs (1,000+ predictions).

**Flow:**
1. Client submits batch prediction with `callback_url`
2. API immediately returns 202 Accepted with job ID
3. API processes predictions asynchronously
4. API POSTs results to `callback_url` when complete

**See:** [Webhook Documentation](./webhooks.md) for complete implementation guide.

### Quick Example

```python
import requests
from flask import Flask, request, jsonify

# Webhook receiver (your server)
app = Flask(__name__)

@app.route("/webhook/predictions", methods=["POST"])
def receive_predictions():
    """Handle prediction webhook from API"""
    payload = request.json

    job_id = payload['job_id']
    status = payload['status']

    if status == "completed":
        results = payload['results']
        successful = results['successful_predictions']
        total = results['total_predictions']

        # Process results
        for prediction in results['predictions']:
            save_to_database(prediction)

        print(f"Job {job_id}: {successful}/{total} predictions completed")
        return jsonify({"status": "received"}), 200
    else:
        print(f"Job {job_id}: Failed - {payload.get('error')}")
        return jsonify({"status": "acknowledged"}), 200

# Submit async batch prediction
def submit_async_batch(predictions: List[Dict], callback_url: str) -> str:
    """Submit batch with webhook callback (future feature)"""
    response = requests.post(
        f"{BASE_URL}/predict/batch/async",
        headers={"X-API-Key": API_KEY},
        json={
            "predictions": predictions,
            "callback_url": callback_url,
            "callback_secret": "your_webhook_secret_here"  # For HMAC verification
        }
    )

    if response.status_code == 202:
        job_data = response.json()
        return job_data['job_id']
    else:
        raise Exception(f"Async batch failed: {response.json()}")

# Usage
job_id = submit_async_batch(
    predictions=large_prediction_set,
    callback_url="https://your-domain.com/webhook/predictions"
)
print(f"Job submitted: {job_id}")
# Results will be POSTed to callback_url when ready
```

---

## Real-Time Odds Integration

### Overview

Integrate ensemble predictions with real-time sportsbook odds for automated betting decisions.

### Example: DraftKings Integration

```python
import requests
from datetime import datetime

class NFLBettingIntegration:
    """Combine API predictions with real-time odds"""

    def __init__(self, api_key: str, draftkings_key: str):
        self.api_key = api_key
        self.dk_key = draftkings_key
        self.base_url = "https://api.nfl-analytics.com"
        self.dk_url = "https://api.draftkings.com/odds/v1"

    def get_prediction(self, game_id: str, player_id: str, stat_type: str) -> Dict:
        """Get ensemble prediction"""
        response = requests.post(
            f"{self.base_url}/predict",
            headers={"X-API-Key": self.api_key},
            json={
                "game_id": game_id,
                "player_id": player_id,
                "stat_type": stat_type
            }
        )
        return response.json()

    def get_odds(self, player_name: str, stat_type: str, line: float) -> Dict:
        """Get DraftKings odds for player prop"""
        response = requests.get(
            f"{self.dk_url}/player_props",
            headers={"X-API-Key": self.dk_key},
            params={
                "player": player_name,
                "market": stat_type,
                "line": line
            }
        )
        return response.json()

    def calculate_edge(self, prediction: Dict, odds: Dict) -> Dict:
        """
        Calculate betting edge.

        Returns:
            edge_pct: Expected value as percentage
            kelly_bet: Kelly criterion bet size
            recommendation: "BET OVER" | "BET UNDER" | "NO BET"
        """
        # Extract prediction bounds
        pred_mean = prediction['prediction']
        pred_lower = prediction['lower_bound']
        pred_upper = prediction['upper_bound']

        # Extract odds
        line = odds['line']
        over_odds = odds['over_odds']  # e.g., -110
        under_odds = odds['under_odds']  # e.g., -110

        # Convert American odds to decimal
        over_prob_implied = self._american_to_prob(over_odds)
        under_prob_implied = self._american_to_prob(under_odds)

        # Calculate true probability (from ensemble)
        # Use normal approximation with ensemble mean/std
        pred_std = (pred_upper - pred_lower) / (2 * 1.645)  # 90% CI
        over_prob_true = 1 - scipy.stats.norm.cdf(line, pred_mean, pred_std)
        under_prob_true = scipy.stats.norm.cdf(line, pred_mean, pred_std)

        # Calculate edge
        over_edge = over_prob_true - over_prob_implied
        under_edge = under_prob_true - under_prob_implied

        # Kelly criterion
        over_kelly = over_edge / (self._american_to_decimal(over_odds) - 1)
        under_kelly = under_edge / (self._american_to_decimal(under_odds) - 1)

        # Determine recommendation (require 5% edge minimum)
        if over_edge > 0.05:
            return {
                "recommendation": "BET OVER",
                "edge_pct": over_edge * 100,
                "kelly_bet": max(0, over_kelly * 0.25),  # 1/4 Kelly for safety
                "line": line,
                "odds": over_odds
            }
        elif under_edge > 0.05:
            return {
                "recommendation": "BET UNDER",
                "edge_pct": under_edge * 100,
                "kelly_bet": max(0, under_kelly * 0.25),
                "line": line,
                "odds": under_odds
            }
        else:
            return {
                "recommendation": "NO BET",
                "edge_pct": max(over_edge, under_edge) * 100,
                "kelly_bet": 0,
                "line": line
            }

    @staticmethod
    def _american_to_prob(american_odds: int) -> float:
        """Convert American odds to implied probability"""
        if american_odds < 0:
            return -american_odds / (-american_odds + 100)
        else:
            return 100 / (american_odds + 100)

    @staticmethod
    def _american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal"""
        if american_odds < 0:
            return 1 + (100 / -american_odds)
        else:
            return 1 + (american_odds / 100)

# Usage
betting = NFLBettingIntegration(
    api_key=os.getenv("NFL_API_KEY"),
    draftkings_key=os.getenv("DK_API_KEY")
)

# Get prediction
prediction = betting.get_prediction(
    game_id="2024_10_KC_LV",
    player_id="00-0033873",
    stat_type="passing_yards"
)

# Get odds
odds = betting.get_odds(
    player_name="Patrick Mahomes",
    stat_type="passing_yards",
    line=275.5
)

# Calculate edge
edge = betting.calculate_edge(prediction, odds)

print(f"Prediction: {prediction['prediction']:.1f} yards (90% CI: [{prediction['lower_bound']:.1f}, {prediction['upper_bound']:.1f}])")
print(f"Line: {edge['line']} yards @ {edge['odds']}")
print(f"Recommendation: {edge['recommendation']}")
print(f"Edge: {edge['edge_pct']:.2f}%")
print(f"Kelly bet size: {edge['kelly_bet']:.2%} of bankroll")
```

---

## Multi-Season Backtesting

### Overview

Validate prediction accuracy across multiple seasons using the API.

### Rolling Window Backtest

```python
import pandas as pd
from datetime import datetime, timedelta

class MultiSeasonBacktest:
    """Backtest ensemble predictions across multiple seasons"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nfl-analytics.com"
        self.results = []

    def backtest_season(self, season: int, stat_type: str = "passing_yards") -> pd.DataFrame:
        """
        Backtest single season.

        Args:
            season: Year (e.g., 2024)
            stat_type: Stat to predict

        Returns:
            DataFrame with predictions vs actuals
        """
        # Get games for season
        games = self.get_season_games(season)

        for game in games:
            # Skip if game hasn't been played
            if game['kickoff'] > datetime.now():
                continue

            # Get starters
            players = self.get_game_starters(game['game_id'])

            for player in players:
                # Get prediction
                try:
                    prediction = self.predict(
                        game_id=game['game_id'],
                        player_id=player['player_id'],
                        stat_type=stat_type
                    )

                    # Get actual result
                    actual = self.get_actual_stats(
                        game_id=game['game_id'],
                        player_id=player['player_id'],
                        stat_type=stat_type
                    )

                    self.results.append({
                        "season": season,
                        "week": game['week'],
                        "game_id": game['game_id'],
                        "player_id": player['player_id'],
                        "player_name": player['name'],
                        "prediction": prediction['prediction'],
                        "actual": actual,
                        "error": abs(prediction['prediction'] - actual),
                        "lower_bound": prediction['lower_bound'],
                        "upper_bound": prediction['upper_bound'],
                        "calibrated": (
                            prediction['lower_bound'] <= actual <= prediction['upper_bound']
                        )
                    })

                except Exception as e:
                    print(f"Error: {game['game_id']} {player['player_id']}: {e}")

        return pd.DataFrame(self.results)

    def backtest_multi_season(self, seasons: List[int]) -> pd.DataFrame:
        """Run backtest across multiple seasons"""
        for season in seasons:
            print(f"\nBacktesting {season}...")
            df = self.backtest_season(season)

            # Print season summary
            mae = df[df['season'] == season]['error'].mean()
            calibration = df[df['season'] == season]['calibrated'].mean()
            print(f"  MAE: {mae:.2f} yards")
            print(f"  Calibration: {calibration:.1%}")

        return pd.DataFrame(self.results)

    def predict(self, game_id: str, player_id: str, stat_type: str) -> Dict:
        """Get prediction from API"""
        response = requests.post(
            f"{self.base_url}/predict",
            headers={"X-API-Key": self.api_key},
            json={
                "game_id": game_id,
                "player_id": player_id,
                "stat_type": stat_type
            },
            timeout=10
        )
        return response.json()

# Usage
backtest = MultiSeasonBacktest(api_key=os.getenv("NFL_API_KEY"))

# Backtest 2022-2024 seasons
results = backtest.backtest_multi_season(seasons=[2022, 2023, 2024])

# Analyze results
summary = results.groupby('season').agg({
    'error': 'mean',
    'calibrated': 'mean'
}).rename(columns={'error': 'MAE', 'calibrated': 'Calibration'})

print("\n" + "="*50)
print("MULTI-SEASON BACKTEST RESULTS")
print("="*50)
print(summary)

# Save results
results.to_csv("backtest_results.csv", index=False)
```

---

## Rate Limit Optimization

### Adaptive Rate Limiting

```python
import time
from datetime import datetime

class AdaptiveRateLimiter:
    """Automatically throttle requests based on rate limit headers"""

    def __init__(self, api_key: str, target_utilization: float = 0.8):
        self.api_key = api_key
        self.target_utilization = target_utilization  # Use 80% of available quota

        # Track rate limits
        self.limit = None
        self.remaining = None
        self.reset_time = None

    def predict(self, game_id: str, player_id: str, stat_type: str) -> Dict:
        """Make prediction with automatic throttling"""

        # Wait if approaching rate limit
        self._check_rate_limit()

        # Make request
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": self.api_key},
            json={
                "game_id": game_id,
                "player_id": player_id,
                "stat_type": stat_type
            }
        )

        # Update rate limit tracking
        self._update_rate_limits(response.headers)

        return response.json()

    def _update_rate_limits(self, headers: Dict):
        """Update rate limit state from response headers"""
        self.limit = int(headers.get("X-RateLimit-Limit", 0))
        self.remaining = int(headers.get("X-RateLimit-Remaining", 0))
        self.reset_time = int(headers.get("X-RateLimit-Reset", 0))

    def _check_rate_limit(self):
        """Throttle if approaching rate limit"""
        if self.limit is None or self.remaining is None:
            return  # First request, no throttling

        # Calculate current utilization
        utilization = 1 - (self.remaining / self.limit)

        if utilization > self.target_utilization:
            # Calculate sleep time to spread remaining requests
            time_until_reset = self.reset_time - int(time.time())
            requests_remaining = self.remaining

            if requests_remaining > 0 and time_until_reset > 0:
                sleep_time = time_until_reset / requests_remaining
                print(f"Throttling: {utilization:.1%} quota used, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

# Usage
limiter = AdaptiveRateLimiter(
    api_key=os.getenv("NFL_API_KEY"),
    target_utilization=0.8  # Use 80% of quota
)

for player in large_player_list:
    prediction = limiter.predict(
        game_id=player['game_id'],
        player_id=player['player_id'],
        stat_type='passing_yards'
    )
    # Process prediction
```

---

## Parallel Request Handling

### Thread Pool for Maximum Throughput

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def predict_parallel(
    predictions: List[Dict],
    max_workers: int = 10,
    timeout: int = 10
) -> List[Dict]:
    """
    Process predictions in parallel.

    WARNING: Can quickly exhaust rate limits. Use with caution.

    Args:
        predictions: List of prediction requests
        max_workers: Number of parallel threads
        timeout: Request timeout in seconds

    Returns:
        List of prediction results
    """
    results = []

    def make_prediction(pred: Dict) -> Dict:
        """Single prediction worker"""
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                headers={"X-API-Key": API_KEY},
                json=pred,
                timeout=timeout
            )

            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.json()
                }
            elif response.status_code == 429:
                # Rate limited
                retry_after = int(response.headers.get("Retry-After", 60))
                return {
                    "status": "rate_limited",
                    "retry_after": retry_after,
                    "request": pred
                }
            else:
                return {
                    "status": "error",
                    "error": response.json()
                }
        except Exception as e:
            return {
                "status": "exception",
                "error": str(e),
                "request": pred
            }

    # Submit all predictions
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(make_prediction, pred): pred for pred in predictions}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result['status'] == 'success':
                print(f"✓ {result['data']['player_id']}")
            elif result['status'] == 'rate_limited':
                print(f"⚠ Rate limited, retry in {result['retry_after']}s")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown')}")

    return results

# Usage
predictions = [
    {"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"},
    {"game_id": "2024_10_KC_LV", "player_id": "00-0034857", "stat_type": "rushing_yards"},
    # ... more predictions
]

results = predict_parallel(predictions, max_workers=10)

# Separate successes and failures
successes = [r for r in results if r['status'] == 'success']
failures = [r for r in results if r['status'] != 'success']

print(f"\nCompleted: {len(successes)}/{len(predictions)} ({len(successes)/len(predictions)*100:.1f}%)")
```

---

## Production-Grade Integration

### Complete Production Client

```python
import requests
import time
import logging
from typing import Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLAnalyticsClient:
    """
    Production-grade API client with:
    - Connection pooling
    - Automatic retries
    - Rate limit handling
    - Error handling
    - Request logging
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.nfl-analytics.com",
        timeout: int = 10,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

        # Create session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            backoff_factor=1,  # Exponential backoff: 1s, 2s, 4s
            respect_retry_after_header=True
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Default headers
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        })

    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise

    def predict(
        self,
        game_id: str,
        player_id: str,
        stat_type: str
    ) -> Dict:
        """
        Single prediction with comprehensive error handling.

        Raises:
            ValueError: Invalid input parameters
            requests.HTTPError: API error (401, 403, 429, 500, 503)
        """
        # Validate inputs
        if not game_id or not player_id or not stat_type:
            raise ValueError("game_id, player_id, and stat_type are required")

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={
                    "game_id": game_id,
                    "player_id": player_id,
                    "stat_type": stat_type
                },
                timeout=self.timeout
            )

            # Log request
            logger.info(
                f"Prediction: {player_id} {stat_type} @ {game_id} "
                f"(status: {response.status_code}, "
                f"latency: {response.elapsed.total_seconds():.3f}s)"
            )

            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Retry in {retry_after}s")
            elif e.response.status_code == 401:
                logger.error("Authentication failed. Check API key.")
            elif e.response.status_code == 503:
                logger.error("Service unavailable. Models not loaded.")
            raise

    def predict_batch(
        self,
        predictions: List[Dict],
        chunk_size: int = 100
    ) -> List[Dict]:
        """Batch predictions with automatic chunking"""
        all_results = []
        chunks = [predictions[i:i + chunk_size] for i in range(0, len(predictions), chunk_size)]

        logger.info(f"Processing {len(predictions)} predictions in {len(chunks)} batches")

        for i, chunk in enumerate(chunks):
            try:
                response = self.session.post(
                    f"{self.base_url}/predict/batch",
                    json={"predictions": chunk},
                    timeout=30
                )

                response.raise_for_status()
                batch_result = response.json()
                all_results.extend(batch_result['predictions'])

                logger.info(
                    f"Batch {i+1}/{len(chunks)}: "
                    f"{batch_result['successful_predictions']}/{batch_result['total_predictions']} successful"
                )

                # Rate limit protection
                if i < len(chunks) - 1:
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Batch {i+1}/{len(chunks)} failed: {e}")

        return all_results

    def close(self):
        """Close session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Usage
with NFLAnalyticsClient(api_key=os.getenv("NFL_API_KEY")) as client:
    # Health check
    health = client.health_check()
    print(f"API Status: {health['status']}")

    # Single prediction
    prediction = client.predict(
        game_id="2024_10_KC_LV",
        player_id="00-0033873",
        stat_type="passing_yards"
    )
    print(f"Prediction: {prediction['prediction']:.1f} yards")

    # Batch predictions
    batch = [
        {"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"},
        {"game_id": "2024_10_KC_LV", "player_id": "00-0034857", "stat_type": "rushing_yards"},
    ]
    results = client.predict_batch(batch)
    print(f"Batch completed: {len(results)} predictions")
```

---

## Performance Summary

### Optimization Techniques

| Technique | Speedup | Use Case |
|-----------|---------|----------|
| Batch API (100 items) | 10x | Multiple predictions for same game |
| Parallel batches (5 workers) | 2-5x | High rate limit, many predictions |
| Connection pooling | 1.5x | Eliminates connection overhead |
| Request caching (Redis) | 100x | Repeated predictions (same inputs) |
| Adaptive rate limiting | N/A | Prevents 429 errors |

### Latency Optimization Checklist

- [ ] Use batch endpoint for multiple predictions
- [ ] Enable connection pooling (HTTPAdapter)
- [ ] Set appropriate timeouts (10s for single, 30s for batch)
- [ ] Implement caching for repeated requests
- [ ] Use parallel requests only if rate limit permits
- [ ] Monitor rate limit headers
- [ ] Implement exponential backoff for 429 errors

---

## Support

**Questions?** See [Troubleshooting Guide](./troubleshooting_guide.md) or contact support@nfl-analytics.com

**Performance Issues?** See [Performance Tuning Guide](./performance_tuning.md)

**Rate Limits?** See [Rate Limiting Guide](./rate_limiting_guide.md)

---

**Last Updated:** November 15, 2025
**Status:** Production-Ready
**Next:** [Performance Tuning Guide](./performance_tuning.md)
