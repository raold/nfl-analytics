# API Webhooks Documentation

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Planned (Future Feature)

## Overview

Webhooks enable asynchronous, event-driven integration with the NFL Analytics API. When specific events occur (e.g., prediction complete, model updated, error), the API sends HTTP POST requests to your registered webhook URL.

**Status:** Webhook functionality is currently in development and planned for API v1.2 (Q1 2026). This documentation describes the proposed implementation.

## Table of Contents

1. [Webhook Overview](#webhook-overview)
2. [Event Types](#event-types)
3. [Webhook Registration](#webhook-registration)
4. [Payload Schemas](#payload-schemas)
5. [Security and Verification](#security-and-verification)
6. [Retry Logic](#retry-logic)
7. [Testing Webhooks](#testing-webhooks)
8. [Best Practices](#best-practices)

---

## Webhook Overview

### How Webhooks Work

1. **Register** your webhook URL via API or dashboard
2. **Events occur** (e.g., batch prediction completes)
3. **API sends HTTP POST** to your webhook URL with event payload
4. **Your server responds** with 200 OK to acknowledge receipt
5. **Retry logic** if your server doesn't respond or returns error

### Use Cases

- **Async Batch Processing:** Submit large batch (1,000+ predictions), receive results via webhook when ready
- **Model Updates:** Get notified when models are refreshed or updated
- **Error Monitoring:** Receive alerts for prediction failures or service issues
- **Real-Time Odds:** Trigger betting logic when new predictions available

### Benefits

- **Non-blocking:** Submit request, continue other work, receive results later
- **Scalable:** Process large batches without keeping connection open
- **Real-time:** Immediate notification when events occur
- **Reliable:** Automatic retries ensure delivery

---

## Event Types

### Supported Events

| Event Type | Trigger | Payload Size | Frequency |
|------------|---------|--------------|-----------|
| `prediction.complete` | Batch prediction finished | Large (all predictions) | Per batch |
| `prediction.failed` | Batch prediction failed | Small (error details) | On error |
| `model.updated` | Model refresh completed | Medium (model info) | Rare (weekly) |
| `model.failed` | Model refresh failed | Small (error details) | On error |
| `error.occurred` | Service error | Small (error details) | On error |

### Event Filtering

Subscribe to specific event types when registering webhook:

```json
{
  "url": "https://your-domain.com/webhook",
  "events": ["prediction.complete", "prediction.failed"],
  "secret": "your_webhook_secret"
}
```

---

## Webhook Registration

### Register Webhook (Future Endpoint)

**Endpoint:** `POST /webhooks`

**Request:**
```json
{
  "url": "https://your-domain.com/webhook/nfl-analytics",
  "events": ["prediction.complete", "prediction.failed"],
  "secret": "your_webhook_secret_here",
  "description": "Production webhook for batch predictions"
}
```

**Response:**
```json
{
  "webhook_id": "wh_1234567890abcdef",
  "url": "https://your-domain.com/webhook/nfl-analytics",
  "events": ["prediction.complete", "prediction.failed"],
  "created_at": "2025-11-15T12:00:00Z",
  "status": "active",
  "secret_preview": "your_web...re"
}
```

### List Webhooks

**Endpoint:** `GET /webhooks`

**Response:**
```json
{
  "webhooks": [
    {
      "webhook_id": "wh_1234567890abcdef",
      "url": "https://your-domain.com/webhook/nfl-analytics",
      "events": ["prediction.complete", "prediction.failed"],
      "created_at": "2025-11-15T12:00:00Z",
      "status": "active",
      "last_triggered": "2025-11-15T14:30:00Z",
      "success_count": 125,
      "failure_count": 2
    }
  ]
}
```

### Update Webhook

**Endpoint:** `PATCH /webhooks/{webhook_id}`

**Request:**
```json
{
  "url": "https://new-domain.com/webhook",
  "events": ["prediction.complete"],
  "status": "active"
}
```

### Delete Webhook

**Endpoint:** `DELETE /webhooks/{webhook_id}`

**Response:**
```json
{
  "status": "deleted",
  "webhook_id": "wh_1234567890abcdef"
}
```

---

## Payload Schemas

### prediction.complete

Sent when batch prediction finishes successfully.

**Headers:**
```http
POST /webhook/nfl-analytics HTTP/1.1
Host: your-domain.com
Content-Type: application/json
X-Webhook-Event: prediction.complete
X-Webhook-ID: wh_1234567890abcdef
X-Webhook-Signature: sha256=abcdef1234567890...
```

**Payload:**
```json
{
  "event": "prediction.complete",
  "event_id": "evt_1234567890abcdef",
  "timestamp": "2025-11-15T12:00:00Z",
  "data": {
    "job_id": "job_abcdef1234567890",
    "status": "completed",
    "submitted_at": "2025-11-15T11:00:00Z",
    "completed_at": "2025-11-15T12:00:00Z",
    "duration_seconds": 3600,
    "results": {
      "total_predictions": 1000,
      "successful_predictions": 985,
      "failed_predictions": 15,
      "predictions": [
        {
          "game_id": "2024_10_KC_LV",
          "player_id": "00-0033873",
          "stat_type": "passing_yards",
          "prediction": 285.3,
          "confidence": 0.78,
          "lower_bound": 215.1,
          "upper_bound": 355.5,
          "timestamp": "2025-11-15T12:00:00Z",
          "status": "success"
        }
        // ... 999 more predictions
      ]
    }
  }
}
```

### prediction.failed

Sent when batch prediction fails.

**Payload:**
```json
{
  "event": "prediction.failed",
  "event_id": "evt_1234567890abcdef",
  "timestamp": "2025-11-15T12:00:00Z",
  "data": {
    "job_id": "job_abcdef1234567890",
    "status": "failed",
    "submitted_at": "2025-11-15T11:00:00Z",
    "failed_at": "2025-11-15T11:05:00Z",
    "error": {
      "code": "SERVICE_UNAVAILABLE",
      "message": "Model not loaded: state_space_v2",
      "details": {
        "missing_models": ["state_space_v2", "hierarchical_v2"]
      }
    }
  }
}
```

### model.updated

Sent when models are refreshed (e.g., `POST /admin/refresh`).

**Payload:**
```json
{
  "event": "model.updated",
  "event_id": "evt_1234567890abcdef",
  "timestamp": "2025-11-15T12:00:00Z",
  "data": {
    "refresh_id": "refresh_abcdef1234567890",
    "status": "completed",
    "models_reloaded": 4,
    "models_failed": 0,
    "reload_time_seconds": 25.3,
    "models": [
      {
        "model_name": "state_space_v2",
        "status": "loaded",
        "version": "2.0",
        "reload_time_seconds": 8.2
      },
      {
        "model_name": "hierarchical_v2",
        "status": "loaded",
        "version": "2.0",
        "reload_time_seconds": 10.1
      },
      {
        "model_name": "hybrid_xgboost",
        "status": "loaded",
        "version": "4.0",
        "reload_time_seconds": 2.5
      },
      {
        "model_name": "informative_priors_v2",
        "status": "loaded",
        "version": "2.0",
        "reload_time_seconds": 4.5
      }
    ]
  }
}
```

### error.occurred

Sent when service errors occur.

**Payload:**
```json
{
  "event": "error.occurred",
  "event_id": "evt_1234567890abcdef",
  "timestamp": "2025-11-15T12:00:00Z",
  "data": {
    "error_type": "database_connection_lost",
    "severity": "critical",
    "message": "PostgreSQL connection lost",
    "affected_endpoints": ["/predict", "/predict/batch"],
    "details": {
      "database": "devdb01",
      "last_successful_query": "2025-11-15T11:58:00Z"
    }
  }
}
```

---

## Security and Verification

### HMAC Signature Verification

All webhook payloads include `X-Webhook-Signature` header for verification.

**Algorithm:** HMAC-SHA256 using webhook secret

```python
import hmac
import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)

WEBHOOK_SECRET = "your_webhook_secret_here"

@app.route("/webhook/nfl-analytics", methods=["POST"])
def receive_webhook():
    """Verify and process webhook"""

    # Get signature from header
    signature_header = request.headers.get("X-Webhook-Signature")
    if not signature_header:
        return jsonify({"error": "Missing signature"}), 401

    # Extract signature (format: "sha256=...")
    signature = signature_header.replace("sha256=", "")

    # Get request body
    payload = request.data

    # Compute expected signature
    expected_signature = hmac.new(
        key=WEBHOOK_SECRET.encode('utf-8'),
        msg=payload,
        digestmod=hashlib.sha256
    ).hexdigest()

    # Verify signature
    if not hmac.compare_digest(signature, expected_signature):
        return jsonify({"error": "Invalid signature"}), 403

    # Signature valid, process webhook
    event = request.json
    process_webhook_event(event)

    return jsonify({"status": "received"}), 200

def process_webhook_event(event: dict):
    """Process webhook event"""
    event_type = event['event']

    if event_type == "prediction.complete":
        handle_prediction_complete(event['data'])
    elif event_type == "prediction.failed":
        handle_prediction_failed(event['data'])
    elif event_type == "model.updated":
        handle_model_updated(event['data'])
    else:
        print(f"Unknown event type: {event_type}")
```

### Timestamp Verification (Replay Attack Prevention)

```python
from datetime import datetime, timedelta

MAX_WEBHOOK_AGE = 300  # 5 minutes

def verify_timestamp(event: dict) -> bool:
    """Verify webhook is recent (prevent replay attacks)"""
    timestamp_str = event['timestamp']
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

    age = datetime.now(timestamp.tzinfo) - timestamp

    if age > timedelta(seconds=MAX_WEBHOOK_AGE):
        print(f"Webhook too old: {age.total_seconds()}s")
        return False

    return True

@app.route("/webhook/nfl-analytics", methods=["POST"])
def receive_webhook():
    # ... signature verification ...

    # Verify timestamp
    event = request.json
    if not verify_timestamp(event):
        return jsonify({"error": "Webhook expired"}), 403

    # Process webhook
    process_webhook_event(event)
    return jsonify({"status": "received"}), 200
```

---

## Retry Logic

### Retry Behavior

If your webhook endpoint doesn't respond with 200 OK, the API will retry:

**Retry Schedule:**
- Attempt 1: Immediate
- Attempt 2: 1 minute later
- Attempt 3: 5 minutes later
- Attempt 4: 15 minutes later
- Attempt 5: 1 hour later
- **Max retries:** 5

**Failed Delivery:**
After 5 failed attempts, webhook is marked as failed and you receive email notification.

### Exponential Backoff

```python
# Server-side retry logic (for reference)
RETRY_DELAYS = [0, 60, 300, 900, 3600]  # seconds

for attempt in range(len(RETRY_DELAYS)):
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={
                "X-Webhook-Event": event_type,
                "X-Webhook-Signature": signature
            },
            timeout=30
        )

        if response.status_code == 200:
            # Success
            break
    except Exception as e:
        if attempt < len(RETRY_DELAYS) - 1:
            time.sleep(RETRY_DELAYS[attempt + 1])
        else:
            # All retries failed
            send_failure_notification()
```

### Idempotency

**Important:** Your webhook handler must be idempotent (safe to receive same event multiple times).

Use `event_id` to deduplicate:

```python
import redis

redis_client = redis.Redis()

def is_duplicate_event(event_id: str) -> bool:
    """Check if event already processed"""
    key = f"webhook_event:{event_id}"

    # Try to set key (returns False if already exists)
    if redis_client.setnx(key, "1"):
        # First time seeing this event
        redis_client.expire(key, 86400)  # Expire after 24 hours
        return False
    else:
        # Already processed
        return True

@app.route("/webhook/nfl-analytics", methods=["POST"])
def receive_webhook():
    event = request.json
    event_id = event['event_id']

    # Check for duplicate
    if is_duplicate_event(event_id):
        print(f"Duplicate event: {event_id}")
        return jsonify({"status": "duplicate"}), 200

    # Process event
    process_webhook_event(event)
    return jsonify({"status": "received"}), 200
```

---

## Testing Webhooks

### Local Development with ngrok

Use ngrok to test webhooks on localhost:

```bash
# Install ngrok
brew install ngrok  # macOS
# or download from https://ngrok.com

# Start your webhook server
python webhook_server.py  # Runs on http://localhost:5000

# Expose localhost to internet
ngrok http 5000
# Output: https://abc123.ngrok.io -> http://localhost:5000

# Register webhook URL (use ngrok URL)
curl -X POST "https://api.nfl-analytics.com/webhooks" \
  -H "X-API-Key: $NFL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://abc123.ngrok.io/webhook/nfl-analytics",
    "events": ["prediction.complete"],
    "secret": "test_secret"
  }'
```

### Webhook Testing Tool

```python
import requests
import hmac
import hashlib
import json

def send_test_webhook(
    webhook_url: str,
    event_type: str,
    payload: dict,
    secret: str
):
    """Send test webhook (for local testing)"""

    # Compute signature
    payload_bytes = json.dumps(payload).encode('utf-8')
    signature = hmac.new(
        key=secret.encode('utf-8'),
        msg=payload_bytes,
        digestmod=hashlib.sha256
    ).hexdigest()

    # Send webhook
    response = requests.post(
        webhook_url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-Webhook-Event": event_type,
            "X-Webhook-Signature": f"sha256={signature}"
        }
    )

    print(f"Response: {response.status_code}")
    print(f"Body: {response.text}")

# Usage
test_payload = {
    "event": "prediction.complete",
    "event_id": "evt_test123",
    "timestamp": "2025-11-15T12:00:00Z",
    "data": {
        "job_id": "job_test123",
        "status": "completed",
        "results": {
            "total_predictions": 10,
            "successful_predictions": 10,
            "predictions": []
        }
    }
}

send_test_webhook(
    webhook_url="http://localhost:5000/webhook/nfl-analytics",
    event_type="prediction.complete",
    payload=test_payload,
    secret="test_secret"
)
```

### Mock Webhook Server

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = "test_secret"

@app.route("/webhook/nfl-analytics", methods=["POST"])
def webhook_handler():
    """Test webhook receiver"""

    # Verify signature
    signature_header = request.headers.get("X-Webhook-Signature", "")
    signature = signature_header.replace("sha256=", "")

    payload = request.data
    expected_signature = hmac.new(
        key=WEBHOOK_SECRET.encode('utf-8'),
        msg=payload,
        digestmod=hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_signature):
        print("❌ Invalid signature")
        return jsonify({"error": "Invalid signature"}), 403

    # Process webhook
    event = request.json
    event_type = event['event']

    print(f"✅ Received: {event_type}")
    print(f"   Event ID: {event['event_id']}")

    if event_type == "prediction.complete":
        total = event['data']['results']['total_predictions']
        successful = event['data']['results']['successful_predictions']
        print(f"   Predictions: {successful}/{total} successful")

    return jsonify({"status": "received"}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

---

## Best Practices

### 1. Respond Quickly (< 30 seconds)

```python
# ❌ BAD: Process webhook synchronously
@app.route("/webhook", methods=["POST"])
def webhook_handler():
    event = request.json

    # This takes 5 minutes (timeout!)
    process_all_predictions(event['data']['results']['predictions'])

    return jsonify({"status": "received"}), 200

# ✅ GOOD: Queue for async processing
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@app.route("/webhook", methods=["POST"])
def webhook_handler():
    event = request.json

    # Queue for async processing (returns immediately)
    celery.send_task('process_webhook', args=[event])

    return jsonify({"status": "queued"}), 200

@celery.task
def process_webhook(event):
    # Process asynchronously
    process_all_predictions(event['data']['results']['predictions'])
```

### 2. Handle Duplicate Events

Always check `event_id` to prevent duplicate processing.

### 3. Log Everything

```python
import logging

logger = logging.getLogger(__name__)

@app.route("/webhook", methods=["POST"])
def webhook_handler():
    event = request.json

    logger.info(
        f"Webhook received: {event['event']} (id: {event['event_id']})",
        extra={
            "event_id": event['event_id'],
            "event_type": event['event'],
            "timestamp": event['timestamp']
        }
    )

    # Process webhook
    try:
        process_webhook_event(event)
        logger.info(f"Webhook processed: {event['event_id']}")
    except Exception as e:
        logger.error(f"Webhook processing failed: {event['event_id']}", exc_info=e)
        return jsonify({"error": "Processing failed"}), 500

    return jsonify({"status": "received"}), 200
```

### 4. Monitor Webhook Health

```python
from prometheus_client import Counter, Histogram

webhook_received = Counter(
    'webhooks_received_total',
    'Total webhooks received',
    ['event_type', 'status']
)

webhook_processing_time = Histogram(
    'webhook_processing_seconds',
    'Webhook processing time',
    ['event_type']
)

@app.route("/webhook", methods=["POST"])
def webhook_handler():
    event = request.json
    event_type = event['event']

    start = time.time()

    try:
        process_webhook_event(event)
        webhook_received.labels(event_type=event_type, status='success').inc()
        status_code = 200
    except Exception:
        webhook_received.labels(event_type=event_type, status='error').inc()
        status_code = 500

    webhook_processing_time.labels(event_type=event_type).observe(time.time() - start)

    return jsonify({"status": "received"}), status_code
```

### 5. Implement Circuit Breaker

```python
from datetime import datetime, timedelta

class WebhookCircuitBreaker:
    """Prevent cascading failures"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == "open":
            # Check if timeout expired
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker open")

        try:
            result = func(*args, **kwargs)

            # Success, reset failures
            if self.state == "half_open":
                self.state = "closed"
            self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.now()

            if self.failures >= self.failure_threshold:
                self.state = "open"

            raise

# Usage
circuit_breaker = WebhookCircuitBreaker()

@app.route("/webhook", methods=["POST"])
def webhook_handler():
    event = request.json

    try:
        circuit_breaker.call(process_webhook_event, event)
        return jsonify({"status": "received"}), 200
    except Exception as e:
        logger.error(f"Circuit breaker triggered: {e}")
        return jsonify({"error": "Service degraded"}), 503
```

---

## Support

**Questions?** Contact support@nfl-analytics.com
**Webhook Debugging:** Include `event_id` and `webhook_id` in support requests

---

**Last Updated:** November 15, 2025
**Status:** Planned (v1.2)
**Next:** [Migration Guide](./migration_guide.md)
