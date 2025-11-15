# Security Best Practices

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

The NFL Analytics Prediction API handles sensitive betting predictions. Follow these security best practices to protect your API keys and ensure secure communication.

## Table of Contents

1. [API Key Security](#api-key-security)
2. [Environment Variables](#environment-variables)
3. [Key Rotation](#key-rotation)
4. [Monitoring and Alerts](#monitoring-and-alerts)
5. [Reporting Security Issues](#reporting-security-issues)
6. [Security Checklist](#security-checklist)

---

## API Key Security

### Never Commit API Keys to Git

**❌ WRONG:**

```python
# config.py
API_KEY = "dev_test_key_12345678901234567890123456789012"  # NEVER DO THIS
```

```javascript
// config.js
const API_KEY = "dev_test_key_12345678901234567890123456789012";  // NEVER DO THIS
```

**✅ CORRECT:**

```python
# config.py
import os
API_KEY = os.getenv("NFL_API_KEY")  # Load from environment
```

```javascript
// config.js
const API_KEY = process.env.NFL_API_KEY;  // Load from environment
```

### Add Secrets to .gitignore

**.gitignore:**
```bash
# Environment variables
.env
.env.local
.env.*.local

# Secret management
secrets/
secrets.yaml
credentials.json

# API keys (just in case)
*api_key*
*apikey*
*.pem
*.key
```

### Use Secret Management Systems

**Recommended Tools:**

1. **1Password** (Team/Business)
   ```bash
   # Install 1Password CLI
   brew install --cask 1password-cli

   # Load secret
   export NFL_API_KEY=$(op read "op://Vault/NFL Analytics API/api_key")
   ```

2. **HashiCorp Vault**
   ```bash
   # Store secret
   vault kv put secret/nfl-api api_key="your_key_here"

   # Retrieve secret
   export NFL_API_KEY=$(vault kv get -field=api_key secret/nfl-api)
   ```

3. **AWS Secrets Manager**
   ```python
   import boto3
   import json

   def get_api_key():
       client = boto3.client('secretsmanager', region_name='us-east-1')
       response = client.get_secret_value(SecretId='nfl-api-key')
       secret = json.loads(response['SecretString'])
       return secret['api_key']

   API_KEY = get_api_key()
   ```

4. **Google Cloud Secret Manager**
   ```python
   from google.cloud import secretmanager

   def get_api_key():
       client = secretmanager.SecretManagerServiceClient()
       name = "projects/PROJECT_ID/secrets/nfl-api-key/versions/latest"
       response = client.access_secret_version(request={"name": name})
       return response.payload.data.decode("UTF-8")

   API_KEY = get_api_key()
   ```

### Never Log API Keys

**❌ WRONG:**

```python
logger.info(f"Using API key: {API_KEY}")  # NEVER DO THIS
print(f"Headers: {headers}")  # Leaks API key in X-API-Key header
```

**✅ CORRECT:**

```python
# Mask API key in logs
masked_key = API_KEY[:8] + "..." if API_KEY else "None"
logger.info(f"Using API key: {masked_key}")

# Redact sensitive headers
safe_headers = {k: v if k != "X-API-Key" else "***REDACTED***"
                for k, v in headers.items()}
logger.debug(f"Headers: {safe_headers}")
```

### Never Use API Keys in URLs

**❌ WRONG:**

```bash
# API key in query parameter (logged in access logs)
curl "https://api.nfl-analytics.com/predict?api_key=your_key_here"
```

**✅ CORRECT:**

```bash
# API key in header (not logged)
curl "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: your_key_here"
```

### Never Use API Keys in Frontend Code

**❌ WRONG:**

```javascript
// frontend/src/api.js
const API_KEY = "dev_test_key_12345678901234567890123456789012";  // EXPOSED TO USERS

fetch("https://api.nfl-analytics.com/predict", {
  headers: {
    "X-API-Key": API_KEY  // Anyone can see this in browser DevTools
  }
});
```

**✅ CORRECT:**

```javascript
// frontend/src/api.js
// API calls go through backend proxy (no API key in frontend)
fetch("/api/predict", {  // Backend endpoint
  method: "POST",
  body: JSON.stringify(payload)
});
```

```python
# backend/api_proxy.py
from fastapi import APIRouter
import requests
import os

router = APIRouter()
NFL_API_KEY = os.getenv("NFL_API_KEY")  # Backend only

@router.post("/api/predict")
async def proxy_predict(payload: dict):
    """Proxy to NFL Analytics API (hides API key from frontend)"""
    response = requests.post(
        "https://api.nfl-analytics.com/predict",
        headers={"X-API-Key": NFL_API_KEY},
        json=payload
    )
    return response.json()
```

---

## Environment Variables

### Setup for Development

**.env file** (add to .gitignore):

```bash
# NFL Analytics API Configuration
NFL_API_KEY="your_development_api_key_here"
NFL_API_BASE_URL="http://localhost:8000"
NFL_API_TIMEOUT=10

# Database (if needed)
DATABASE_URL="postgresql://user:pass@localhost:5432/db"

# Redis (if caching)
REDIS_URL="redis://localhost:6379/0"

# Environment
ENVIRONMENT="development"
DEBUG="true"
```

**.env.example** (commit to git as template):

```bash
# NFL Analytics API Configuration
NFL_API_KEY="your_api_key_here"
NFL_API_BASE_URL="https://api.nfl-analytics.com"
NFL_API_TIMEOUT=10

# Database
DATABASE_URL="postgresql://user:pass@host:port/db"

# Redis
REDIS_URL="redis://host:port/db"

# Environment
ENVIRONMENT="production"
DEBUG="false"
```

### Loading Environment Variables

#### Python (python-dotenv)

```python
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access variables
API_KEY = os.getenv("NFL_API_KEY")
BASE_URL = os.getenv("NFL_API_BASE_URL", "https://api.nfl-analytics.com")
TIMEOUT = int(os.getenv("NFL_API_TIMEOUT", 10))

# Validate required variables
if not API_KEY:
    raise ValueError("NFL_API_KEY environment variable is required")
```

#### R (dotenv)

```r
library(dotenv)

# Load .env file
load_dot_env()

# Access variables
API_KEY <- Sys.getenv("NFL_API_KEY")
BASE_URL <- Sys.getenv("NFL_API_BASE_URL", "https://api.nfl-analytics.com")

# Validate required variables
if (API_KEY == "") {
  stop("NFL_API_KEY environment variable is required")
}
```

#### Node.js (dotenv)

```javascript
require('dotenv').config();

// Access variables
const API_KEY = process.env.NFL_API_KEY;
const BASE_URL = process.env.NFL_API_BASE_URL || "https://api.nfl-analytics.com";

// Validate required variables
if (!API_KEY) {
  throw new Error("NFL_API_KEY environment variable is required");
}
```

### Separate Keys Per Environment

**Development:**
```bash
NFL_API_KEY="dev_test_key_12345..."
NFL_API_BASE_URL="http://localhost:8000"
```

**Staging:**
```bash
NFL_API_KEY="staging_key_67890..."
NFL_API_BASE_URL="https://staging.nfl-analytics.com"
```

**Production:**
```bash
NFL_API_KEY="prod_key_abcdef..."
NFL_API_BASE_URL="https://api.nfl-analytics.com"
```

Benefits:
- Isolate environments (dev errors don't affect prod)
- Track usage per environment
- Revoke staging key without affecting production

---

## Key Rotation

### Rotation Schedule

**Recommended Frequency:**
- **Production:** Every 90 days
- **Staging:** Every 180 days
- **Development:** Annually (or on developer turnover)

### Rotation Process

**Step 1: Request New Key**

Email: support@nfl-analytics.com

```
Subject: API Key Rotation Request

Current Key Prefix: prod_key_
Environment: Production
Reason: Scheduled 90-day rotation
Requested Expiration: 2026-02-15
```

**Step 2: Test New Key in Staging**

```bash
# Update staging environment
export NFL_API_KEY="new_staging_key_xyz..."

# Run integration tests
pytest tests/integration/test_api.py

# Verify predictions work
python scripts/verify_api.py
```

**Step 3: Deploy to Production**

```bash
# Update production secret (example: AWS Secrets Manager)
aws secretsmanager update-secret \
  --secret-id nfl-api-key \
  --secret-string '{"api_key":"new_prod_key_123..."}'

# Restart services to pick up new secret
kubectl rollout restart deployment/api-consumer
```

**Step 4: Monitor for Errors**

```bash
# Check logs for auth errors
kubectl logs -f deployment/api-consumer | grep "401\|403"

# Monitor error rate in Grafana
# Alert if error rate > 1%
```

**Step 5: Revoke Old Key (Grace Period)**

Wait 7 days before revoking old key:

```
Subject: Revoke API Key

Key Prefix: old_prod_key_
Reason: Rotated to new_prod_key_123
Confirmed: All services migrated to new key
```

### Automated Rotation Reminder

```python
import os
from datetime import datetime, timedelta

def check_key_age():
    """Alert if API key needs rotation"""

    key_created_at = os.getenv("NFL_API_KEY_CREATED_AT")
    if not key_created_at:
        print("⚠️  Warning: API key creation date not set")
        return

    created = datetime.fromisoformat(key_created_at)
    age_days = (datetime.utcnow() - created).days

    ROTATION_THRESHOLD = 90  # days

    if age_days >= ROTATION_THRESHOLD:
        print(f"🚨 URGENT: API key is {age_days} days old (threshold: {ROTATION_THRESHOLD})")
        print("   Action: Request key rotation from support@nfl-analytics.com")
    elif age_days >= ROTATION_THRESHOLD - 14:
        print(f"⚠️  Warning: API key rotation due in {ROTATION_THRESHOLD - age_days} days")
        print("   Action: Schedule key rotation")
    else:
        print(f"✅ API key age: {age_days} days (next rotation in {ROTATION_THRESHOLD - age_days} days)")

# Run on startup
check_key_age()
```

---

## Monitoring and Alerts

### Monitor for Suspicious Activity

**Set up alerts for:**

1. **Unusual Request Volume**
   - Spike in requests (>200% of normal)
   - Sustained high usage (>80% of rate limit)

2. **Failed Authentication**
   - Multiple 401/403 errors in short period
   - Could indicate compromised key being tested

3. **Unusual Access Patterns**
   - Requests from unexpected IP ranges
   - Requests during off-hours (if not expected)
   - Requests to unusual endpoints

4. **Rate Limit Violations**
   - Frequent 429 errors
   - Could indicate runaway script or attack

### Example: Grafana Alert

```yaml
# grafana/alerts/api_auth_failures.yaml
alerts:
  - name: High Auth Failure Rate
    condition: |
      rate(auth_failures_total[5m]) > 10
    annotations:
      summary: "High API authentication failure rate"
      description: "{{ $value }} auth failures/min in last 5 minutes"
    for: 5m
    labels:
      severity: warning
    notifications:
      - email: security@company.com
      - slack: "#alerts-security"
```

### Example: Python Monitoring Script

```python
import requests
from datetime import datetime

def check_api_health():
    """Monitor API key health and usage"""

    try:
        response = requests.get(
            "https://api.nfl-analytics.com/models",
            headers={"X-API-Key": API_KEY},
            timeout=5
        )

        # Check rate limit
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        limit = int(response.headers.get("X-RateLimit-Limit", 100))

        usage_pct = ((limit - remaining) / limit) * 100

        # Alert if high usage
        if usage_pct > 80:
            send_alert(
                severity="warning",
                message=f"High API usage: {usage_pct:.1f}% of quota used",
                details={
                    "remaining": remaining,
                    "limit": limit,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        # Check for auth errors
        if response.status_code in [401, 403]:
            send_alert(
                severity="critical",
                message=f"API authentication failed: {response.status_code}",
                details={
                    "error": response.json(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        return True

    except Exception as e:
        send_alert(
            severity="error",
            message=f"API health check failed: {str(e)}",
            details={"timestamp": datetime.utcnow().isoformat()}
        )
        return False

# Run every 5 minutes
```

---

## Reporting Security Issues

### What to Report

Report immediately if:

- ✅ API key committed to public repository
- ✅ API key leaked in logs or monitoring
- ✅ Suspicious activity on your API key
- ✅ Unauthorized access to systems with API key
- ✅ Data breach affecting systems with API key

### How to Report

**For CRITICAL security issues (leaked keys):**

**Email:** security@nfl-analytics.com
**Subject:** URGENT: Security Incident - API Key Compromised

**Include:**
1. **Incident Description:** What happened
2. **Key Prefix:** First 8 characters of compromised key
3. **Exposure Scope:** Where key was leaked (public repo, logs, etc.)
4. **Timeline:** When leak occurred
5. **Mitigation Actions:** What you've done (revoke commit, delete logs)

**Response Time:** Within 1 hour during business hours

**For Non-Critical Issues:**

**Email:** support@nfl-analytics.com
**Subject:** Security Concern - API Usage

**Response Time:** Within 24 hours

### Incident Response Process

1. **Immediate Key Revocation** (within 1 hour)
   - Compromised key disabled
   - Temporary key issued

2. **Investigation** (1-2 days)
   - Review access logs
   - Identify unauthorized usage
   - Assess impact

3. **Remediation** (1-2 days)
   - Issue new permanent key
   - Update security controls
   - Document lessons learned

4. **Report** (within 1 week)
   - Incident summary
   - Impact assessment
   - Recommendations

---

## Security Checklist

### Development

- [ ] API keys loaded from environment variables
- [ ] `.env` file added to `.gitignore`
- [ ] No API keys in code (checked via `git grep`)
- [ ] API keys masked in logs
- [ ] Separate keys for dev/staging/prod
- [ ] Integration tests use test API keys

### Pre-Production

- [ ] Production API key provisioned
- [ ] Key stored in secret management system
- [ ] Key not in version control (verified)
- [ ] Deployment scripts tested in staging
- [ ] Monitoring and alerts configured
- [ ] Incident response plan documented

### Production

- [ ] API key rotation schedule set (90 days)
- [ ] Rate limit monitoring enabled
- [ ] Auth failure alerts configured
- [ ] Usage tracking dashboard created
- [ ] Backup API key provisioned
- [ ] Key expiration reminder configured

### Ongoing

- [ ] Monthly usage review
- [ ] Quarterly security audit
- [ ] Annual penetration testing
- [ ] Review access logs weekly
- [ ] Update dependencies monthly

### Verification Commands

```bash
# Check for hardcoded API keys in code
git grep -i "api_key.*=" | grep -v "os.getenv\|process.env"

# Check for secrets in git history
git log -S "api_key" --all

# Verify .env is in .gitignore
grep -q "^\.env$" .gitignore && echo "✅ .env ignored" || echo "❌ .env NOT ignored"

# Check for API keys in environment
env | grep -i "api_key" && echo "⚠️  API key in environment" || echo "✅ No API keys in env output"
```

---

## Resources

**Documentation:**
- [Authentication Guide](authentication_guide.md)
- [Rate Limiting Guide](rate_limiting_guide.md)
- [OpenAPI Specification](openapi.yaml)

**Tools:**
- [1Password](https://1password.com/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)
- [git-secrets](https://github.com/awslabs/git-secrets) - Prevent committing secrets

**Support:**
- **Security Issues:** security@nfl-analytics.com
- **General Support:** support@nfl-analytics.com
- **Status Page:** https://status.nfl-analytics.com

---

**Remember:** Treat API keys like passwords. When in doubt, rotate the key.
