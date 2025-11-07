# Causal Inference Framework: 4-Phase Implementation Plan

## Executive Summary

This document outlines the remaining work to complete the causal inference framework for shock event modeling in NFL betting markets. Phase 1 (Complete) established the foundation with BNN calibration experiments, treatment effect estimation, and preliminary betting validation. Phases 2-4 focus on fixing identified issues, automating detection, and deploying to production.

**Status:** Phase 1 Complete (Oct 2024) | Phases 2-4 Planned (Q1-Q3 2025)

---

## Phase 1: Framework Implementation (✅ COMPLETE - October 2024)

### Deliverables Completed

1. **Causal Inference Module** (4,500 lines Python)
   - Panel data constructor
   - Treatment definitions (injuries, coaching, trades, weather)
   - Confounder identification (backdoor criterion)
   - Synthetic control method
   - Difference-in-differences estimator
   - Structural causal models (DAGs)
   - Model integration layer
   - Validation framework

2. **BNN Prior Sensitivity Analysis**
   - Trained 4 models with noise σ ∈ {0.5, 0.7, 1.0, 1.5}
   - Result: Prior insensitivity confirmed (26% calibration regardless of σ)
   - Conclusion: Under-calibration is architectural, not prior-related
   - Models saved: `models/bayesian/bnn_rushing_sigma{0.5,0.7,1.0,1.5}.pkl`

3. **Treatment Effect Estimation**
   - Injuries: -15.0 yards (SE: 8.0), p=0.042
   - Coaching changes: +5.1 points first 3 weeks (95% CI: [1.8, 8.4])
   - Trades: -8.0 yards for 4 weeks (95% CI: [-12.3, -3.7])
   - Weather: -3.2 points for precipitation >5mm (minimal impact)

4. **Betting Performance Backtest**
   - Sample: 147 shock events (2020-2024)
   - Win rate: 56.8% vs 53.2% baseline (+3.6 pp)
   - ROI: +5.1% vs +1.8% baseline (+3.3 pp)
   - MAE: 16.2 yards vs 18.7 baseline (13.4% improvement)

5. **Dissertation Integration**
   - Chapter 8 section: `section_causal_inference.tex` (51 sections, 3,500 words)
   - 3 LaTeX tables generated and integrated
   - 1 causal DAG figure created
   - Full documentation of negative results (BNN calibration, sample size limits)

### Key Findings

**Positive:**
- Framework is methodologically sound (synthetic control, DiD, DAGs)
- Treatment effects are statistically significant and economically meaningful
- Preliminary betting validation shows promise (+3.3 pp ROI)
- Negative results (weather, sample size) prevent premature deployment

**Challenges Identified:**
- BNN under-calibration persists across all priors (26% vs 90% target)
- Small sample size (n=147 shocks over 5 years) limits statistical power
- Manual shock detection is labor-intensive and error-prone
- Integration with baseline models is ad-hoc, not systematic

---

## Phase 2: Fix BNN Calibration & Expand Database (Q1 2025)

### Objective
Address the BNN under-calibration issue identified in Phase 1 and expand the shock event database to improve statistical power.

### 2.1 Improved BNN Architecture

**Current Issue:**
- 2-layer MLP achieves good MAE (18.7 yards) but poor calibration (26% vs 90% target)
- Prior sensitivity analysis ruled out tight priors as cause
- Likely causes: Model misspecification, feature limitations, or aleatoric uncertainty

**Approaches to Test:**

#### Option A: Deeper Networks
```python
# Current: 2 layers (input → hidden → output)
# Proposed: 4-5 layers with skip connections
model = pm.Model()
with model:
    # Layer 1: 64 units
    W1 = pm.Normal('W1', mu=0, sigma=1, shape=(n_features, 64))
    b1 = pm.Normal('b1', mu=0, sigma=1, shape=64)
    h1 = pm.math.tanh(X @ W1 + b1)

    # Layer 2: 32 units
    W2 = pm.Normal('W2', mu=0, sigma=1, shape=(64, 32))
    b2 = pm.Normal('b2', mu=0, sigma=1, shape=32)
    h2 = pm.math.tanh(h1 @ W2 + b2)

    # Skip connection
    h2_skip = pm.math.concatenate([h1, h2], axis=1)

    # Output layer with increased noise
    W_out = pm.Normal('W_out', mu=0, sigma=1, shape=(96, 1))
    mu = h2_skip @ W_out

    # Key change: Learned noise per sample
    log_sigma = pm.Normal('log_sigma', mu=2, sigma=0.5)
    sigma = pm.Deterministic('sigma', pm.math.exp(log_sigma))

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
```

**Expected improvement:** Better capacity to model complex distributions → improved calibration

#### Option B: Mixture-of-Experts
```python
# 3 expert networks, each specialized for different outcome regimes
with pm.Model() as model:
    # Gating network: Which expert for which sample?
    gate_logits = pm.Normal('gate_logits', 0, 1, shape=(n_features, 3))
    gates = pm.Deterministic('gates', pm.math.softmax(X @ gate_logits, axis=1))

    # 3 expert networks
    experts = []
    for i in range(3):
        W = pm.Normal(f'W_expert_{i}', 0, 1, shape=(n_features, 32))
        b = pm.Normal(f'b_expert_{i}', 0, 1, shape=32)
        h = pm.math.tanh(X @ W + b)
        W_out = pm.Normal(f'W_out_expert_{i}', 0, 1, shape=(32, 1))
        mu_i = h @ W_out
        experts.append(mu_i)

    # Weighted combination
    mu = sum(gates[:, i:i+1] * experts[i] for i in range(3))

    # Expert-specific uncertainties
    sigmas = pm.HalfNormal('sigmas', sigma=5, shape=3)
    sigma = sum(gates[:, i] * sigmas[i] for i in range(3))

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
```

**Expected improvement:** Different experts for high/medium/low variance scenarios → better uncertainty quantification

#### Option C: Structured Player Priors
```python
# Hierarchical structure: League → Team → Player
with pm.Model() as model:
    # League-level hyperprior
    league_mu = pm.Normal('league_mu', 0, 10)
    league_sigma = pm.HalfNormal('league_sigma', 20)

    # Team-level effects
    team_mu = pm.Normal('team_mu', league_mu, league_sigma, shape=32)
    team_sigma = pm.HalfNormal('team_sigma', 10, shape=32)

    # Player-level effects (nested within teams)
    player_effects = pm.Normal('player_effects',
                                 team_mu[team_ids],
                                 team_sigma[team_ids],
                                 shape=n_players)

    # Network with structured priors
    mu = player_effects[player_ids] + network_prediction
    sigma = pm.HalfNormal('sigma', 15)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
```

**Expected improvement:** Borrows strength across hierarchy → better calibration for low-data players

**Evaluation Protocol:**
1. Train all 3 architectures on same data (2,663 games, 2020-2024)
2. Holdout test: 374 games
3. Primary metric: 90% CI coverage (target: 90%, baseline: 26%)
4. Secondary metrics: MAE, RMSE, ±1σ coverage
5. Select best architecture or ensemble

**Timeline:** 2 weeks (1 week implementation, 1 week evaluation)

### 2.2 Expand Shock Event Database

**Current Limitation:**
- Only 147 shock events over 5 years (2020-2024)
- Breakdown: 60 injuries, 47 coaching changes, 27 trades, 13 weather
- Insufficient for robust statistical inference

**Expansion Strategy:**

#### 2.2.1 Historical Backfill (1999-2019)
- Scrape injury reports from pro-football-reference.com
- Identify coaching changes from Wikipedia/ESPN archives
- Collect trade data from Spotrac transaction database
- Weather: Extend Meteostat queries to 1999-2019

**Target:** +500 events (25 events/year × 20 years)

#### 2.2.2 Additional Event Types
1. **Suspension returns** (PED, conduct violations)
   - Similar dynamics to injury returns
   - Source: NFL suspension database

2. **Bye week effects**
   - Rest advantage vs rust
   - Especially relevant for short/long weeks

3. **Prime time games**
   - Thursday/Monday night performance shifts
   - Narrative-driven line movement

4. **Rivalry games**
   - Division matchups with historical context
   - Market overreaction to narratives

**Target:** +200 events across new categories

#### 2.2.3 Data Quality Improvements
- Severity classification for injuries (games missed: 1, 2-4, 5+)
- Coaching change context (mid-season vs offseason, cause of change)
- Trade impact scores (star player vs depth chart)
- Weather interaction effects (dome vs outdoor, cold vs precipitation)

**Deliverable:** Expanded database with 850+ events (147 + 500 + 200)

**Timeline:** 3 weeks (1 week scraping, 1 week classification, 1 week validation)

### 2.3 Hybrid Calibration Method

While improving BNN architecture, implement hybrid approach:

```python
def hybrid_calibration(X_test, bnn_model, glm_model):
    """
    Combine BNN mean prediction with GLM residual calibration
    """
    # BNN provides mean estimate
    bnn_mean = bnn_model.predict(X_test).mean(axis=0)

    # Estimate residual distribution from GLM
    glm_pred = glm_model.predict(X_test)
    residuals = y_train - glm_model.predict(X_train)

    # Fit kernel density to residuals
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)

    # Generate calibrated intervals
    lower = bnn_mean + kde.ppf(0.05)
    upper = bnn_mean + kde.ppf(0.95)

    return {
        'mean': bnn_mean,
        'lower_90': lower,
        'upper_90': upper,
        'sigma_effective': (upper - lower) / 3.29  # 90% CI ≈ 3.29σ
    }
```

**Expected improvement:** Separates mean estimation (BNN strength) from uncertainty quantification (residual analysis)

### Phase 2 Success Criteria

1. **BNN Calibration:** 90% CI coverage ≥ 75% (vs 26% baseline)
2. **Database Size:** ≥850 shock events (vs 147 baseline)
3. **Treatment Effect Precision:** Standard errors reduced by 50%
4. **Betting Validation:** ROI improvement maintained with larger sample

---

## Phase 3: Real-Time Shock Detection (Q2 2025)

### Objective
Automate shock event detection and causal adjustment pipeline to enable real-time exploitation of market inefficiencies.

### 3.1 Automated Shock Monitoring

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                  Shock Detection Pipeline                │
├─────────────────────────────────────────────────────────┤
│  1. Data Sources (APIs, Scrapers, RSS feeds)            │
│     ├─ NFL Injury Reports (official API)                │
│     ├─ Twitter/X feed monitoring (@AdamSchefter, etc)   │
│     ├─ Weather forecasts (NOAA API, 24h window)         │
│     ├─ Coaching announcements (ESPN, NFL.com)           │
│     └─ Trade deadline monitoring (Spotrac API)          │
├─────────────────────────────────────────────────────────┤
│  2. Event Classification (NLP + Rules)                   │
│     ├─ Keyword detection ("out", "doubtful", "fired")   │
│     ├─ Severity assessment (games missed, role)         │
│     ├─ Context enrichment (team record, opponent)       │
│     └─ Duplicate detection (same event, multiple sources)│
├─────────────────────────────────────────────────────────┤
│  3. Leverage Scoring                                     │
│     ├─ Historical treatment effect lookup               │
│     ├─ Market efficiency check (line movement)          │
│     ├─ Opportunity window estimation (time to close)    │
│     └─ Priority queue (leverage × urgency)              │
├─────────────────────────────────────────────────────────┤
│  4. Alert Generation                                     │
│     ├─ High leverage (score >5): Immediate alert        │
│     ├─ Medium leverage (3-5): Batched alert             │
│     ├─ Low leverage (<3): Log only                      │
│     └─ Integration: Slack, email, dashboard             │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# py/causal/shock_detector.py
import asyncio
from dataclasses import dataclass
from typing import List, Dict
import aiohttp
from datetime import datetime, timedelta

@dataclass
class ShockEvent:
    event_type: str  # 'injury', 'coaching', 'trade', 'weather'
    player_id: Optional[str]
    team_id: str
    severity: float  # 0-1 scale
    detected_at: datetime
    games_affected: List[str]
    leverage_score: float
    treatment_effect_est: float
    treatment_effect_se: float

class ShockDetector:
    def __init__(self):
        self.sources = {
            'nfl_api': NFLInjuryAPI(),
            'twitter': TwitterMonitor(),
            'weather': WeatherForecastAPI(),
            'news_feed': ESPNNewsFeed()
        }
        self.event_history = []

    async def monitor(self, interval_seconds=300):
        """Poll all sources every 5 minutes"""
        while True:
            events = await self.check_all_sources()
            new_events = self.filter_duplicates(events)

            for event in new_events:
                event.leverage_score = self.compute_leverage(event)

                if event.leverage_score > 5.0:
                    await self.alert_high_priority(event)
                elif event.leverage_score > 3.0:
                    await self.alert_medium_priority(event)
                else:
                    self.log_event(event)

            await asyncio.sleep(interval_seconds)

    async def check_all_sources(self) -> List[ShockEvent]:
        """Parallel source polling"""
        tasks = [source.poll() for source in self.sources.values()]
        results = await asyncio.gather(*tasks)
        return [event for result in results for event in result]

    def compute_leverage(self, event: ShockEvent) -> float:
        """
        Leverage = treatment_effect_magnitude × market_inefficiency × urgency
        """
        # Load historical treatment effect for this event type
        effect = self.load_treatment_effect(event.event_type, event.severity)

        # Check if market has already adjusted
        market_efficiency = self.check_market_adjustment(event)

        # Time window (larger for games further out)
        urgency = self.compute_urgency(event.games_affected[0])

        return abs(effect.magnitude) * (1 - market_efficiency) * urgency
```

**Data Sources:**

1. **NFL Official Injury API**
   - Endpoint: `https://api.nfl.com/v1/injuries`
   - Update frequency: Real-time (webhook available)
   - Free tier: 1000 requests/day

2. **Twitter/X Monitoring**
   - Accounts: @AdamSchefter, @RapSheet, @JayGlazer, etc.
   - Method: Twitter API v2 filtered stream
   - Keywords: "out", "questionable", "doubtful", "fired", "traded"

3. **Weather Forecasts**
   - API: NOAA (National Weather Service)
   - Metrics: Wind speed, precipitation, temperature
   - Update: Every 6 hours, 7-day forecast

4. **Coaching/Trade News**
   - ESPN API: `http://site.api.espn.com/apis/site/v2/sports/football/nfl/news`
   - NFL.com RSS feeds
   - Spotrac transaction database

**Timeline:** 3 weeks (1 week per component: scrapers, classification, alerting)

### 3.2 Automated Adjustment Workflow

**Objective:** When shock detected, automatically adjust baseline predictions and recompute betting recommendations.

**Workflow:**

```
┌──────────────────────────────────────────────────────────────┐
│  Shock Detected (Leverage > 3.0)                              │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  1. Retrieve Baseline Prediction                              │
│     - Load most recent model prediction for affected game     │
│     - Baseline: p(home_win) = 0.58, spread = -3.5            │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Lookup Treatment Effect                                   │
│     - Query: event_type='injury', severity=0.8, position='QB' │
│     - Result: τ = -5.2 points (SE: 1.8), 95% CI: [-8.7, -1.7]│
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  3. Compute Adjusted Prediction                               │
│     - Adjusted spread = -3.5 + (-5.2) = -8.7 points          │
│     - Inflate uncertainty: σ_adjusted = 1.5 × σ_baseline     │
│     - Recompute p(home_win) with adjusted parameters          │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Market Comparison                                         │
│     - Current market line: Home -4.5                          │
│     - Model fair value: Home -8.7                             │
│     - Edge: 4.2 points toward Away                            │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  5. Generate Recommendation                                   │
│     - Bet: Away +4.5 (model fair value: +8.7)                │
│     - Confidence: High (edge = 4.2 points, leverage = 4.5)   │
│     - Kelly fraction: 2.5% (conservative, shock adjustment)   │
│     - Urgency: Place within 2 hours (line likely to move)    │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  6. Execution & Monitoring                                    │
│     - Auto-place if leverage >5.0 AND edge >3 points          │
│     - Manual review if 3.0 < leverage < 5.0                   │
│     - Track line movement post-placement                      │
│     - Update shock event database with outcome               │
└──────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# py/causal/adjustment_workflow.py
class CausalAdjustmentWorkflow:
    def __init__(self, baseline_model, treatment_db, market_api):
        self.baseline_model = baseline_model
        self.treatment_db = treatment_db
        self.market_api = market_api

    async def process_shock(self, event: ShockEvent):
        """End-to-end adjustment pipeline"""
        # Step 1: Retrieve baseline
        baseline = self.baseline_model.predict(event.games_affected[0])

        # Step 2: Lookup treatment effect
        treatment = self.treatment_db.lookup(
            event_type=event.event_type,
            severity=event.severity,
            position=event.position if hasattr(event, 'position') else None
        )

        # Step 3: Adjust prediction
        adjusted = self.compute_adjustment(baseline, treatment)

        # Step 4: Market comparison
        market_line = await self.market_api.get_current_line(
            event.games_affected[0]
        )
        edge = adjusted.fair_value - market_line.value

        # Step 5: Generate recommendation
        if abs(edge) > 3.0 and event.leverage_score > 5.0:
            recommendation = self.generate_recommendation(
                game_id=event.games_affected[0],
                side='away' if edge > 0 else 'home',
                line=market_line.value,
                fair_value=adjusted.fair_value,
                confidence='high',
                urgency='immediate'
            )

            # Step 6: Auto-execute or alert
            if self.should_auto_execute(recommendation):
                await self.execute_bet(recommendation)
            else:
                await self.alert_manual_review(recommendation)

        return adjusted, edge

    def compute_adjustment(self, baseline, treatment):
        """
        Bayesian update: Combine baseline with causal estimate
        """
        # Baseline: μ_baseline, σ_baseline
        # Treatment: τ, σ_τ

        # Posterior mean (precision-weighted average)
        prec_baseline = 1 / baseline.sigma**2
        prec_treatment = 1 / treatment.se**2

        mu_adjusted = (
            (prec_baseline * baseline.mean + prec_treatment * treatment.estimate) /
            (prec_baseline + prec_treatment)
        )

        # Posterior variance (sum of precisions)
        sigma_adjusted = np.sqrt(1 / (prec_baseline + prec_treatment))

        # Additional inflation for model uncertainty
        sigma_adjusted *= 1.5

        return AdjustedPrediction(
            mean=mu_adjusted,
            sigma=sigma_adjusted,
            fair_value=self.mean_to_line(mu_adjusted)
        )
```

**Safety Checks:**

1. **Sanity bounds:** Adjusted spread must be within ±14 points of baseline
2. **Uncertainty inflation:** Always increase σ by 1.5-2.0× for shock adjustments
3. **Market validation:** If market hasn't moved, reduce confidence
4. **Lookback validation:** Compare to similar historical shocks
5. **Human review:** All bets >2.5% Kelly fraction require manual approval

**Timeline:** 2 weeks (1 week implementation, 1 week backtesting)

### Phase 3 Success Criteria

1. **Detection latency:** <5 minutes from event occurrence to alert
2. **False positive rate:** <20% (i.e., 80% of high-leverage alerts are actionable)
3. **Adjustment accuracy:** Adjusted predictions closer to outcomes than baseline (tested on Phase 1 data)
4. **Automation rate:** ≥60% of medium/high leverage events processed without manual intervention

---

## Phase 4: Production Deployment (Q3 2025)

### Objective
Deploy causal adjustment system to production with comprehensive risk gates, monitoring, and fallback procedures.

### 4.1 Production Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Deployment Stack                   │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                      │
│    ├─ TimescaleDB: Historical shocks, treatment effects         │
│    ├─ Redis: Real-time event cache, market data                 │
│    └─ PostgreSQL: User preferences, bet history                 │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                               │
│    ├─ Shock Detector Service (FastAPI, async)                   │
│    ├─ Adjustment Workflow Service (Celery workers)              │
│    ├─ Market Integration Service (odds API clients)             │
│    └─ Execution Service (bet placement, confirmation)           │
├─────────────────────────────────────────────────────────────────┤
│  Risk Management Layer                                           │
│    ├─ Pre-Trade Gates (sanity checks, model drift)              │
│    ├─ Position Limits (per-game, per-week, portfolio)           │
│    ├─ Anomaly Detection (outlier predictions, data quality)     │
│    └─ Kill Switch (manual override, auto-pause on losses)       │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring & Observability                                      │
│    ├─ Metrics: Prometheus (detection latency, error rates)      │
│    ├─ Logging: ELK Stack (all events, adjustments, bets)        │
│    ├─ Dashboards: Grafana (real-time performance)               │
│    └─ Alerts: PagerDuty (critical failures, model drift)        │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                  │
│    ├─ Containers: Docker (all services)                         │
│    ├─ Orchestration: Kubernetes (auto-scaling, failover)        │
│    ├─ CI/CD: GitHub Actions (automated testing, deployment)     │
│    └─ Cloud: AWS (EC2, RDS, ElastiCache, S3)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Risk Gates

**Pre-Trade Validation:**

```python
class RiskGates:
    def validate_bet(self, recommendation: BetRecommendation) -> ValidationResult:
        """
        Multi-stage validation before executing shock-adjusted bet
        """
        checks = [
            self.sanity_bounds_check(recommendation),
            self.model_drift_check(recommendation),
            self.position_limit_check(recommendation),
            self.market_efficiency_check(recommendation),
            self.historical_analog_check(recommendation),
            self.uncertainty_budget_check(recommendation)
        ]

        results = [check() for check in checks]

        if all(r.passed for r in results):
            return ValidationResult(passed=True, confidence='high')
        elif sum(r.passed for r in results) >= 5:
            return ValidationResult(passed=True, confidence='medium',
                                     warnings=[r.message for r in results if not r.passed])
        else:
            return ValidationResult(passed=False, reason='Failed multiple risk gates',
                                     failed_checks=[r.name for r in results if not r.passed])

    def sanity_bounds_check(self, rec: BetRecommendation) -> CheckResult:
        """Adjusted prediction must be within reasonable bounds"""
        if abs(rec.adjusted_value - rec.baseline_value) > 14:
            return CheckResult(passed=False, name='sanity_bounds',
                                message=f'Adjustment too large: {rec.adjusted_value - rec.baseline_value} points')
        return CheckResult(passed=True, name='sanity_bounds')

    def model_drift_check(self, rec: BetRecommendation) -> CheckResult:
        """Check if model performance has degraded recently"""
        recent_performance = self.get_recent_brier_score(window_days=7)
        baseline_performance = self.get_baseline_brier_score()

        if recent_performance > baseline_performance * 1.2:
            return CheckResult(passed=False, name='model_drift',
                                message=f'Model drift detected: Brier {recent_performance:.4f} vs baseline {baseline_performance:.4f}')
        return CheckResult(passed=True, name='model_drift')

    def position_limit_check(self, rec: BetRecommendation) -> CheckResult:
        """Enforce portfolio-level position limits"""
        current_exposure = self.get_current_week_exposure()
        max_weekly_exposure = self.config['max_weekly_bankroll_pct']

        if current_exposure + rec.stake_pct > max_weekly_exposure:
            return CheckResult(passed=False, name='position_limit',
                                message=f'Weekly exposure limit: {current_exposure + rec.stake_pct:.1f}% > {max_weekly_exposure}%')
        return CheckResult(passed=True, name='position_limit')

    def historical_analog_check(self, rec: BetRecommendation) -> CheckResult:
        """Compare to similar historical shocks"""
        analogs = self.find_historical_analogs(rec.shock_event, n=10)

        if len(analogs) < 5:
            return CheckResult(passed=False, name='historical_analog',
                                message='Insufficient historical analogs for validation')

        analog_mean_effect = np.mean([a.treatment_effect for a in analogs])
        analog_std = np.std([a.treatment_effect for a in analogs])

        if abs(rec.treatment_effect - analog_mean_effect) > 2 * analog_std:
            return CheckResult(passed=False, name='historical_analog',
                                message=f'Treatment effect outlier: {rec.treatment_effect:.2f} vs historical {analog_mean_effect:.2f} ± {analog_std:.2f}')

        return CheckResult(passed=True, name='historical_analog')
```

**Kill Switch Conditions:**

1. **Weekly loss threshold:** Pause if down >5% of bankroll in a week
2. **Model degradation:** Pause if Brier score >1.2× baseline for 3+ days
3. **Data anomalies:** Pause if >10% missing features or stale data
4. **Market disruption:** Pause if odds APIs down or stale (>15 min)
5. **Manual override:** Admin can pause system anytime

### 4.3 Monitoring Dashboard

**Key Metrics:**

1. **Detection Performance**
   - Events detected per day
   - Detection latency (event → alert)
   - False positive rate
   - Missed events (manual review)

2. **Adjustment Quality**
   - Adjusted MAE vs baseline MAE
   - Calibration (90% CI coverage)
   - Treatment effect accuracy (observed vs predicted)
   - Market efficiency (how quickly lines move post-shock)

3. **Betting Performance**
   - Win rate (shock-adjusted bets)
   - ROI (shock-adjusted vs baseline)
   - Sharpe ratio
   - Max drawdown

4. **System Health**
   - API uptime (NFL, weather, market)
   - Service latency (p50, p95, p99)
   - Error rates (by service)
   - Resource utilization (CPU, memory, DB connections)

**Alert Thresholds:**

- **Critical:** Detection latency >15 min, API downtime >30 min, model drift >1.5×
- **Warning:** Detection latency >5 min, false positive rate >30%, weekly loss >3%
- **Info:** New shock detected, bet placed, model updated

### 4.4 Gradual Rollout Plan

**Week 1-2: Shadow Mode**
- System runs in parallel with baseline
- All shocks detected and adjusted predictions computed
- NO bets placed
- Goal: Validate detection accuracy, adjustment logic

**Week 3-4: Pilot (10% allocation)**
- Place bets only for highest leverage events (score >6.0)
- Limit to 10% of weekly bankroll
- Manual review required for all bets
- Goal: Test execution, validate performance

**Week 5-8: Controlled Expansion (30% allocation)**
- Increase to leverage >4.0
- 30% of weekly bankroll
- Auto-execute for leverage >6.0, manual review for 4.0-6.0
- Goal: Build confidence, refine risk gates

**Week 9+: Full Production (50% allocation)**
- All medium/high leverage events (>3.0)
- 50% of weekly bankroll (shock strategies)
- Auto-execute for leverage >5.0
- Continuous monitoring and tuning

### Phase 4 Success Criteria

1. **Uptime:** ≥99.5% (excluding planned maintenance)
2. **Detection accuracy:** ≥80% true positives, ≤20% false positives
3. **Performance:** ROI ≥+5% on shock-adjusted bets (vs +1.8% baseline)
4. **Risk management:** Zero risk gate failures leading to losses >2.5% bankroll
5. **Scalability:** System handles ≥50 concurrent shocks without degradation

---

## Appendix A: Timeline & Resource Allocation

### Q1 2025 (Phase 2): Fix BNN Calibration & Expand Database

**Weeks 1-2: BNN Architecture Experiments**
- Research: 3 days (literature review on hierarchical Bayes, mixture models)
- Implementation: 7 days (3 architectures × 2 days + 1 day debugging)
- Evaluation: 4 days (training, testing, comparison)
- **Resources:** 1 ML engineer, NVIDIA RTX 4090 GPU

**Weeks 3-5: Database Expansion**
- Scraping: 5 days (historical injury/coaching/trade data)
- Classification: 4 days (severity scoring, quality checks)
- Validation: 3 days (spot checks, data integrity tests)
- **Resources:** 1 data engineer, web scraping infrastructure

**Week 6: Hybrid Calibration & Integration**
- Implementation: 3 days (hybrid method, codebase integration)
- Testing: 2 days (unit tests, integration tests)
- **Resources:** 1 ML engineer

**Total:** 6 weeks, 2 FTEs (ML + data engineering)

### Q2 2025 (Phase 3): Real-Time Shock Detection

**Weeks 1-3: Automated Monitoring**
- API clients: 5 days (NFL, Twitter, weather, news)
- Classification: 4 days (NLP keyword detection, severity scoring)
- Alerting: 3 days (Slack integration, dashboard)
- Testing: 3 days (end-to-end, load testing)
- **Resources:** 1 backend engineer

**Weeks 4-6: Adjustment Workflow**
- Core logic: 5 days (adjustment computation, Bayesian updates)
- Market integration: 4 days (odds API clients, line tracking)
- Safety checks: 3 days (risk gates, validation)
- Backtesting: 3 days (validate on Phase 1 data)
- **Resources:** 1 ML engineer, 1 backend engineer

**Total:** 6 weeks, 2 FTEs (backend + ML engineering)

### Q3 2025 (Phase 4): Production Deployment

**Weeks 1-2: Infrastructure Setup**
- Containerization: 3 days (Docker images, docker-compose)
- Kubernetes: 4 days (deployment configs, auto-scaling)
- CI/CD: 3 days (GitHub Actions, automated testing)
- **Resources:** 1 DevOps engineer

**Weeks 3-4: Risk Gates & Monitoring**
- Risk gates: 5 days (implementation, testing)
- Monitoring: 5 days (Prometheus, Grafana, alerts)
- **Resources:** 1 backend engineer, 1 DevOps engineer

**Weeks 5-12: Gradual Rollout**
- Shadow mode: 2 weeks (monitoring, validation)
- Pilot: 2 weeks (10% allocation, manual review)
- Controlled expansion: 4 weeks (30% allocation, tuning)
- **Resources:** 1 ML engineer (monitoring), 1 operations analyst

**Total:** 12 weeks, 2.5 FTEs (DevOps + backend + part-time ML/ops)

**Grand Total:** 24 weeks, ~2 FTEs averaged across phases

---

## Appendix B: Risk Mitigation Strategies

### Technical Risks

**Risk 1: BNN calibration improvements insufficient**
- **Mitigation:** Hybrid approach (BNN + residual calibration) as fallback
- **Contingency:** Use GLM with uncertainty quantification instead of BNN

**Risk 2: API rate limits / downtime**
- **Mitigation:** Multi-source redundancy (official API + web scraping backup)
- **Contingency:** Manual shock detection workflow (human-in-the-loop)

**Risk 3: Model drift in production**
- **Mitigation:** Continuous monitoring, auto-pause on drift detection
- **Contingency:** Revert to baseline model until retrained

### Operational Risks

**Risk 4: False positives cause bad bets**
- **Mitigation:** Historical analog check, human review for borderline cases
- **Contingency:** Tighten leverage threshold (e.g., >6.0 instead of >3.0)

**Risk 5: Market adapts (closes inefficiency)**
- **Mitigation:** Continuous A/B testing (shock-adjusted vs baseline)
- **Contingency:** Reduce allocation if ROI drops below +2% for 4 consecutive weeks

**Risk 6: Regulatory/compliance issues**
- **Mitigation:** Legal review before production launch
- **Contingency:** Limit to research mode (paper trading only)

---

## Appendix C: Success Metrics & KPIs

### Phase 2 KPIs (Q1 2025)

1. **BNN Calibration**
   - 90% CI coverage: Target ≥75% (vs 26% baseline)
   - MAE: Maintain ≤19 yards (current: 18.7)
   - R-hat: <1.01 for all parameters

2. **Database Quality**
   - Sample size: ≥850 events (vs 147 baseline)
   - Classification accuracy: ≥90% (spot check 100 random events)
   - Missing data: <5% for key fields

3. **Treatment Effect Precision**
   - Standard errors: Reduced by ≥40% (more data → tighter CIs)
   - Statistical power: ≥80% to detect effect size of 3 points

### Phase 3 KPIs (Q2 2025)

1. **Detection Performance**
   - Latency: <5 min (p95)
   - True positive rate: ≥80%
   - False positive rate: ≤20%

2. **Adjustment Accuracy**
   - Adjusted MAE vs baseline: Improvement ≥10%
   - Calibration: 90% CI coverage ≥75%

3. **Automation**
   - Auto-processed: ≥60% of medium/high leverage events
   - Manual review time: <10 min per event

### Phase 4 KPIs (Q3 2025)

1. **System Reliability**
   - Uptime: ≥99.5%
   - Detection latency: <5 min (p95)
   - Zero risk gate failures causing losses >2.5%

2. **Business Impact**
   - ROI (shock bets): ≥+5% (vs +1.8% baseline in Phase 1)
   - Win rate: ≥56% (vs 53% baseline)
   - Sharpe ratio: ≥1.2

3. **Scalability**
   - Concurrent shocks handled: ≥50
   - API throughput: ≥1000 req/s
   - Database query latency: <100ms (p95)

---

## Conclusion

The causal inference framework for shock event modeling offers a promising path to exploiting market inefficiencies during regime breaks—the most actionable opportunity given semi-strong efficiency of typical NFL game markets. Phase 1 established a solid foundation with rigorous methodology, negative result documentation, and preliminary validation (+3.3 pp ROI on 147 events).

Phases 2-4 address identified weaknesses (BNN calibration, sample size, manual processes) and build toward a production-ready system with real-time detection, automated adjustment, and comprehensive risk management. The gradual rollout plan ensures safe deployment with multiple feedback loops.

**Key success factors:**
1. **Methodological rigor:** Continue using state-of-the-art causal inference (SC, DiD, DAGs)
2. **Honest validation:** Document failures alongside successes
3. **Risk management:** Multiple safety gates, kill switches, conservative sizing
4. **Continuous learning:** A/B testing, model retraining, database expansion

**Expected outcomes by Q3 2025:**
- Production-ready shock detection system with <5 min latency
- Well-calibrated BNN or hybrid model (≥75% coverage)
- Database of 850+ shock events with precise treatment effects
- Demonstrated ROI improvement (≥+5%) on live shock event betting
- Comprehensive monitoring, risk gates, and fallback procedures

This framework represents a competitive advantage in NFL betting markets: while most participants rely on correlational models and narratives, our causal reasoning provides quantified, testable estimates of treatment effects with proper uncertainty bounds. The systematic approach to exploiting shock events—combined with patient capital allocation and rigorous risk management—positions the system for sustainable alpha generation in sparse-but-significant market opportunities.
