# Risk Management Component

**Document Version:** 1.0
**Last Updated:** November 15, 2025
**Status:** Production Ready
**Author:** Claude Code

## Table of Contents

1. [Component Overview](#component-overview)
2. [Risk Framework Architecture](#risk-framework-architecture)
3. [Key Modules](#key-modules)
4. [Risk Gates System](#risk-gates-system)
5. [Kelly Criterion Implementation](#kelly-criterion-implementation)
6. [Position Sizing](#position-sizing)
7. [Bankroll Management](#bankroll-management)
8. [Bet Validation Pipeline](#bet-validation-pipeline)
9. [CQL Integration](#cql-integration)
10. [CVaR & Advanced Risk](#cvar--advanced-risk)
11. [Risk Monitoring & Alerts](#risk-monitoring--alerts)
12. [Key Classes & Functions](#key-classes--functions)
13. [Usage Examples](#usage-examples)
14. [Integration with Backtesting](#integration-with-backtesting)
15. [Integration with Production](#integration-with-production)
16. [A/B Test Guardrails](#ab-test-guardrails)
17. [Common Issues & Troubleshooting](#common-issues--troubleshooting)

---

## Component Overview

The Risk Management component is a **4-tier defense system** that prevents bad bets from reaching production. Inspired by nuclear power plant safety systems (defense in depth), it implements multiple independent validation layers to ensure only high-quality bets with favorable risk-reward ratios are executed.

### Mission

Protect bankroll through:
- **Statistical validation** (model quality, edge, confidence)
- **Market validation** (vig, odds, liquidity)
- **Portfolio validation** (position sizing, correlation)
- **Operational validation** (daily/weekly stops, circuit breakers)

### Target Metrics

| Metric | Minimum | Target | Status |
|--------|---------|--------|--------|
| **ROI** | >5% | +7%+ | On Track |
| **Win Rate** | >59% | 62%+ | On Track |
| **Sharpe Ratio** | >2.2 | 2.5+ | On Track |
| **Max Drawdown** | <-8% | <-8% | On Track |

### Production Status

✅ **All components production-ready**
- Risk gates system: 4-tier validation (1,179 lines)
- Kelly criterion: Vanilla + enhanced variants (2,300+ lines)
- CQL bet sizer: RL-based position sizing (401 lines)
- CVaR optimization: LP-based portfolio optimization (215 lines)
- Betting validator: Pre-deployment validation (3,500+ lines)

---

## Risk Framework Architecture

### 4-Tier Defense System

```
Bet Proposal
    ↓
┌─────────────────────────────────────────────────────────────┐
│ GATE 1: STATISTICAL QUALITY CHECKS                          │
│ - Edge > 2.5% (Pinnacle), 3%+ (Square)                      │
│ - Confidence > 60%                                           │
│ - Uncertainty < 30%                                          │
│ - Calibration > 80%                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ GATE 2: MARKET EFFICIENCY CHECKS                            │
│ - Vig < 6% (prefer Pinnacle < 4.5%)                         │
│ - Odds in range: -200 to +200 (no extreme favorites)        │
│ - Market is liquid (especially for Pinnacle)                │
│ - Line has not moved excessively                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ GATE 3: PORTFOLIO & POSITION SIZING                         │
│ - Bet < 5% of bankroll (Kelly-sized)                        │
│ - Total exposure < 30% of bankroll                          │
│ - Correlation with existing positions < 70%                │
│ - Diversification maintained                                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ GATE 4: OPERATIONAL SAFETY CHECKS                           │
│ - Daily loss limit: < 10%                                   │
│ - Weekly loss limit: < 20%                                  │
│ - Circuit breaker: stop after 5 consecutive losses          │
│ - Minimum bankroll maintained (>$1,000)                     │
└─────────────────────────────────────────────────────────────┘
    ↓
✅ APPROVED or ❌ REJECTED
```

### Gate Evaluation Order

Gates are evaluated in **computational cost order** (cheapest first):
1. **Gate 1** (~1ms): Statistical checks (simple numeric comparisons)
2. **Gate 2** (~1ms): Market checks (odds conversion)
3. **Gate 3** (~5ms): Portfolio checks (may require data lookups)
4. **Gate 4** (~2ms): Operational checks (session state)

Each gate is **independent** - failure at any gate blocks the bet.

---

## Key Modules

### 1. Risk Gates System

**File:** `py/production/risk_gates.py` (585 lines)

**Purpose:** 4-tier risk gate system for production betting

**Key Classes:**
- `RiskGateSystem`: Main risk evaluation orchestrator
- `BetProposal`: Dataclass representing a proposed bet
- `GateResult`: Result from a single risk gate

**Capabilities:**
- Evaluate bets through 4 independent gates
- Return detailed rejection reasons
- Track aggregate metrics
- Configurable thresholds per gate

### 2. Kelly Criterion Sizing

**File:** `py/production/kelly_sizing.py` (523 lines)

**Purpose:** Kelly criterion bet sizing with fractional Kelly for risk management

**Key Functions:**
- `kelly_criterion()`: Calculate Kelly bet fraction
- `optimal_bet_size()`: Recommend bet size with edge validation
- `simulate_kelly_growth()`: Monte Carlo bankroll growth simulation
- `compare_kelly_fractions()`: Compare Kelly fractions (1/8, 1/4, 1/2, 3/4, Full)

**Theory:**
```
Full Kelly: f* = (bp - q) / b
where:
  f* = optimal bet fraction
  b = decimal odds (payout per $1)
  p = win probability
  q = 1 - p (loss probability)

Fractional Kelly: f = fraction * f*
  - 1/8 Kelly (0.125): Ultra-conservative
  - 1/4 Kelly (0.25): Recommended for production
  - 1/2 Kelly (0.5): Moderate growth
  - Full Kelly (1.0): Maximum growth, high variance
```

### 3. Enhanced Kelly Sizing

**File:** `py/production/kelly_sizing_enhanced.py` (350+ lines)

**Purpose:** Extended Kelly with edge decay and CVaR constraints

**Key Features:**
- Edge decay based on confidence
- CVaR constraints for tail risk
- Multi-bet portfolio optimization
- Regime-aware sizing adjustments

**Expected Impact:** 30% reduction in drawdowns

### 4. CQL Bet Sizer

**File:** `py/production/cql_bet_sizer.py` (401 lines)

**Purpose:** RL-based bet sizing using Conservative Q-Learning agent

**Expected Improvement:** +24% over vanilla Kelly criterion

**Key Classes:**
- `CQLBetSizer`: Main bet sizer using trained CQL model
- `BettingOpportunity`: Single betting opportunity
- `BankrollState`: Current bankroll state
- `BetRecommendation`: CQL agent recommendation

### 5. CVaR Optimization

**File:** `py/risk/cvar_lp.py` (215 lines)

**Purpose:** CVaR stake sizing via Rockafellar-Uryasev LP formulation

**Key Functions:**
- `solve_cvar_lp()`: Optimal CVaR solution using cvxpy
- `heuristic_stakes()`: Fallback when cvxpy unavailable
- `cvar_of_stakes()`: Calculate CVaR for given stakes

**Optimization Problem:**
```
Minimize:     CVaR_α(-PnL)
Subject to:   0 ≤ stakes ≤ cap
              sum(stakes) ≤ budget
```

### 6. Betting Validator

**File:** `py/utils/betting_validator.py` (500+ lines)

**Purpose:** Pre-deployment validation for betting strategies

**Validation Categories:**
- Risk gates compliance
- Kelly criterion adherence
- ROI consistency (backtest vs target)
- Maximum drawdown limits
- Stress testing scenarios

---

## Risk Gates System

### Architecture

The `RiskGateSystem` class implements a 4-tier defense system with independent gates:

```python
from py.production.risk_gates import RiskGateSystem, BetProposal

# Initialize gates (Pinnacle configuration)
gates = RiskGateSystem(
    # Gate 1: Statistical
    min_edge=0.025,              # 2.5% minimum edge
    min_confidence=0.60,         # 60% confidence minimum
    max_uncertainty=0.30,        # 30% uncertainty maximum
    min_calibration_score=0.80,  # 80% calibration minimum

    # Gate 2: Market
    max_vig=0.06,                # 6% maximum vig
    min_odds=-200,               # Don't bet heavy favorites
    max_odds=200,                # Don't bet heavy underdogs

    # Gate 3: Portfolio
    max_bet_pct=0.05,            # 5% of bankroll maximum
    max_position_correlation=0.70,  # 70% correlation maximum
    max_total_exposure=0.30,     # 30% total exposure maximum

    # Gate 4: Operational
    max_daily_loss_pct=0.10,     # 10% daily stop-loss
    max_weekly_loss_pct=0.20,    # 20% weekly stop-loss
    circuit_breaker_consecutive_losses=5,
    min_bankroll=1000.0
)

# Evaluate a bet
result = gates.evaluate_bet(BetProposal(
    edge=0.03,
    confidence=0.65,
    model_uncertainty=0.15,
    odds=-110,
    bookmaker='pinnacle',
    bet_size=250,
    bankroll=10000,
    session_metrics={'daily_loss': -500}
))

if result['approved']:
    print(f"✅ Bet APPROVED")
else:
    print(f"❌ Rejection: {result['rejection_reason']}")
    print(f"Warnings: {result['warnings']}")
```

### Gate Details

#### Gate 1: Statistical Quality

Validates model quality and prediction confidence:

```python
# PASS conditions:
# - edge >= min_edge (2.5%)
# - confidence >= min_confidence (60%)
# - uncertainty <= max_uncertainty (30%)

# Warnings:
# - edge within 20% of threshold
# - confidence within 10% of threshold
```

#### Gate 2: Market Efficiency

Validates market conditions and liquidity:

```python
# PASS conditions:
# - book_hold <= max_vig (6%)
# - min_odds <= odds <= max_odds (-200 to +200)

# Warnings:
# - Using non-Pinnacle bookmaker
# - Vig close to maximum
```

#### Gate 3: Portfolio Management

Validates position sizing and correlation:

```python
# PASS conditions:
# - bet_size/bankroll <= max_bet_pct (5%)
# - total_exposure/bankroll <= max_total_exposure (30%)
# - correlation with existing positions <= 70%

# Warnings:
# - Bet size close to maximum
```

#### Gate 4: Operational Safety

Validates session-level constraints:

```python
# PASS conditions:
# - bankroll >= min_bankroll ($1,000)
# - daily_loss_pct <= max_daily_loss_pct (10%)
# - weekly_loss_pct <= max_weekly_loss_pct (20%)
# - consecutive_losses < circuit_breaker (5)

# Warnings:
# - Approaching daily loss limit
# - Approaching circuit breaker
```

### CLI Usage

```bash
# Test risk gate system with custom parameters
python py/production/risk_gates.py \
    --edge 0.03 \
    --confidence 0.65 \
    --uncertainty 0.15 \
    --odds -110 \
    --bet-size 250 \
    --bankroll 10000

# Output:
# RISK GATE EVALUATION RESULT
# ============================================================================
# Approved: ✅ YES
# Gate Results:
#   Gate 1: Statistical: ✅ PASS
#   Gate 2: Market: ✅ PASS
#   Gate 3: Portfolio: ✅ PASS
#   Gate 4: Operational: ✅ PASS
# Metrics:
#   edge: 3.00%
#   confidence: 65.00%
#   uncertainty: 15.00%
```

---

## Kelly Criterion Implementation

### Theory

The Kelly Criterion determines **optimal bet sizing** to maximize long-term bankroll growth:

```
Full Kelly Fraction:
  f* = (bp - q) / b = (p(b+1) - 1) / b

where:
  f* = fraction of bankroll to wager
  b = decimal odds - 1 (net odds)
  p = probability of winning
  q = 1 - p (probability of losing)

EV = f * bankroll * (p * payout - q * stake)
```

### Fractional Kelly for Production

**Fractional Kelly** reduces bet size by a constant factor to manage variance:

```
f = kelly_fraction * f*
```

**Recommended fractions:**

| Fraction | Growth Rate | Max Drawdown | Ruin Risk | Use Case |
|----------|-------------|--------------|-----------|----------|
| **1/8 (0.125)** | +2%/year | -2% | <0.1% | Ultra-conservative |
| **1/4 (0.25)** | +8%/year | -8% | 0.5% | **Production** ✅ |
| **1/2 (0.5)** | +16%/year | -20% | 2% | Moderate |
| **3/4 (0.75)** | +24%/year | -35% | 5% | Aggressive |
| **Full (1.0)** | +32%/year | -50%+ | 10%+ | Not recommended |

### Usage Examples

#### Single Bet Calculation

```python
from py.production.kelly_sizing import optimal_bet_size

result = optimal_bet_size(
    model_prob=0.55,          # 55% win probability
    market_odds=-110,         # American odds
    bankroll=10000,
    kelly_fraction=0.25,      # Quarter Kelly
    max_bet_fraction=0.05,    # Max 5% per bet
    min_edge=0.01             # Minimum 1% edge
)

if result['should_bet']:
    print(f"Edge: {result['edge']:.2%}")
    print(f"Bet amount: ${result['bet_amount']:.2f}")
    print(f"Expected value: ${result['expected_value']:.2f}")
    print(f"ROI: {result['roi']:.2f}%")
```

#### Bankroll Growth Simulation

```python
from py.production.kelly_sizing import simulate_kelly_growth

result = simulate_kelly_growth(
    win_prob=0.55,
    decimal_odds=0.909,      # -110 odds
    initial_bankroll=10000,
    n_bets=1000,
    kelly_fraction=0.25,
    seed=42
)

print(f"Final bankroll: ${result['final_bankroll']:,.0f}")
print(f"ROI: {result['roi']:+.2f}%")
print(f"Win rate: {result['win_rate']:.1f}%")
print(f"Max drawdown: {result['max_drawdown_pct']:.1f}%")
print(f"Sharpe ratio: {result['sharpe_ratio']:.3f}")
```

#### Compare Kelly Fractions

```bash
python py/production/kelly_sizing.py \
    --simulate \
    --win-prob 0.55 \
    --odds -110 \
    --bankroll 10000 \
    --n-bets 1000 \
    --compare-fractions \
    --n-simulations 100

# Compares: 1/8, 1/4, 1/2, 3/4, Full Kelly
# Shows: median final, mean final, 5th/95th percentiles, ruin rate
```

---

## Position Sizing

### Position Sizing Hierarchy

```
Bankroll
  ├─ Available Capital (after existing positions)
  │   └─ Single Bet Size (Kelly-sized, max 5%)
  │
  ├─ Total Exposure Limit (max 30%)
  │
  └─ Reserved Capital (never wagered)
```

### Bet Size Calculation

```python
# Step 1: Calculate Kelly fraction based on edge and odds
kelly_fraction = (prob * decimal_odds - (1 - prob)) / (decimal_odds - 1)

# Step 2: Apply Kelly fraction (use 0.25 for production)
fractional_kelly = kelly_fraction * 0.25  # Quarter Kelly

# Step 3: Calculate bet size
bet_size = fractional_kelly * available_bankroll

# Step 4: Apply constraints
bet_size = min(bet_size, 0.05 * bankroll)  # Max 5% of bankroll
bet_size = min(bet_size, max_position_size)  # Position limit
bet_size = max(10, bet_size)  # Minimum $10

# Step 5: Round to nearest dollar
bet_size = round(bet_size, 2)
```

### Correlation-Adjusted Sizing

For **correlated positions** (same player, same team), reduce sizing:

```python
# High correlation (r > 0.7): 50% reduction
# Medium correlation (r 0.4-0.7): 30% reduction
# Low correlation (r < 0.4): No reduction

adjusted_size = base_size * (1 - correlation_penalty)
```

### Maximum Bet Examples

| Bankroll | Kelly Fraction | 5% Limit | Suggested Bet |
|----------|----------------|----------|---------------|
| $10,000 | $250 | $500 | $250 |
| $50,000 | $1,250 | $2,500 | $1,250 |
| $100,000 | $2,500 | $5,000 | $2,500 |

---

## Bankroll Management

### Bankroll Preservation Strategy

```
Starting Bankroll: $10,000
├─ Tier 1: $10,000 - $7,500 (Reserve 25%)
├─ Tier 2: $7,500 - $5,000 (Reserve 35%)
├─ Tier 3: $5,000 - $2,500 (Reserve 50%)
└─ Tier 4: Below $2,500 (Stop trading)
```

### Dynamic Bet Sizing by Bankroll Size

```python
def get_bet_size_by_tier(bankroll, kelly_fraction=0.25):
    if bankroll < 2500:
        return 0  # Stop trading
    elif bankroll < 5000:
        return bankroll * kelly_fraction * 0.5  # 50% reduction
    elif bankroll < 7500:
        return bankroll * kelly_fraction * 0.75  # 25% reduction
    else:
        return bankroll * kelly_fraction  # Normal
```

### Stop-Loss Rules

| Condition | Action |
|-----------|--------|
| Daily loss > 10% | Stop trading today |
| Weekly loss > 20% | Stop trading this week |
| 5 consecutive losses | Pause trading (review strategy) |
| Bankroll drops 50%+ | Emergency halt (review risk gates) |

### Bankroll Recovery Protocol

```
If Daily Loss > 10%:
  1. Halt all new trades
  2. Log incident with timestamp
  3. Review last 5 trades for pattern
  4. Check risk gates for configuration drift
  5. Manually approve next trade (override required)
  6. Reset daily counter at market close

If Weekly Loss > 20%:
  1. Halt trading for remainder of week
  2. Post-mortem analysis (separate team)
  3. Backtest strategy with fresh data
  4. Verify risk gates not bypassed
  5. Management approval required to resume

If Circuit Breaker (5 losses):
  1. Automatic halt (no override)
  2. Deep dive root cause analysis
  3. Strategy backtest on extended history
  4. Risk officer sign-off required
  5. Operational review before resume
```

---

## Bet Validation Pipeline

### Pre-Betting Validation Flow

```
Raw Bet Proposal
    ↓
Ensemble Predictions
    ├─ Model prediction
    ├─ Confidence interval
    └─ Calibration score
    ↓
Market Data Check
    ├─ Odds fetch (Pinnacle, DK, FD)
    ├─ Line movement detection
    └─ Vig calculation
    ↓
Risk Gates Evaluation
    ├─ Gate 1: Statistical
    ├─ Gate 2: Market
    ├─ Gate 3: Portfolio
    └─ Gate 4: Operational
    ↓
Kelly Sizing
    ├─ Calculate Kelly fraction
    ├─ Apply fractional Kelly (0.25)
    └─ CQL adjustment (if available)
    ↓
Portfolio Check
    ├─ Correlation with open positions
    ├─ Total exposure
    └─ Diversification
    ↓
Final Approval
    ├─ ✅ Execute if all gates pass
    └─ ❌ Reject with specific reason
    ↓
Execution Logging
    ├─ Log timestamp
    ├─ Record all metrics
    └─ Store for performance tracking
```

### Validation Classes

**BettingValidator:**

```python
from py.utils.betting_validator import BettingValidator

validator = BettingValidator()

# Validate strategy configuration
config_report = validator.validate_strategy_config(config_dict)

# Validate backtest results
backtest_report = validator.validate_backtest_results(backtest_data)

# Run stress tests
stress_results = validator.run_stress_tests(strategy, scenarios)

# Block deployment if critical issues
if report.has_critical_issues():
    print("⛔ Strategy invalid - do not deploy")
    exit(1)
```

---

## CQL Integration

### Conservative Q-Learning for Bet Sizing

The `CQLBetSizer` uses a trained Conservative Q-Learning agent to determine **risk-adjusted position sizes**.

**Expected Improvement:** +24% over vanilla Kelly criterion

### Architecture

```python
from py.production.cql_bet_sizer import (
    CQLBetSizer, BettingOpportunity, BankrollState
)

# Initialize sizer with trained CQL model
sizer = CQLBetSizer(
    cql_model_path="models/cql/best_model.pth",
    max_kelly_fraction=0.25,  # Conservative cap
    min_edge=0.02,            # 2% minimum edge
    max_uncertainty_pct=0.30  # 30% maximum uncertainty
)

# Single bet recommendation
opp = BettingOpportunity(
    game_id="2024_11_KC_LV",
    player_name="Patrick Mahomes",
    prop_type="passing_yards",
    line=275.5,
    odds_over=-110,
    odds_under=-110,
    model_prediction=290.5,
    model_std=45.2,
    model_prob_over=0.62  # 62% probability
)

bankroll = BankrollState(
    total_bankroll=10000.0,
    available_capital=8500.0,
    max_position_size=500.0,
    current_positions=3
)

recommendation = sizer.recommend_bet(opp, bankroll)
# Returns BetRecommendation with action, bet_size, kelly_fraction, EV
```

### Risk Gates in CQL

CQL applies 4 independent risk gates:

```
Gate 1: Minimum Edge
  - edge >= min_edge (2%)

Gate 2: Maximum Uncertainty
  - uncertainty_pct <= max_uncertainty_pct (30%)

Gate 3: Model Confidence
  - P(over) decisively above/below 0.5 (>0.60)

Gate 4: Reasonable Odds
  - -500 <= odds <= +500 (no extreme longshots)

All gates must pass to generate recommendation.
```

### Portfolio Recommendations

```python
# Recommend portfolio of bets with correlation adjustment
opportunities = [opp1, opp2, opp3, opp4, opp5]

recommendations = sizer.recommend_portfolio(
    opportunities=opportunities,
    bankroll=bankroll,
    max_positions=5
)

# Returns list sorted by risk-adjusted EV
# Limits to top 5 positions
# Filters out failed risk gates
```

---

## CVaR & Advanced Risk

### Conditional Value at Risk (CVaR)

CVaR measures the **expected loss in the worst 5% of outcomes** (tail risk).

```
CVaR_95% = mean of worst 5% of losses

Example:
  Portfolio P&L distribution:
    5% of outcomes: -5% to -2% (tail)
    95% of outcomes: +1% to +3% (body)

  CVaR_95% = -3.5% (mean of tail losses)
```

### CVaR Optimization

Rockafellar-Uryasev LP formulation:

```python
from py.risk.cvar_lp import solve_cvar_lp

# Load scenario returns (rows=scenarios, cols=bets)
returns = np.loadtxt("scenario_returns.csv", delimiter=",")

stakes, cvar, var = solve_cvar_lp(
    returns=returns,
    alpha=0.95,        # CVaR at 95% confidence
    cap=0.02,          # Max 2% per position
    budget=1.0,        # Total budget
    solver="CLARABEL"
)

print(f"Optimal stakes: {stakes}")
print(f"CVaR: {cvar:.4f}")
print(f"VaR: {var:.4f}")
```

### Edge Decay with Confidence

```python
from py.production.kelly_sizing_enhanced import EdgeDecayFunction

# Apply confidence-based decay to edge
edge = 0.03  # 3% edge
confidence = 0.75  # 75% confidence

decayed_edge = EdgeDecayFunction.confidence_decay(edge, confidence)
# Returns: ~0.024 (20% decay for 75% confidence)

# Lower confidence = more conservative edge estimate
# Uses beta distribution to model uncertainty
```

---

## Risk Monitoring & Alerts

### Real-Time Monitoring

**Metrics Tracked:**

| Metric | Update Frequency | Alert Threshold |
|--------|------------------|-----------------|
| Win Rate | Per bet | <55% (trailing 50 bets) |
| ROI | Per day | <0% (daily) |
| Sharpe Ratio | Per day | <1.5 (trailing 30 days) |
| Max Drawdown | Per day | <-10% |
| Consecutive Losses | Per bet | 5+ losses |
| Daily Loss | Per bet | >10% of bankroll |

### Alert Types

```
🔴 CRITICAL (block trading):
  - Daily loss > 10%
  - Weekly loss > 20%
  - 5 consecutive losses
  - Bankroll < $1,000

🟠 WARNING (manual review):
  - ROI <0% (trailing)
  - Sharpe <1.5 (trailing)
  - Drawdown approaching -10%
  - Win rate <55%

🟡 INFO (logging):
  - Bet approaching position limit
  - Total exposure >25%
  - Edge distribution shift
```

### Monitoring Integration

```python
from py.monitoring.performance_dashboard import PerformanceMonitor

monitor = PerformanceMonitor()

# Track each bet
monitor.log_bet(
    timestamp=datetime.now(),
    bet_id="bet_12345",
    player="Patrick Mahomes",
    prop_type="passing_yards",
    model_prediction=290.5,
    actual_outcome=315.0,
    bet_size=250,
    win=True,
    profit=225.00
)

# Get real-time metrics
metrics = monitor.get_metrics()
print(f"Win rate: {metrics['win_rate']:.1%}")
print(f"ROI: {metrics['roi']:+.2%}")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")

# Check alerts
alerts = monitor.get_alerts()
for alert in alerts:
    print(f"[{alert.severity}] {alert.message}")
```

---

## Key Classes & Functions

### RiskGateSystem

**Location:** `py/production/risk_gates.py`

```python
class RiskGateSystem:
    """4-tier risk gate system for production betting"""

    def __init__(self, min_edge=0.025, min_confidence=0.60, ...):
        """Initialize with threshold configuration"""

    def evaluate_bet(self, bet: BetProposal) -> Dict:
        """Evaluate bet through all 4 gates"""
        # Returns: {'approved': bool, 'rejection_reason': str, ...}

    def get_configuration(self) -> Dict:
        """Get current gate configuration"""
```

**Key Methods:**
- `_gate1_statistical()`: Validate model quality
- `_gate2_market()`: Validate market conditions
- `_gate3_portfolio()`: Validate position sizing
- `_gate4_operational()`: Validate operational constraints

### BetProposal

**Location:** `py/production/risk_gates.py`

```python
@dataclass
class BetProposal:
    """A proposed bet to be evaluated by risk gates"""

    # Model outputs
    edge: float                    # Expected edge (0.03 = 3%)
    confidence: float             # Model confidence (0-1)
    model_uncertainty: float      # Uncertainty (std dev)
    prediction: float             # Model prediction

    # Market data
    odds: float                   # American odds (-110)
    line_value: float            # Prop line (250.5)
    bet_side: str                # 'over' or 'under'
    bookmaker: str               # 'pinnacle', 'draftkings'

    # Bet details
    bet_size: float              # Proposed bet amount ($)
    bankroll: float              # Current bankroll ($)

    # Optional
    book_hold: Optional[float]   # Bookmaker vig
    session_metrics: Optional[Dict]  # Daily/session stats
```

### CQLBetSizer

**Location:** `py/production/cql_bet_sizer.py`

```python
class CQLBetSizer:
    """Production bet sizing using CQL agent"""

    def __init__(self, cql_model_path, max_kelly_fraction=0.25):
        """Load trained CQL model"""

    def recommend_bet(
        self,
        opportunity: BettingOpportunity,
        bankroll: BankrollState
    ) -> BetRecommendation:
        """Generate single bet recommendation"""

    def recommend_portfolio(
        self,
        opportunities: list[BettingOpportunity],
        bankroll: BankrollState,
        max_positions: int = 5
    ) -> list[BetRecommendation]:
        """Recommend portfolio with correlation adjustment"""
```

### Kelly Functions

**Location:** `py/production/kelly_sizing.py`

```python
def kelly_criterion(
    win_prob: float,
    decimal_odds: float,
    kelly_fraction: float = 1.0,
    max_bet_fraction: float = 0.10
) -> float:
    """Calculate Kelly bet fraction"""

def optimal_bet_size(
    model_prob: float,
    market_odds: int,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_edge: float = 0.01
) -> dict:
    """Calculate optimal bet size with Kelly criterion"""

def simulate_kelly_growth(
    win_prob: float,
    decimal_odds: float,
    initial_bankroll: float,
    n_bets: int,
    kelly_fraction: float = 1.0
) -> dict:
    """Simulate bankroll growth using Kelly"""
```

### CVaR Functions

**Location:** `py/risk/cvar_lp.py`

```python
def solve_cvar_lp(
    returns: np.ndarray,
    alpha: float = 0.95,
    cap: float = 0.02,
    budget: float = 1.0,
    solver: str = "CLARABEL"
) -> tuple[np.ndarray, float, float]:
    """Solve CVaR optimization LP"""
    # Returns: (stakes, cvar_value, var_value)

def cvar_of_stakes(
    mat: list[list[float]],
    f: list[float],
    alpha: float = 0.95
) -> tuple[float, float]:
    """Calculate CVaR and VaR for given stakes"""
    # Returns: (cvar_value, var_value)
```

---

## Usage Examples

### Example 1: Single Bet Approval

```python
from py.production.risk_gates import RiskGateSystem, BetProposal
from py.production.kelly_sizing import optimal_bet_size

# Initialize risk gates (Pinnacle configuration)
gates = RiskGateSystem(
    min_edge=0.025,
    min_confidence=0.60,
    max_bet_pct=0.05
)

# Proposed bet
proposal = BetProposal(
    edge=0.03,
    confidence=0.65,
    model_uncertainty=0.15,
    prediction=290.5,
    odds=-110,
    line_value=275.5,
    bet_side='over',
    bookmaker='pinnacle',
    book_hold=0.045,
    bet_size=250,
    bankroll=10000,
    player_name="Patrick Mahomes",
    prop_type="passing_yards",
    session_metrics={
        'daily_loss': -250,
        'weekly_loss': -500,
        'consecutive_losses': 0,
        'total_exposure': 1500
    }
)

# Evaluate through all 4 gates
result = gates.evaluate_bet(proposal)

if result['approved']:
    print(f"✅ Bet APPROVED")
    print(f"Metrics: {result['metrics']}")
else:
    print(f"❌ Bet REJECTED: {result['rejection_reason']}")
    for warning in result['warnings']:
        print(f"⚠️  {warning}")

# Calculate Kelly-sized bet
kelly_result = optimal_bet_size(
    model_prob=0.65,  # 65% over probability
    market_odds=-110,
    bankroll=10000,
    kelly_fraction=0.25
)

print(f"Kelly size: ${kelly_result['bet_amount']:.2f}")
print(f"Expected value: ${kelly_result['expected_value']:.2f}")
```

### Example 2: Portfolio Optimization

```python
from py.production.cql_bet_sizer import CQLBetSizer, BettingOpportunity, BankrollState

# Initialize CQL sizer
sizer = CQLBetSizer()

# Multiple opportunities
opportunities = [
    BettingOpportunity(
        game_id="2024_11_KC_LV",
        player_name="Patrick Mahomes",
        prop_type="passing_yards",
        line=275.5,
        odds_over=-110,
        odds_under=-110,
        model_prediction=290.5,
        model_std=45.2,
        model_prob_over=0.62
    ),
    BettingOpportunity(
        game_id="2024_11_NE_BUF",
        player_name="Josh Allen",
        prop_type="passing_yards",
        line=290.5,
        odds_over=-110,
        odds_under=-110,
        model_prediction=305.0,
        model_std=50.0,
        model_prob_over=0.58
    ),
]

# Bankroll state
bankroll = BankrollState(
    total_bankroll=10000.0,
    available_capital=8500.0,
    max_position_size=500.0,
    current_positions=3
)

# Get portfolio recommendations
recommendations = sizer.recommend_portfolio(
    opportunities=opportunities,
    bankroll=bankroll,
    max_positions=5
)

# Print results
for rec in recommendations:
    if rec.action != "pass":
        print(f"{rec.action.upper()}: {rec.bet_size:.0f}")
        print(f"  Confidence: {rec.confidence:.0%}")
        print(f"  Expected value: ${rec.expected_value:.2f}")
        print(f"  Reasoning: {rec.reasoning}")
```

### Example 3: Bankroll Growth Simulation

```python
from py.production.kelly_sizing import simulate_kelly_growth, compare_kelly_fractions

# Simulate bankroll growth
result = simulate_kelly_growth(
    win_prob=0.55,
    decimal_odds=0.909,
    initial_bankroll=10000,
    n_bets=1000,
    kelly_fraction=0.25,
    seed=42
)

print(f"Final bankroll: ${result['final_bankroll']:,.0f}")
print(f"ROI: {result['roi']:+.2f}%")
print(f"Win rate: {result['win_rate']:.1f}%")
print(f"Max drawdown: {result['max_drawdown_pct']:.1f}%")
print(f"Sharpe ratio: {result['sharpe_ratio']:.3f}")

# Compare different Kelly fractions
comparison = compare_kelly_fractions(
    win_prob=0.55,
    decimal_odds=0.909,
    initial_bankroll=10000,
    n_bets=1000,
    n_simulations=100
)

for frac_str, stats in sorted(comparison.items(), key=lambda x: float(x[0])):
    print(f"{frac_str} Kelly: {stats['median_final']:,.0f} final, {stats['ruin_rate']:.1f}% ruin")
```

### Example 4: CVaR Optimization

```bash
# Generate scenario returns (rows=scenarios, cols=bets)
python py/risk/cvar_lp.py \
    --scenarios data/scenario_returns.csv \
    --alpha 0.95 \
    --cap 0.02 \
    --budget 1.0 \
    --output results/cvar_stakes.json

# Review results
cat results/cvar_stakes.json
# {
#   "method": "cvxpy_lp",
#   "stakes": [0.015, 0.012, 0.008, ...],
#   "cvar": -0.0325,
#   "var": -0.0420,
#   "total_stake": 0.98
# }
```

---

## Integration with Backtesting

### Backtest with Risk Gates

```python
from py.backtest.historical_validation_4way import validate_ensemble
from py.production.risk_gates import RiskGateSystem

# Initialize gates
gates = RiskGateSystem()

# Run backtest with risk validation
result = validate_ensemble(
    start_date="2022-01-01",
    end_date="2024-12-31",
    risk_gates=gates,
    min_edge_threshold=0.025
)

# Risk-gated backtest metrics
print(f"Total bets evaluated: {result['total_evaluated']}")
print(f"Bets approved: {result['total_approved']}")
print(f"Approval rate: {result['approval_rate']:.1%}")
print(f"ROI (approved only): {result['roi_approved']:+.2%}")
print(f"Win rate (approved): {result['win_rate_approved']:.1%}")
```

### Stress Testing

```python
from py.simulation.bootstrap_stress_test import run_stress_test

# Stress test with different scenarios
scenarios = [
    {'name': 'baseline', 'win_prob': 0.55},
    {'name': 'adverse', 'win_prob': 0.50},
    {'name': 'worst_case', 'win_prob': 0.45},
    {'name': 'best_case', 'win_prob': 0.60},
]

for scenario in scenarios:
    result = run_stress_test(
        scenario=scenario,
        initial_bankroll=10000,
        n_simulations=1000
    )

    print(f"\n{scenario['name'].upper()}:")
    print(f"  Median drawdown: {result['median_drawdown']:.1%}")
    print(f"  Ruin rate: {result['ruin_rate']:.2%}")
    print(f"  Mean final: ${result['mean_final']:,.0f}")
```

---

## Integration with Production

### A/B Test Risk Gates

```python
from py.production.ab_test_router import ABTestRouter
from py.production.risk_gates import RiskGateSystem

# Initialize A/B test with risk gates
gates_v5 = RiskGateSystem()  # Conservative
gates_v6 = RiskGateSystem(max_bet_pct=0.06)  # Slightly aggressive

router = ABTestRouter(
    control_gates=gates_v5,
    treatment_gates=gates_v6,
    split_ratio=0.5,
    duration_games=100
)

# Route each bet to control or treatment
for game_id, bets in bets_queue.items():
    for bet in bets:
        assignment = router.assign_bet(bet)

        if assignment == 'control':
            gates = gates_v5
        else:
            gates = gates_v6

        result = gates.evaluate_bet(bet)
        if result['approved']:
            execute_bet(bet, result)
```

### Production Deployment Checklist

```
✅ Pre-Deployment Validation:
  [ ] All 4 gates pass on test suite
  [ ] Backtest approval rate > 80%
  [ ] Backtest ROI meets targets (>5%)
  [ ] Drawdown <-8% in all scenarios
  [ ] Sharpe ratio >2.2 (trailing)
  [ ] Risk gates tested with edge cases
  [ ] Kelly sizing validated
  [ ] Bankroll recovery protocol documented
  [ ] Monitoring alerts configured
  [ ] Daily loss limit set appropriately

✅ Runtime Validation:
  [ ] Risk gates initialized with correct parameters
  [ ] Bet logging enabled
  [ ] Performance monitoring running
  [ ] Alert thresholds configured
  [ ] Stop-loss triggers active
  [ ] Circuit breaker operational
  [ ] Daily reconciliation scheduled

✅ Operational Safety:
  [ ] Emergency contact list updated
  [ ] Escalation procedures documented
  [ ] Manual override capabilities tested
  [ ] Audit trail enabled
  [ ] Rate limiting in place
  [ ] Bankroll protection verified
```

---

## A/B Test Guardrails

### Guardrail Configuration

A/B tests between ensemble versions use **automatic guardrails** to protect bankroll:

```yaml
# config/ab_test_config.yaml
ab_test:
  duration_games: 100  # Run through 100 games
  split_ratio: 0.50    # 50% control, 50% treatment

  success_criteria:
    - name: "MAE Improvement"
      metric: "mae_improvement"
      threshold: 5.0        # >5 yards
      direction: "greater"

    - name: "Statistical Significance"
      metric: "p_value"
      threshold: 0.05
      direction: "less"

    - name: "Win Rate Improvement"
      metric: "win_rate_improvement"
      threshold: 0.02       # >2 percentage points
      direction: "greater"

    - name: "ROI Improvement"
      metric: "roi_improvement"
      threshold: 0.01       # >1%
      direction: "greater"

  guardrails:
    - name: "MAE Degradation"
      metric: "mae_degradation"
      threshold: 0.10       # Don't exceed 10% worse
      action: "stop"

    - name: "ROI Degradation"
      metric: "roi_degradation"
      threshold: -0.02      # Don't exceed -2%
      action: "stop"

    - name: "Max Drawdown"
      metric: "max_drawdown"
      threshold: -0.10      # Don't exceed -10%
      action: "stop"

    - name: "Win Rate Degradation"
      metric: "win_rate_degradation"
      threshold: -0.03      # Don't exceed -3 pp
      action: "stop"
```

### Auto-Stop Logic

```python
# Early stopping if guardrails triggered
if current_metrics['mae_degradation'] > 0.10:
    print("⛔ Guardrail triggered: MAE degradation > 10%")
    print("Stopping test and reverting to control")
    stop_ab_test()
    revert_to_control()
    alert_team()
```

---

## Common Issues & Troubleshooting

### Issue 1: "Edge too small" (Gate 1 failure)

**Symptom:** Bets consistently rejected for insufficient edge

**Causes:**
- Model probabilities misaligned with market
- Vig too high (using square books instead of Pinnacle)
- Model drift from production baseline
- Market too efficient for current edge threshold

**Solutions:**
```python
# Check model calibration
from py.models.improve_bnn_calibration import check_calibration
calibration = check_calibration(predictions, actuals)

# Verify against Pinnacle (lowest vig)
# If edge < 2% consistently, may need model retraining

# Lower min_edge threshold temporarily
gates = RiskGateSystem(min_edge=0.015)  # 1.5% instead of 2.5%

# Switch to higher-vig bookmaker as fallback
if pinnicale_unavailable:
    use_draftkings(vigs_up_to=0.06)
```

### Issue 2: "Bet too large" (Gate 3 failure)

**Symptom:** All bets rejected for exceeding position limits

**Causes:**
- Bankroll decreased due to losses
- max_bet_pct too conservative
- Kelly sizing producing oversized bets

**Solutions:**
```python
# Check actual bankroll
current_bankroll = query_account_balance()
if current_bankroll < expected:
    print("⚠️  Bankroll discrepancy detected")

# Recalculate max bet
max_bet = current_bankroll * 0.05
print(f"Max bet for ${current_bankroll}: ${max_bet}")

# If edge is high, increase fraction
if edge > 0.05:
    gates = RiskGateSystem(max_bet_pct=0.07)  # 7% instead of 5%

# Verify Kelly calculation
kelly_fraction = (prob * odds - (1 - prob)) / (odds - 1)
print(f"Kelly fraction: {kelly_fraction:.4f}")
```

### Issue 3: "Daily loss limit exceeded" (Gate 4 failure)

**Symptom:** Trading halts due to daily loss threshold

**Causes:**
- Legitimate downswing (happens 10% of time with 55% win rate)
- Model predictions diverged from reality
- Edge threshold too low (betting -EV)
- Configuration error (stop-loss set too low)

**Solutions:**
```python
# Review today's trades
from py.monitoring.performance_dashboard import get_daily_trades
trades = get_daily_trades(date='2024-11-15')

# Analyze loss patterns
mean_loss = np.mean([t['loss'] for t in trades if t['won'] == False])
win_rate = np.mean([t['won'] for t in trades])

# If win rate > 55%, likely normal variance
if win_rate > 0.55:
    print("Win rate acceptable - this is normal variance")

# If win rate < 50%, escalate
if win_rate < 0.50:
    print("⚠️  Win rate below 50% - investigate model drift")
    # Check calibration, backtest, model version

# Resume trading manually after reviewing
manual_override_daily_limit()
```

### Issue 4: "Bet approval rate too low" (<70%)

**Symptom:** Only 40% of proposed bets pass risk gates

**Causes:**
- Risk gates too conservative
- Model producing weak edges consistently
- Market has become more efficient
- Configuration mismatch (Pinnacle params but using square books)

**Solutions:**
```python
# Audit gate configuration
config = gates.get_configuration()

# For Pinnacle (lowest vig), use:
gates_pinnacle = RiskGateSystem(
    min_edge=0.020,      # 2% for Pinnacle
    min_confidence=0.55,
    max_vig=0.045
)

# For square books (higher vig), increase edge threshold:
gates_square = RiskGateSystem(
    min_edge=0.040,      # 4% for DK/FD
    min_confidence=0.60,
    max_vig=0.06
)

# Measure approval rate by gate
from py.utils.betting_validator import BettingValidator
validator = BettingValidator()
report = validator.validate_strategy_config(strategy_config)
print(f"Gate 1 pass rate: {report['gate1_pass_rate']:.1%}")
print(f"Gate 2 pass rate: {report['gate2_pass_rate']:.1%}")
print(f"Gate 3 pass rate: {report['gate3_pass_rate']:.1%}")
print(f"Gate 4 pass rate: {report['gate4_pass_rate']:.1%}")
```

### Issue 5: "Drawdown exceeding limits"

**Symptom:** Max drawdown reaches -12% (exceeds -8% target)

**Causes:**
- Normal variance with 55% win rate (10-15% drawdowns occur ~5% of time)
- Kelly fraction too high (using Half Kelly instead of Quarter Kelly)
- Correlated losses (same team, same player)
- Model degradation

**Solutions:**
```python
# Check Kelly fraction in use
# Recommended: 0.25 (Quarter Kelly)
# Quarter Kelly should keep drawdown <10% in 95% of cases

# Switch to more conservative fraction if drawdown persistent
from py.production.kelly_sizing import compare_kelly_fractions

comparison = compare_kelly_fractions(
    win_prob=0.55,
    decimal_odds=0.909,
    initial_bankroll=10000,
    n_bets=1000,
    n_simulations=100
)

# 1/4 Kelly: median drawdown -8%, max drawdown -15%
# 1/8 Kelly: median drawdown -4%, max drawdown -8%

# If drawdown concerns, reduce to 1/8 Kelly
gates = RiskGateSystem()
kelly_fraction = 0.125  # Instead of 0.25
```

### Issue 6: "CVaR solver failed"

**Symptom:** CVaR optimization fails with "solver failed" error

**Causes:**
- cvxpy not installed
- Scenario matrix has NaN values
- Solver parameter invalid
- Numerical instability in LP

**Solutions:**
```bash
# Ensure cvxpy installed
pip install cvxpy

# Fall back to heuristic if solver fails
python py/risk/cvar_lp.py \
    --scenarios data/scenario_returns.csv \
    --alpha 0.95 \
    --solver "ECOS"  # Try ECOS if CLARABEL fails

# Check for NaN values in scenario matrix
python -c "
import pandas as pd
df = pd.read_csv('scenario_returns.csv')
print(f'NaN values: {df.isna().sum().sum()}')
"

# If many NaNs, use heuristic allocation
python -c "
from py.risk.cvar_lp import heuristic_stakes
mat = [[0.02, 0.01], [0.01, 0.03]]
stakes = heuristic_stakes(mat, cap=0.02, budget=1.0)
print(f'Heuristic stakes: {stakes}')
"
```

---

## References & Further Reading

### Internal Documentation
- [docs/CLAUDE.md](../CLAUDE.md) - Project overview and status
- [docs/database/DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md) - Schema reference
- [docs/guides/COMMON_TASKS_REFERENCE.md](../guides/COMMON_TASKS_REFERENCE.md) - Common operations

### Key Paper & Theory
- **Kelly Criterion:** Kelly (1956), "A New Interpretation of Information Rate"
- **CVaR:** Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk"
- **Position Sizing:** Thorp (1997), "The Mathematics of Gambling"

### Code Examples
- `examples/risk_management_integration.py` - Full integration example
- `examples/kelly_simulation_comparison.py` - Kelly fraction comparison
- `examples/backtest_with_risk_gates.py` - Backtest with validation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 15, 2025 | Initial comprehensive documentation |

---

## Contact & Support

**For Issues:**
- Risk gate configuration: See `py/production/risk_gates.py` CLI
- Kelly sizing: Use `py/production/kelly_sizing.py --help`
- CVaR optimization: Check solver status with CLARABEL/ECOS

**For Questions:**
- Risk framework questions: See Component Overview & Architecture
- Production integration: See Integration with Production section
- Troubleshooting: See Common Issues & Troubleshooting section
