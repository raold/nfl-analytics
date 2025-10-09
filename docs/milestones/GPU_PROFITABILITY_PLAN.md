# GPU-Accelerated Profitability Enhancement Plan
**From Calibration to Cash: Closing the 1.4% Gap**

## Executive Summary

**Current State** (Dissertation Complete):
- ✅ Brier Score: 0.2515 (excellent calibration)
- ✅ CLV: +14.9 basis points (beating closing lines)
- ❌ Win Rate: 51.0% (below 52.4% breakeven at -110 odds)
- ❌ ROI: -7.5%, Sharpe: -1.22 (unprofitable)

**The Gap**: Need +1.4% win rate improvement **OR** access to lower-vig markets

**The Plan**: 12-week GPU-accelerated research program to bridge the gap via:
1. **Advanced RL** (Conservative Q-Learning, Implicit Q-Learning)
2. **Uncertainty-Aware Betting** (filter low-confidence bets)
3. **Neural Simulation** (stress test policies before deployment)
4. **Alternative Markets** (2% vig exchanges, player props)

**Compute Resources**:
- MacBook M4 (10-core GPU, MPS): Development + light training
- 2× RTX 5090 (96GB VRAM): Heavy training + hyperparameter sweeps
- Architecture: Unified Redis task queue (hardware-agnostic)

**Expected Outcomes**:
- **Conservative** (70% confidence): Win rate 51.0% → 52.8%, ROI +0.8%, Sharpe +0.45
- **Optimistic** (40% confidence): Win rate 51.0% → 53.5%, ROI +2.5%, Sharpe +1.15
- **Quick Win** (60% confidence): 2% vig exchanges → +1.5% to +3.0% ROI **with current models**

---

## The Current Problem: Market Efficiency

The dissertation demonstrates **semi-strong form market efficiency** in NFL betting:

> "Our models achieve strong calibration (Brier = 0.2515) and beat closing lines on average (CLV = +14.9 bps). Yet they lose money (ROI = -7.5%, Sharpe = -1.22). This is not a failure of implementation—it is a demonstration of market efficiency."

**Translation**:
- Our predictions are better than other bettors (positive CLV)
- But not better enough to overcome the house edge (4.5% vig at -110 odds)
- To profit at -110, we need 52.4% win rate; we achieve 51.0%

**The Path Forward**:
1. Improve win rate by +1.4% via GPU-intensive methods
2. **OR** deploy to lower-vig markets where 51.0% is already profitable

---

## Phase 0: Distributed Compute Infrastructure (Week 1)

### Architecture: Hardware-Agnostic Task Queue

**Why This Matters**:
- M4 MacBook is available now for development
- RTX 5090 workstation may not always be accessible
- Need to utilize whatever compute is available without code changes

**Implementation**:

```
┌─────────────┐
│   Redis     │ ← Central task queue (network-accessible)
│   Queue     │
└─────────────┘
      ↓
    Task Distribution
      ↓
   ┌────────────────────┬─────────────────────┐
   ↓                    ↓                     ↓
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ M4 Worker│    │ RTX Worker 1 │    │ RTX Worker 2 │
│  (MPS)   │    │   (CUDA:0)   │    │   (CUDA:1)   │
└──────────┘    └──────────────┘    └──────────────┘
     │                 │                    │
     └─────────────────┴────────────────────┘
                       ↓
              Shared Model Registry
           (Checkpoints, metrics, configs)
```

**Key Components**:

1. **Training Task Specification** (`py/compute/tasks/training_task.py`)
   ```python
   class TrainingTask:
       task_id: str
       model_type: str  # "cql", "iql", "gnn", "transformer"
       config: dict     # hyperparameters
       priority: int    # 1-10, RTX claims priority ≥ 5
       min_gpu_memory: int  # GB, worker checks before claiming
       estimated_hours: float
   ```

2. **Enhanced Worker** (`py/compute/worker_enhanced.py`)
   - Auto-detects device: CUDA > MPS > CPU
   - Claims tasks only if GPU memory sufficient
   - Checkpoints every N epochs
   - Graceful shutdown on SIGTERM

3. **Model Registry** (`py/compute/model_registry.py`)
   - Stores checkpoints with versioning
   - Syncs between M4 and RTX (rsync, S3, or NFS)
   - Metadata: config, metrics, device used, timestamp

**Starting Workers**:
```bash
# On M4 MacBook:
python py/compute/worker_enhanced.py --config config.yaml --worker macbook

# On RTX Workstation:
CUDA_VISIBLE_DEVICES=0 python py/compute/worker_enhanced.py --worker rtx_gpu0
CUDA_VISIBLE_DEVICES=1 python py/compute/worker_enhanced.py --worker rtx_gpu1
```

**Submitting Jobs** (from either machine):
```bash
python py/compute/submit_sweep.py \
  --model cql \
  --config sweeps/cql_alpha_sweep.yaml \
  --priority 10 \
  --min-gpu-memory 32  # RTX-only
```

**Deliverables**:
- [ ] Redis installed and accessible
- [ ] Worker auto-detection working (MPS/CUDA/CPU)
- [ ] Model registry syncing checkpoints
- [ ] End-to-end test: M4 submit → RTX execute → checkpoint back to M4

**Timeline**: 3-4 days (concurrent with CQL development)

---

## Phase 1: Advanced Offline RL (Weeks 2-4, P0)

### Goal: Improve win rate 51.0% → 52.5%+ via superior policy learning

**Why Conservative Q-Learning (CQL)?**
- Offline RL (no exploration, learn from logged data only)
- Prevents overestimation of Q-values on out-of-distribution actions
- Critical for betting: can't explore bad bets in production
- Literature: CQL improves offline performance by 10-30%

**CQL Implementation** (`py/rl/cql_agent.py`, ~800 LOC):

```python
class CQLAgent:
    def __init__(self, state_dim=6, action_dim=4, alpha=1.0):
        self.qnet = QNetwork(
            state_dim,
            action_dim,
            hidden=[256, 256, 256, 256],  # 4-6 layers
            dropout=0.2
        )
        self.alpha = alpha  # Conservative penalty

    def cql_loss(self, states, actions, rewards, next_states):
        # Standard Q-learning loss
        q_loss = F.mse_loss(Q_pred, Q_target)

        # Conservative penalty: push down Q on unobserved actions
        logsumexp = torch.logsumexp(self.qnet(states), dim=1)
        cql_penalty = (logsumexp - Q_pred).mean()

        return q_loss + self.alpha * cql_penalty
```

**Hyperparameter Sweep** (135 configs):
| Parameter | Values | Count |
|-----------|--------|-------|
| alpha (conservatism) | [0.1, 0.5, 1.0, 2.0, 5.0] | 5 |
| learning_rate | [1e-5, 5e-5, 1e-4] | 3 |
| network_depth | [4, 5, 6 layers] | 3 |
| hidden_units | [256, 384, 512] | 3 |
| **Total** | | **135** |

**Execution**:
- RTX: Trains full configs (4-6 hours each)
- M4: Trains small configs (2-3 layers) as backup
- **Time**: 135 × 5h = 675 GPU-hours → **17 days on 2×RTX @ 80% utilization**

**Implicit Q-Learning (IQL)** (`py/rl/iql_agent.py`, ~700 LOC):
- Alternative approach: expectile regression (τ=0.7)
- Avoids distributional shift differently than CQL
- 90 configs, runs in parallel with CQL

**Meta-Policy Ensemble** (`py/rl/meta_policy.py`, ~300 LOC):
- Combine predictions from DQN, PPO, CQL, IQL
- Weight learning via Thompson sampling
- Expected gain: +0.2-0.4% from diversity

**Validation Criteria**:
- Win rate ≥ 52.0% on held-out test set (2022-2024)
- OPE lower bound > 0 (off-policy evaluation confidence)
- Simulator acceptance (CVaR, drawdown thresholds)

**Expected Gain**: +0.5-1.0% win rate (CQL) + 0.3-0.8% (IQL) + 0.2-0.4% (ensemble) = **+1.0-2.2% total**

**Compute**: 900 RTX GPU-hours + 100 M4 GPU-hours

---

## Phase 2: Uncertainty-Aware Selective Betting (Weeks 5-6, P0)

### Goal: Filter low-confidence bets to boost win rate on deployed capital

**Key Insight**: Not all bets are created equal. By only betting when the ensemble is confident, we can:
- Reduce bet volume by 30-50%
- Increase win rate on remaining bets by 1.5-2.5%
- Target: 53-54% win rate on high-confidence subset

**Ensemble Prediction Uncertainty** (`py/models/ensemble_uncertainty.py`, ~500 LOC):

Train 10-20 diverse models:
| Model Type | Count | Training Device | Time |
|------------|-------|-----------------|------|
| GLM (various α) | 5 | M4 | 2h total |
| Shallow XGBoost | 5 | M4 | 3h total |
| Deep XGBoost | 3 | RTX | 6h |
| Deep NN | 3 | RTX | 8h |
| GNN | 1 | RTX | 20h (Phase 4) |
| Transformer | 1 | RTX | 30h (Phase 5) |

**Uncertainty Metric**: Prediction variance across ensemble
```python
def compute_uncertainty(predictions):
    return np.std(predictions)  # Higher = less confident
```

**Betting Gate**:
```python
def should_bet(ensemble_preds, edge):
    uncertainty = np.std(ensemble_preds)
    agreement = (ensemble_preds > 0.5).sum() / len(ensemble_preds)

    return (
        edge > 0 and  # Positive expected value
        uncertainty < 0.08 and  # High confidence
        agreement > 0.7  # 70%+ models agree on direction
    )
```

**Bayesian Neural Network** (`py/models/bnn_predictor.py`, ~400 LOC):
- Monte Carlo Dropout: 50 forward passes per prediction
- Epistemic uncertainty quantification
- GPU acceleration critical for inference speed

**Stake Sizing Integration**:
```python
kelly_fraction = kelly_criterion(edge, win_prob)
uncertainty_adjustment = 1 - normalize(epistemic_uncertainty)
final_stake = kelly_fraction * uncertainty_adjustment
```

**Expected Impact**:
- 30-50% fewer bets (filter low-confidence)
- 1.5-2.5% win rate increase on remaining bets
- **Net Effect**: 51.0% overall → 53-54% on deployed bets

**Compute**: 200 RTX GPU-hours (mostly deep models)

---

## Phase 3: Neural Simulator for Stress Testing (Weeks 7-8, P1)

### Goal: Validate policies under extreme scenarios before deployment

**Problem**: Historical backtests don't cover all possible futures
- What if 2025 has many underdog upsets (like 2008)?
- What if favorites dominate (reduce close games)?
- What if key QBs get injured in clusters?

**Solution**: Train a transformer to generate synthetic game outcomes

**Transformer Game Outcome Generator** (`py/simulation/neural_simulator.py`, ~900 LOC):

**Architecture**: GPT-style decoder
- **Small** (M4-trainable): 6 layers, 384 dim, 50M params
- **Large** (RTX-only): 12 layers, 768 dim, 200M params

**Training Data**:
- 5,529 historical games
- Play-by-play sequences (if available)
- Conditional on: week, teams, spread, total, weather

**Usage**:
```python
# Generate 10,000 synthetic seasons
simulator = NeuralSimulator.load("models/transformer_large.pth")
for scenario in ["underdog_upsets", "favorite_blowouts", "injured_qbs", "weather_extremes"]:
    synthetic_seasons = simulator.generate(
        n_seasons=10000,
        scenario=scenario,
        conditional={...}
    )

    # Validate policy on synthetic data
    policy_performance = evaluate_policy(policy, synthetic_seasons)
    assert policy_performance["cvar_95"] < 0.15
    assert policy_performance["max_drawdown"] < 0.25
```

**Stress Test Scenarios**:
1. **Underdog Upsets**: Increase upset probability by 20%
2. **Favorite Blowouts**: Reduce close games (±3 points) by 30%
3. **Injured-QB Cascades**: Simulate mass uncertainty
4. **Weather Extremes**: High wind, blizzards, heat

**Pass Criteria**: Policy must achieve CVaR₉₅ < 15%, Max DD < 25% in 95% of scenarios

**Policy Refinement Loop**:
- If policy fails stress tests → add scenario-specific constraints
- Re-train RL with augmented data (synthetic + real)
- Iterate until all scenarios pass

**Compute**: 300 RTX GPU-hours (transformer training + 10K simulations) + 50 M4 GPU-hours (validation)

---

## Phase 4-7 Summary (Weeks 9-12)

**Phase 4: Graph Neural Networks** (100 RTX-hours + 20 M4-hours)
- Capture transitive strength (team A beat B, B beat C → A likely beats C)
- 672 nodes (32 teams × 21 seasons), 5,529 edges (games)
- Expected gain: +0.2-0.5% win rate

**Phase 5: Transformers** (200 RTX-hours + 30 M4-hours)
- Temporal feature extraction: encode last N games per team
- Multi-game policy: allocate bankroll across 16 games/week jointly
- Expected gain: +0.3-0.7% win rate

**Phase 6: Production Monitoring** (60 hours over 12 weeks, mostly M4)
- Automated retraining every 4 weeks
- Drift detection & alerting
- Online learning with rollback capability

**Phase 7: Alternative Markets** (150 RTX-hours + 10 M4-hours)
- **QUICK WIN**: Simulate 2% vig exchanges → **already profitable!**
- Player props modeling (less efficient markets)
- Expected ROI: +2-4% on props

---

## Resource Summary

### Total Compute Budget
- **RTX 5090 Hours**: 1,410 hours
  - 37 days on 2× GPUs @ 80% utilization
  - Cloud equivalent: $45,120 (AWS p4d.24xlarge @ $32/hour)
  - **Your cost**: $0 (hardware owned)

- **M4 MacBook Hours**: 210 hours
  - Development, prototyping, light training
  - Saves ~20% RTX time by handling easy tasks

### Development Effort
- **New Code**: ~6,500 LOC across 7 phases
- **Testing**: ~2,000 LOC (pytest suites)
- **Documentation**: Pipeline docs, runbooks, API reference

### Timeline
- **Optimistic**: 10 weeks (if RTX runs 24/7)
- **Realistic**: 12 weeks (accounting for debugging, holidays)
- **Conservative**: 16 weeks (if RTX availability limited)

---

## Expected Outcomes & ROI

### Win Rate Improvements (Cumulative)
| Source | Conservative | Optimistic |
|--------|--------------|------------|
| CQL/IQL RL | +0.5% | +1.0% |
| Uncertainty Filtering | +1.0% | +2.0% |
| GNN | +0.2% | +0.5% |
| Transformers | +0.3% | +0.7% |
| **Total** | **+2.0%** | **+4.2%** |

### Profitability Scenarios

**Scenario A: Standard Books (-110 odds, 4.5% vig)**
- Current: 51.0% win rate → -7.5% ROI
- Conservative (+1.8%): 52.8% → **+0.8% ROI**, Sharpe +0.45
- Optimistic (+2.5%): 53.5% → **+2.5% ROI**, Sharpe +1.15

**Scenario B: Exchanges (2% vig)**
- Current: 51.0% win rate → **+1.5% ROI** (already profitable!)
- Conservative (+1.8%): 52.8% → **+3.5% ROI**
- Optimistic (+2.5%): 53.5% → **+5.0% ROI**

**Scenario C: Player Props**
- Literature: Props 30% less efficient than spreads
- Expected: 53-55% win rate → **+3-5% ROI**

### Annual Return on $10,000 Bankroll

| Market | Conservative | Optimistic |
|--------|--------------|------------|
| Standard Books | $10,800 (8% growth) | $12,500 (25% growth) |
| Exchanges | $13,500 (35% growth) | $15,000 (50% growth) |
| Props | $13,000 (30% growth) | $15,000 (50% growth) |
| **Combined** | $12,000-14,000 | $14,000-18,000 |

---

## Risk Mitigation

### Technical Risks
1. **Overfitting**:
   - Enforced 80/20 train/test split
   - OPE lower bounds must be positive
   - Simulator acceptance required (10K scenarios)

2. **Compute Failures**:
   - Checkpoint every epoch
   - Resume from checkpoint
   - Graceful degradation to CPU

3. **Market Changes**:
   - Weekly drift monitoring
   - Automatic model refresh every 4 weeks
   - Rollback capability if OPE worsens

### Operational Risks
4. **Regulatory**:
   - Focus on research/paper trading first
   - Document responsible gambling limits
   - Geographic restrictions awareness

5. **Liquidity**:
   - Start with 10-20% bankroll on exchanges
   - Monitor fill rates and slippage
   - Scale up gradually over 12 weeks

---

## Week 1 Checklist (Phase 0)

### Day 1-2: Infrastructure
- [ ] Install Redis on network-accessible machine
- [ ] Update `worker_enhanced.py` with device auto-detection
- [ ] Create `model_registry.py` for checkpoint sharing
- [ ] Test worker startup on M4 and RTX (if available)

### Day 3-4: CQL Implementation
- [ ] Implement `cql_agent.py` skeleton with conservative Q-learning
- [ ] Add checkpoint save/load
- [ ] Create hyperparameter sweep config (`sweeps/cql_sweep.yaml`)
- [ ] Test on M4 with toy config (1 layer, 64 hidden, 100 epochs)

### Day 5: Integration Testing
- [ ] Submit 1 CQL job to queue from M4
- [ ] Verify worker claims and executes (on M4 or RTX)
- [ ] Monitor via dashboard
- [ ] Validate checkpoint syncing

### Weekend: Scale Up (if RTX available)
- [ ] Submit full CQL sweep (135 configs, priority 10)
- [ ] Submit IQL sweep (90 configs, priority 10) in parallel
- [ ] Let workers run over weekend
- [ ] Monitor progress remotely

---

## Success Criteria

### Must Achieve (P0)
- [ ] CQL/IQL agents trained; best config identified
- [ ] Ensemble uncertainty reduces bet volume 30%+
- [ ] Neural simulator validates policy (10K scenarios pass)
- [ ] Test set win rate ≥ 52.5%
- [ ] Positive ROI in exchange simulation (2% vig)

### Should Achieve (P1)
- [ ] GNN improves ensemble Brier ≥ 0.001
- [ ] Transformer features improve win rate ≥ +0.3%
- [ ] Monitoring pipeline deployed
- [ ] Task queue coordinates M4 + RTX heterogeneously

### Nice to Have (P2)
- [ ] Real-money pilot on exchange (small stakes)
- [ ] Research paper submission
- [ ] Open-source framework release

---

## Next Steps

**Immediate** (This Week):
1. Review and approve this plan
2. Setup Redis on accessible machine
3. Begin CQL implementation on M4
4. Test worker on M4 (RTX not required yet)

**Week 2** (When RTX Available):
1. Deploy workers on RTX workstation
2. Submit CQL hyperparameter sweep
3. M4 continues: GLM ensemble training, validation
4. Monitor via dashboard

**Weeks 3-12**:
- Execute phases 1-7 in parallel
- M4: Development, light training, monitoring
- RTX: Heavy training (RL, GNN, transformers)
- Continuous integration testing

**The Goal**: Transform "impressive but unprofitable research" into **"production-ready profitable system"** by systematically applying GPU compute to close the 1.4% gap.
