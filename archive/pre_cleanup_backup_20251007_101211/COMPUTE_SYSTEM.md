# 🔥 NFL Analytics Distributed Compute System

A SETI@home-style distributed computing system for NFL analytics that keeps your laptop hot while doing useful work! Enhanced with **formal statistical testing frameworks** and **multi-armed bandit optimization**.

## Features

- **🧠 Multi-Armed Bandit Optimization** - UCB1, Thompson Sampling, Epsilon-Greedy strategies
- **📊 Statistical Testing Framework** - Permutation tests, effect sizes, multiple comparisons
- **🎯 Priority-based task queue** - Automatically selects highest ROI compute jobs
- **💾 Checkpointing & recovery** - Resume interrupted tasks from last checkpoint
- **🔥 Heat generation modes** - Configurable intensity (low/medium/high/inferno)
- **📈 Performance tracking** - Real-time progress with statistical significance testing
- **🖥️ Resource monitoring** - CPU, GPU, memory, and temperature tracking
- **🔬 Multiple task types** - RL training, Monte Carlo, state-space models, A/B tests

## Quick Start

```bash
# Initialize the task queue with standard compute jobs
python run_compute.py --init

# Start computing with adaptive scheduling (default)
python run_compute.py

# MAXIMUM HEAT GENERATION 🔥🔥🔥
python run_compute.py --intensity inferno

# Check progress
python run_compute.py --status

# View performance scoreboard
python run_compute.py --scoreboard

# Rebalance queue based on performance
python run_compute.py --rebalance

# Web dashboard with live monitoring
python run_compute.py --dashboard
```

## 🆕 Statistical Testing Framework

The compute system now includes a comprehensive statistical testing framework for rigorous model validation:

### Statistical Tests Available
- **Permutation Tests**: Non-parametric significance testing (5000+ permutations)
- **Bootstrap Methods**: Confidence intervals and effect size estimation
- **Bayesian Testing**: Thompson Sampling with posterior distributions
- **Classical Tests**: t-tests, ANOVA, chi-square (as fallbacks)

### Effect Size Calculations
- **Cohen's d**: Standardized mean difference with bias correction
- **Cliff's delta**: Non-parametric effect size for ordinal data
- **Eta-squared**: Proportion of variance explained
- **Odds ratios**: Effect sizes for binary outcomes

### Multiple Comparison Corrections
- **FDR Control**: Benjamini-Hochberg procedure
- **FWER Control**: Bonferroni, Holm-Bonferroni methods
- **Adaptive FDR**: Dynamic false discovery rate control

### Automated Reporting
- **Quarto Integration**: Reproducible research documents
- **LaTeX Tables**: Publication-ready statistical tables
- **R Integration**: Seamless R/Python statistical workflows
- **Methodology Documentation**: Automated methods sections

## 🆕 Multi-Armed Bandit Optimization

The adaptive scheduler now uses formal bandit algorithms for principled exploration-exploitation:

### Bandit Strategies
- **UCB1**: Upper Confidence Bound with statistical guarantees
- **Thompson Sampling**: Bayesian posterior sampling
- **Epsilon-Greedy**: Classic exploration with tunable ε
- **EXP3**: Exponential-weight algorithm for adversarial settings

### Bandit Features
- **Arm Management**: Each task type/configuration is a bandit arm
- **Reward System**: ROI-based rewards normalized to [0,1]
- **Confidence Intervals**: Statistical precision for mean rewards
- **Regret Minimization**: Formal guarantees on compute allocation

### Performance Analytics
- **Strategy Comparison**: Simulation-based bandit strategy evaluation
- **Exploration Efficiency**: Real-time exploration vs exploitation tracking
- **Dynamic Switching**: Hot-swap bandit strategies while preserving history
- **Cumulative Regret**: Track suboptimality over time

## Task Types

### 1. RL Training (High Priority)
- **DQN Extended Training**: 500-1000 epochs with multi-seed ensembles
- **PPO Training**: Advanced policy gradient with 10-20 seeds
- **Hyperparameter Sweeps**: Grid search over LR, gamma, batch size
- **Advanced Techniques**: Double DQN, prioritized replay

### 2. State-Space Models (Medium Priority)
- **Parameter Sweeps**: Process noise (q), observation noise (r), HFA
- **Kalman Smoother**: Time-varying home field advantage
- **LOYO Cross-validation**: Leave-one-year-out with uncertainty

### 3. Monte Carlo Simulations (Low Priority)
- **Large-scale Scenarios**: 100k to 1M simulations
- **Risk Metrics**: VaR, CVaR, max drawdown, Sharpe ratio
- **Bootstrap CIs**: 5000+ resamples for confidence intervals

### 4. OPE Evaluation (High Priority)
- **Robustness Grids**: Clip/shrink parameter sweeps
- **Multiple Methods**: SNIS, DR, WIS, CWPDIS
- **Bootstrap Testing**: 5000 iterations for stability

### 5. Model Calibration (Medium Priority)
- **Platt/Isotonic Scaling**: Repeated k-fold CV
- **Feature Ablations**: L1 regularization, enhanced features
- **Reliability Metrics**: Brier score, ECE, calibration plots

### 6. Copula Fitting (Low Priority)
- **Multiple Families**: Gaussian, t, Clayton, Gumbel
- **Goodness-of-fit**: Bootstrap CvM tests
- **Time-varying Parameters**: Weekly/team-specific calibration

## Performance Tracking & Adaptive Scheduling

The system tracks performance metrics over time and uses this data to intelligently prioritize tasks:

### Performance Metrics
- **Improvement Tracking**: Monitors performance deltas from baseline
- **Regression Detection**: Statistical tests to detect performance drops
- **ROI Calculation**: Performance improvement per compute hour
- **Trend Analysis**: Identifies improving, regressing, or plateauing models

### Adaptive Features
- **Expected Value Calculation**: `EV = (expected_improvement × importance) / compute_hours`
- **Dynamic Prioritization**: Tasks with high EV get higher priority
- **Smart Task Generation**: Suggests follow-up tasks for improving models
- **Exploration/Exploitation**: 20% exploration of new configurations

### Performance Dashboard
The web dashboard shows:
- **Live Scoreboard**: Current vs best performance for all models
- **Heat Meter**: System temperature and resource usage
- **Task Value Matrix**: Pending tasks ranked by expected value
- **Recommendations**: Compute allocation suggestions
- **Milestone Timeline**: Recent breakthroughs and regressions

## Architecture

```
run_compute.py              # Main entry point
│
├── py/compute/
│   ├── task_queue.py      # SQLite-based persistent queue
│   ├── performance_tracker.py    # Performance metrics tracking
│   ├── adaptive_scheduler.py     # Intelligent task prioritization
│   ├── dashboard.py       # Web-based monitoring dashboard
│   ├── worker.py          # Task executor with GPU support
│   └── tasks/
│       ├── rl_trainer.py         # DQN/PPO training
│       ├── state_space_trainer.py # Kalman filter sweeps
│       ├── monte_carlo_runner.py  # Risk simulations
│       ├── ope_evaluator.py      # Off-policy evaluation
│       ├── model_calibrator.py   # GLM/XGB calibration
│       └── copula_fitter.py      # Copula GOF testing
│
└── compute_queue.db       # Persistent task & performance database
```

## Intensity Levels

- **Low**: Gentle compute with delays (60-70°C)
- **Medium**: Balanced performance (70-80°C)
- **High**: All CPU cores, max threads (80-85°C)
- **Inferno**: Maximum everything! 🔥 (85-95°C)

## Database Schema

The system uses SQLite for persistent task management:

```sql
tasks:
  - id (UUID)
  - name (task description)
  - type (rl_train, monte_carlo, etc.)
  - priority (1-5, lower is higher)
  - status (pending/running/completed/failed)
  - config (JSON parameters)
  - progress (0.0-1.0)
  - checkpoint_path
  - cpu_hours, gpu_hours

compute_stats:
  - timestamp
  - cpu_usage, gpu_usage
  - memory_usage
  - temperature
  - active_tasks
```

## Monitoring

### Command Line
```bash
# Check queue status
python run_compute.py --status

# Output:
📊 NFL ANALYTICS COMPUTE QUEUE STATUS
=====================================
📈 Queue Overview:
  Total tasks: 87
  Pending: 75 ⏳
  Running: 2 🔥
  Completed: 10 ✅
  Failed: 0 ❌

⚡ Resource Usage:
  CPU hours: 24.3
  GPU hours: 12.1

🔥 Currently Running:
  • DQN Training (seed=3): 45.2% complete
  • Monte Carlo (500,000 scenarios): 78.9% complete

🌡️ System Stats (latest):
  CPU usage: 87.3%
  Memory usage: 62.1%
  Temperature: 82.5°C
  GPU usage: 95.2%
```

### Web Dashboard

Visit http://localhost:5000 for real-time monitoring

## Adding Custom Tasks

```python
from py.compute.task_queue import TaskQueue, TaskPriority

queue = TaskQueue()

# Add a high-priority RL training task
task_id = queue.add_task(
    name="Custom DQN Experiment",
    task_type="rl_train",
    config={
        "model": "dqn",
        "epochs": 1000,
        "seed": 99,
        "double_dqn": True,
        "lr": 5e-5
    },
    priority=TaskPriority.HIGH,
    estimated_hours=3.0
)

print(f"Added task: {task_id}")
```

## Performance Notes

- DQN training: ~2 hours for 500 epochs
- Monte Carlo: ~1 hour per 100k scenarios
- State-space sweep: ~30 min per parameter combo
- OPE bootstrap: ~20 min for 5000 iterations
- Copula GOF: ~90 min for 10k bootstraps

## Tips for Maximum Heat

1. Close all unnecessary apps
2. Place laptop on hard surface (not bed/couch)
3. Use `--intensity inferno` mode
4. Run multiple compute-heavy tasks simultaneously
5. Disable any power-saving features
6. Consider external cooling pad for extended runs

## Safety

The system monitors temperature and will show warnings if:
- CPU temp > 90°C
- GPU temp > 85°C
- Memory usage > 90%

To stop gracefully, press Ctrl+C. The current task will checkpoint and stop.

---

**Warning**: Running on `inferno` mode for extended periods will generate significant heat. Ensure adequate ventilation! 🔥

**Note**: This system is designed for research/experimentation. For production workloads, consider using proper distributed computing infrastructure.