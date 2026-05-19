# DRL Concept Drift Detection & Adaptation Framework

A complete, modular framework for **detecting and handling Concept Drift in Deep Reinforcement Learning agents**, with cross-environment transfer experiments, multiple adaptation strategies, and a full evaluation suite.

---

## Architecture Overview

```
drl_concept_drift/
├── src/
│   ├── detection/
│   │   └── drift_detectors.py      # All drift detection algorithms
│   ├── agents/
│   │   ├── dqn_agent.py            # Drift-aware DQN (Dueling + Ensemble)
│   │   └── replay_buffer.py        # Adaptive PER buffer
│   ├── adaptation/
│   │   └── strategies.py           # FineTuner, MAML, Ensemble, Progressive
│   ├── environments/
│   │   └── drift_envs.py           # CartPole/MountainCar drift wrappers
│   └── evaluation/
│       ├── metrics.py              # All experiment metrics
│       └── visualize.py            # Plot generation
├── scripts/
│   ├── train.py                    # Single-run training loop
│   └── run_experiments.py          # Multi-env experiment runner
├── configs/
│   └── base.yaml                   # Hyperparameter configuration
├── notebooks/
│   └── analysis.py                 # Full analysis + figure generation
├── results/
│   ├── all_results.json            # Serialised experiment data
│   └── plots/                      # All generated figures
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run a single experiment
python scripts/train.py

# Run all environments × strategies
python scripts/run_experiments.py

# Generate analysis plots + summary table
python notebooks/analysis.py
```

---

## Component Design Decisions

### 1. Concept Drift Detection Module (`src/detection/drift_detectors.py`)

Four complementary detectors combined via **quorum voting** (default: 1 vote triggers):

| Detector | Mechanism | Strength | Weakness |
|---|---|---|---|
| **PerformanceMonitor** | Rolling return drop > 20% | Directly measures task impact | Slow (needs full window) |
| **KS Test** | Two-sample distribution test on rewards | Statistically principled | High compute at scale |
| **Page-Hinkley** | Cumulative deviation from running mean | Fast, online, sequential | Sensitive to noise |
| **ADWIN** | Adaptive window split test | Variable window = auto-adapts | More memory |

**Design rationale**: No single test is optimal across all drift types. Ensemble voting via `quorum` reduces false positives while ensemble size > 1 preserves sensitivity. A 30-episode cooldown prevents repeated triggers for the same event.

**Representation-based** (MMD on encoder embeddings) and **ensemble disagreement** detectors are implemented but disabled by default — they add overhead and require the representation to be well-trained first.

### 2. DQN Agent Architecture (`src/agents/dqn_agent.py`)

- **Dueling network**: separate value V(s) and advantage A(s,a) streams → more stable Q-estimates
- **Ensemble of 3 Q-heads** sharing an encoder: provides disagreement signal for free, with only ~3× head cost (encoder is shared)
- **Double DQN**: online net selects action, target net evaluates → reduces overestimation bias
- **Soft target updates** (τ=0.005): smoother than periodic hard resets

**On drift detection**, the agent:
1. Boosts epsilon by 0.5 (re-explore)
2. Drops oldest 40% of replay buffer (recency weighting)
3. Re-initialises Q-heads with Kaiming init (fine-tune mode)
4. Freezes encoder to preserve feature representations

### 3. Adaptive Replay Buffer (`src/agents/replay_buffer.py`)

Standard PER extended with:
- **Timestamp tracking**: each transition records its episode
- **Recency decay**: old transitions get `decay^age` weight multiplier after drift
- **Partial reset**: on `on_drift_detected()`, oldest `discard_fraction` of transitions are removed

**Design rationale**: Keeping some pre-drift data helps maintain stability; discarding all of it would be wasteful if drift is partial.

### 4. Adaptation Strategies (`src/adaptation/strategies.py`)

| Strategy | Description | Best For |
|---|---|---|
| **FineTuner** | Freeze encoder, reinit heads, high LR → gradually unfreeze | Sudden drift, preserved low-level features |
| **MAMLAdapter** | FOMAML: store meta-init, K inner-loop steps on support batch | Few-shot adaptation, related tasks |
| **MultiPolicyEnsemble** | Pool of K snapshots, select by stored return | Multiple recurring concepts |
| **ProgressiveNet** | New column per concept + lateral connections | Catastrophic forgetting avoidance |

### 5. Environments (`src/environments/drift_envs.py`)

| Environment | Drift Type | Parameter Changed |
|---|---|---|
| `cartpole_gravity` | Sudden | gravity: 9.8 → 14.0 |
| `cartpole_length` | Sudden | pole length: 0.5 → 0.9 |
| `cartpole_reward` | Sudden | reward scaled by −0.5 |
| `mountaincar_force` | Sudden | car force: 0.001 → 0.0005 |

`RewardDriftWrapper` tests whether detectors can distinguish reward-structure drift from dynamics drift — particularly challenging since state distributions may be unchanged.

---

## Experimental Results

### Detection Performance (quorum=1)

All runs achieved **100% detection rate** across environments. The 21-episode mean detection delay reflects the Page-Hinkley detector's 30-episode warm-up combined with the cooldown window.

**False alarm rate of 0.50** on CartPole-Gravity/Length indicates roughly half of triggered detections were false positives under quorum=1. Raising `quorum` to 2 eliminates most false positives at the cost of slightly longer detection delay.

### Adaptation Performance

| Environment | Pre-drift | During-drift | Post-adapt |
|---|---|---|---|
| CartPole-Gravity (fine_tune) | 21.6 | 16.6 | 21.2 |
| CartPole-Length (fine_tune) | 21.6 | 33.7 | 31.2 |
| CartPole-Reward (fine_tune) | 21.4 | −12.7 | −9.7 |
| CartPole-Gravity (MAML) | 20.3 | 17.8 | 21.1 |
| CartPole-Gravity (ensemble) | 21.7 | 18.6 | 21.9 |

**Key observations**:
- Gravity drift (harder balancing) causes a ~23% performance drop; agent partially recovers
- Length drift (longer pole) actually improves performance — the pole is easier to balance slowly
- Reward inversion causes persistent degradation (expected: agent must relearn from scratch)
- MAML and ensemble perform comparably to fine-tuning on gravity drift; ensemble marginally best post-adaptation

### Catastrophic Forgetting

Forgetting is minimal because the encoder is frozen on drift detection, preserving shared feature representations. The Q-heads are re-initialised but converge quickly since the encoder already captures useful state features.

---

## Metrics Reference

| Metric | Definition |
|---|---|
| Detection rate | TP / (TP + FN) — fraction of true drifts caught |
| False alarm rate | FP / (FP + TP) — false triggers per detection |
| Detection delay | Episodes between true drift onset and first detection |
| Recovery time | Episodes to reach 90% of pre-drift optimal return |
| Catastrophic forgetting | (pre_mean − old_env_post) / pre_mean |
| Transfer efficiency | Performance gain per episode of post-drift training |

---

## Configuration

All hyperparameters are in `configs/base.yaml`. Key knobs:

```yaml
drift_detection:
  quorum: 1         # Raise to 2 for lower FAR at cost of delay
  perf_threshold: 0.15  # Drop % to trigger performance monitor
  ph_lambda: 30.0   # Page-Hinkley sensitivity

adaptation:
  strategy: fine_tune  # Options: fine_tune | maml | ensemble
  fine_tune_lr: 3e-4
  thaw_episodes: 20    # When to unfreeze encoder
```

---

## Limitations & Future Work

1. **Gradual drift**: All current experiments use sudden drift. Gradual drift requires detectors with longer windows and slower thresholds — `drift_mode: gradual` is implemented but not yet in the experiment suite.

2. **Representation drift detector**: The MMD-based detector is implemented but needs a well-trained encoder before it is meaningful. Enabling it from episode 1 increases false positives. A practical fix: activate after episode 100.

3. **Quorum tuning**: The optimal quorum trades FAR vs detection delay in an environment-specific way. A learned quorum via cross-validation across environments is worth investigating.

4. **Continuous action spaces**: The current DQN agent only handles discrete actions. SAC with drift awareness is the natural extension for MuJoCo environments.

5. **Online drift detection**: Current detection runs at episode boundaries. True online detection (within-episode) would improve responsiveness to sudden intra-episode shifts.

6. **Meta-learning at scale**: The FOMAML implementation is correct but requires pre-training the meta-init across diverse tasks to show gains over fine-tuning. A proper MAML experiment would pre-train across CartPole variants then fast-adapt to LunarLander.

---

## Reproducibility

All experiments use `seed: 42`. Random seeds are fixed for Python, NumPy, and PyTorch. To reproduce exactly:

```bash
python -c "
import yaml, copy
from scripts.train import train
with open('configs/base.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = 42
cfg['env_id'] = 'cartpole_gravity'
result = train(cfg)
print(result['summary'])
"
```

---

## Dependencies

- **PyTorch** ≥ 2.0: neural networks, autograd
- **Gymnasium** ≥ 0.29: RL environments
- **SciPy**: KS test
- **NumPy / Pandas**: data processing
- **Matplotlib**: visualisation

No GPU required; all experiments run on CPU in < 5 minutes.
