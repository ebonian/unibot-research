# Simulation 10: ML Improvements

## Overview

Builds on Simulation 9 with ML-focused improvements to beat the baseline more consistently.

**Base:** Simulation 9 (unified state space, corrected bugs, no offsets)
**Goal:** Fix reward explosion, add LP-specific features, improve RL training

## Key Changes from Simulation 9

See [PLAN.md](PLAN.md) for the full improvement roadmap.

## Baseline Reference (sim_9 run_005)

| Model | Reward | $1000 → |
|---|---|---|
| Fixed Width=1 (baseline) | 873 | ~$1,149 |
| LSTM DQN | 883 | $1,150 |
| DQN | 789 | $1,134 |
| PPO | — | error (fixed) |

## How to Run

```bash
source .venv/bin/activate
python compare_algorithms.py --data-dir training_data --run-dir run_001 \
  --ppo-timesteps 200000 --dqn-episodes 800 --lstm-episodes 1000
```
