# Simulation 9: Tuned & Improved (Based on Simulation 8)

## Overview

This simulation builds on Simulation 8 (paper-exact per-swap method) with
hyperparameter tuning and improvements to beat the baseline strategies.

**Base:** Simulation 8 (exact per-swap fee/LVR, paper method)
**Goal:** Beat the fixed_width=5 baseline consistently through tuning

## Changes from Simulation 8

(To be filled as we make changes)

## Baseline to Beat

From Simulation 8 (paper method, no tuning):

| Model | Test Reward | Notes |
|---|---|---|
| Fixed width=5 | $652.9 | Best baseline — always narrow, always rebalance |
| LSTM DQN | $653.5 | Barely beating baseline (50 episodes) |
| DQN | $644.0 | Below baseline (50 episodes) |
| PPO | $567.5 | Below baseline (50 episodes) |
