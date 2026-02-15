# Next Steps for Simulation 8

## Current Status

All 3 models trained with minimal episodes and are **profitable** on test data.
Fee calculation is exact (per-swap, following paper). Environment is correct.

### Current Results (50 episodes / 50k timesteps)

| Model | Test Reward | vs Best Baseline | Notes |
|---|---|---|---|
| PPO | $567.5 | -13% | Wide ranges, could improve with more training |
| DQN | $644.0 | -1.4% | Narrow ranges, nearly matching baseline |
| LSTM DQN | $653.5 | +0.1% | Barely beating baseline, learned 4 HOLDs |
| Fixed width=5 | $652.9 | baseline | Always narrow, always rebalance |

**Problem:** Agents basically learned "always use narrow range, always rebalance" —
the same as the simplest baseline. They haven't learned *when* to HOLD or *when*
to use directional offsets.

---

## Step 1: Train Longer (Highest Priority)

**Why:** 50 episodes is very little. Papers use 500-1000 episodes. The agent needs
more exploration to discover that sometimes HOLD saves gas, and offsets can
capture directional moves.

**What to run:**
```bash
# DQN: 500 episodes (~3-4 hours)
python compare_algorithms.py --data-dir training_data --dqn-episodes 500 --eval-episodes 5

# LSTM: 500 episodes (~8-10 hours)
python compare_algorithms.py --data-dir training_data --lstm-only --lstm-episodes 500 --eval-episodes 5

# PPO: 500k timesteps (~30 min)
python compare_algorithms.py --data-dir training_data --ppo-timesteps 500000 --eval-episodes 5
```

**Expected improvement:** Agents should learn to:
- HOLD when position is still in range (saves $0.50 gas)
- Use offsets during trending markets
- Choose wider ranges during high volatility

---

## Step 2: Tune Hyperparameters

### 2a. Epsilon Decay (DQN / LSTM)

Current: `epsilon_decay = 0.99` per episode (reaches 0.05 at ~300 episodes)

**Options:**
- Slower decay (0.995): More exploration, better for finding offset strategies
- Faster decay (0.98): Exploit sooner, might converge faster but miss strategies

### 2b. Discount Factor (gamma)

Current: `gamma = 0.9`

**Options:**
- Higher (0.95-0.99): Agent values future rewards more → learns long-term strategies
- Lower (0.8): Agent is more myopic → focuses on immediate fee income

### 2c. Learning Rate

Current: `lr = 1e-4` for all models

**Options:**
- Lower (3e-5): More stable but slower learning
- Learning rate scheduling: Start high, decay over training

### 2d. Network Architecture

Current: `[64, 64]` hidden layers, LSTM hidden=64

**Options:**
- Larger: `[128, 128]` or LSTM hidden=128 — more capacity
- Deeper: `[64, 64, 64]` — might capture more complex patterns
- Tradeoff: Larger networks need more training data

---

## Step 3: Tune Reward Shaping

### 3a. In-Range Bonus

Current: `+0.05` per step when in range

**Issue:** This is tiny compared to actual fees ($2-3/hour). May not affect learning.

**Options:**
- Remove it (reward should come naturally from fees)
- Scale it relative to position value
- Replace with: bonus proportional to fee earned

### 3b. Opportunity Cost Penalty

Current: `0.006%` per hour when out of range

**Issue:** Also tiny. Agent barely notices it.

**Options:**
- Increase to make out-of-range positions more costly
- Remove (LVR=0 when out of range is already a signal)

### 3c. Gas Cost

Current: `$0.50` per rebalance

**Issue:** With fees at $2-3/hour, gas is always worth paying.
This is realistic (Arbitrum L2), but means HOLD is rarely optimal.

**Options:**
- Keep as-is (realistic for Arbitrum)
- Test higher gas ($2-5) to see if agents learn better HOLD strategies
- Make gas dynamic (higher during congestion)

---

## Step 4: Tune State Space

### 4a. PPO State Space (Currently 8 features)

The PPO has minimal features. Could benefit from:
- Add RSI, MACD, Bollinger Band width (volatility signals)
- Add recent returns (1h, 24h) for trend detection
- Add position age (hours since last rebalance)

### 4b. DQN/LSTM State Space (Currently 37 features)

Already rich. Could try:
- Feature selection: remove low-importance features
- Add swap volume per hour (trading activity signal)
- Add fee earned in last step (feedback signal)

### 4c. State Normalization

Current: Features clipped to [-10, 10]

**Options:**
- Z-score normalization per feature
- Running mean/std normalization
- This can significantly affect learning stability

---

## Step 5: Tune Action Space

### 5a. Width Options

Current: `[5, 10, 25, 50]` tick spacings

**Observation:** All agents converge to width=5. Consider:
- Finer granularity: `[3, 5, 8, 10, 15, 25]`
- Remove wide options that are never profitable: drop 50
- Add very narrow: width=2 or 3

### 5b. Offset Options

Current: `[-2, -1, 0, 1, 2]` (in half-width units)

**Observation:** Agents rarely use offsets. Consider:
- Finer granularity: `[-1, -0.5, 0, 0.5, 1]`
- Remove extreme offsets: `[-1, 0, 1]`
- This reduces action space → easier to learn

### 5c. HOLD vs Rebalance

Current: HOLD only if action=0

**Consider:** Add "smart HOLD" — HOLD unless price moves outside X% of range
This would reduce the action space and let the agent focus on width/offset decisions.

---

## Step 6: Advanced Improvements

### 6a. Prioritized Experience Replay (DQN/LSTM)
- Replay important transitions more often
- Helps learn from rare but informative events (e.g., price crashes)

### 6b. Multi-Step Returns
- Use n-step returns instead of 1-step
- Helps with credit assignment over longer horizons

### 6c. Curriculum Learning
- Train on easy periods first (sideways market)
- Gradually introduce harder periods (crashes, pumps)

### 6d. Ensemble Methods
- Train multiple agents with different seeds
- Use majority vote or average for final policy

---

## Recommended Order

```
Priority 1: Train longer (Step 1)
    └── If plateaus → Priority 2: Tune hyperparameters (Step 2)
        └── If still plateaus → Priority 3: Tune reward (Step 3)
            └── Priority 4: Tune state/action space (Steps 4-5)
                └── Priority 5: Advanced methods (Step 6)
```

Each step should be evaluated by comparing against the fixed_width=5 baseline.
The goal is to consistently **beat the best baseline** on test data.
