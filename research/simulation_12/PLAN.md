# Simulation 10 — Improvement Plan

> Cloned from simulation_9. Same data, same widths [1,3,5,10,20,40], same unified 36-dim state space.
> Goal: fix known issues and add targeted ML improvements to beat the baseline consistently.

## Baseline Reference (from sim_9 run_005)

| Model | Reward | Notes |
|---|---|---|
| Fixed Width=1 (baseline) | 873 | Always recenter, always width=1 |
| LSTM DQN | 883 | Beat baseline via 6.3% HOLDs |
| DQN | 789 | Hurt by using wider widths |

---

## Improvements (ordered by priority)

### 1. 🔴 Fix Reward Explosion (CRITICAL)

**Problem:** Training rewards ~300,000 but eval rewards ~360. The agent is exploiting the simulation — learning a broken strategy that only partially transfers to eval.

**Root cause:** Liquidity calculation can produce unrealistically large L values → massive fee calculations during training.

**Fix:**
- [ ] Cap per-step reward to a realistic max (e.g. ±$10/hour for $1000 capital)
- [ ] Add sanity check on liquidity: `position_value(L) ≈ invested_capital`
- [ ] Log reward statistics during training to verify the fix

**Expected impact:** 🔴 Critical — without this, the agent learns garbage.

---

### 2. 🟡 Add LP-Specific Features (+2 features → 38-dim state)

**Problem:** Current 31 tech features are generic market signals. The agent lacks features directly useful for LP decisions.

**Fix:**
- [ ] Add **distance to range boundary** — how many ticks until price exits the LP range (normalized). Directly useful for HOLD vs rebalance decision.
- [ ] Add **hours since last rebalance** — helps agent learn gas-saving patterns (normalized by e.g. 24h).

**Why only 2:** Other candidates (fee income rate, position PnL) are mostly redundant with existing features. Keep it simple — more features ≠ better learning.

**Expected impact:** 🟡 Medium — gives the agent the most relevant signals for HOLD decisions.

---

### 3. � N-Step Returns (DQN/LSTM)

**Problem:** 1-step TD learning has poor credit assignment. The "hold now → earn fees later" pattern spans multiple steps, but 1-step TD only sees immediate reward.

**Fix:**
- [ ] Implement 3-step returns in DQN replay buffer
- [ ] Implement 3-step returns in LSTM replay buffer
- [ ] Adjust target calculation: `R = r_t + γ*r_{t+1} + γ²*r_{t+2} + γ³*Q(s_{t+3})`

**Expected impact:** 🟡 Medium — textbook DQN improvement, usually 10-20% better learning. ~20 lines of code.

---

### 4. � Hyperparameter Tuning

**Problem:** Current hyperparameters were not optimized for the simplified 7-action space.

**Fix:**
- [ ] **DQN gamma: 0.9 → 0.95 or 0.99** — current gamma is very short-sighted, agent barely considers future rewards
- [ ] **Learning rate sweep** — test 5e-5, 1e-4, 5e-4
- [ ] **Network size** — try [128, 64] instead of [64, 64]

**Expected impact:** 🟡 Medium — often more impactful than architecture changes. Gamma especially matters for HOLD decisions (need to value future fees).

---

### 5. � Prioritized Experience Replay (Optional)

**Problem:** Uniform sampling wastes time replaying unimportant transitions.

**Fix:**
- [ ] Add PER to DQN replay buffer
- [ ] Prioritize transitions with high TD-error

**Expected impact:** 🟢 Small — nice-to-have, but with only ~2200 steps per episode and 7 actions, uniform replay is probably fine.

---

## Dropped Items (from initial plan)

| Item | Reason dropped |
|---|---|
| Reward shaping (ΔV) | Risk of reward hacking; current reward is paper-grounded |
| Noisy Nets | Exploration isn't the bottleneck; epsilon is already 0.05 |
| Attention for LSTM | Over-engineering; 24 timesteps is tiny, LSTM handles it fine |

---

## Implementation Order

```
Fix reward explosion → Add 2 LP features → N-step returns → Tune hyperparameters → Run
```

Each improvement builds on the previous. We'll implement and test them incrementally.
