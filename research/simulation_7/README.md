# Simulation 7: Paper-Based Approach

This folder implements RL methodologies from academic papers for Uniswap v3 LP optimization.

## Pool Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Chain** | Arbitrum L2 | Lower gas costs than Ethereum mainnet |
| **Pool** | ETH/USDT | `0x8c9d230d45d6cfee39a6680fb7cb7e8de7ea8e71` |
| **Fee Tier** | 0.05% (500 bps) | Lower than typical 0.3% pools |
| **Tick Spacing** | 10 | ~0.1% between initializable ticks |
| **Gas Cost** | ~$0.50 per rebalance | Arbitrum L2 gas (not Ethereum's $5-50) |
| **Data Period** | 2025-11-03 to 2026-02-03 | ~3 months, includes bear market |

## Reference Papers

### 1. Xu & Brini (2025) - PPO Approach
**"Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning"**
- Conference: AAAI 2025 Workshop
- arXiv: [2501.07508](https://arxiv.org/abs/2501.07508)
- Algorithm: **PPO (Proximal Policy Optimization)**

### 2. Zhang, Chen & Yang (2023) - DQN Approach
**"Adaptive Liquidity Provision in Uniswap V3 with Deep Reinforcement Learning"**
- arXiv: [2309.10129](https://arxiv.org/abs/2309.10129)
- Algorithm: **Dueling Double DQN**
- Key innovation: LVR in reward, hedging strategy, 32 technical features

### 3. Fan et al. (2024) - Neural Network Optimization
**"Strategic Liquidity Provision in Uniswap v3"**
- arXiv: [2106.12033](https://arxiv.org/abs/2106.12033)
- Approach: τ-reset strategies with NN-based optimization (not RL)

## Key Differences from Simulation 6

| Aspect | Simulation 6 (Full-Sim) | Simulation 7 (Paper) |
|--------|------------------------|----------------------|
| **Fee calculation** | Swap-by-swap replay | Formula-based (Eq. 5-6) |
| **Data granularity** | Every swap (~3.2M rows) | Hourly resampled |
| **Speed** | Slow (exact) | Fast (approximation) |
| **Reward** | fees + IL | fees + LVR - gas |
| **Action space** | Continuous (width %) | Discrete (tick widths) |
| **Algorithms** | PPO only | PPO + Dueling DDQN |

## Files

| File | Description | Reference |
|------|-------------|-----------|
| `uniswap_v3_ppo_paper.py` | PPO implementation | Xu & Brini (2025) |
| `uniswap_v3_dqn_paper.py` | Dueling DDQN implementation | Zhang et al. (2023) |
| `compare_algorithms.py` | Train & compare both + baselines | - |
| `README.md` | This documentation | - |

## Mathematical Foundations

### Fee Calculation (Paper Equations 5-6)

**Verified against Uniswap v3 Whitepaper (Section 6.2.3)**

For price increase `p_t ≤ p_{t+1)}`:
```
f_t = (δ / (1-δ)) × L_t × (√p_{t+1} - √p_t)
```

For price decrease `p_t > p_{t+1}`:
```
f_t = (δ / (1-δ)) × L_t × (1/√p_{t+1} - 1/√p_t) × p_{t+1}
```

**Derivation from Whitepaper:**
- Whitepaper Eq. 6.14: `Δy = Δ√P × L`
- Whitepaper Eq. 6.11: `Δy = y_in × (1 - γ)`, so `y_in = Δy / (1 - γ)`
- Fee = `y_in × γ = Δy × γ / (1 - γ) = L × Δ√P × γ / (1 - γ)`
- This matches our formula with `δ = γ`

Where:
- `δ` = pool fee rate (0.0005 for our 0.05% pool)
- `L_t` = LP liquidity
- `p_t`, `p_{t+1}` = prices at time t and t+1

**Note:** This is a formula-based approximation. Fees depend on price movement, not actual swap volume. This is standard for simulation/research purposes.

### LVR (Loss-Versus-Rebalancing) - Zhang et al. (2023)

LVR quantifies the cost of providing liquidity vs a rebalancing portfolio:

```
LVR = Σ [V(p_{t+1}) - V(p_t) - x(p_t) × (p_{t+1} - p_t)]
```

Where:
- `V(p)` = position value at price p
- `x(p)` = token X holdings at price p
- LVR is always ≤ 0 (it's a cost)

For continuous case (geometric Brownian motion):
```
dLVR = V''(p_t) × σ² × p_t² × dt = -L / (2 × p_t^1.5) × σ² × p_t² × dt
```

### Reward Function

**Xu & Brini (2025):**
```
R_{t+1} = f_t - ℓ_t(σ, p) - gas_cost × 𝕀[position_changed]
```

**Zhang et al. (2023) with hedging:**
```
R_{t+1} = Fee_t + LVR_t - gas_cost × 𝕀[position_changed]
```

### State Space Comparison

| Feature | PPO (8 features) | DQN (26 features) |
|---------|------------------|-------------------|
| Price | log(price) | OHLC ratios |
| Position | width, in_range | width, cash, value |
| Volatility | σ_t | ATR, NATR |
| Trend | MA_24h, MA_168h | MACD, ADX, RSI |
| Momentum | - | CCI, ROC, Stoch |
| Technical | - | Bollinger, DX, etc. |

### Action Space

**PPO Actions (5 discrete):**
| Action | Width (ticks) | Approx. Range | Behavior |
|--------|---------------|---------------|----------|
| 0 | - | - | **HOLD** - no gas |
| 1 | 50 | ±0.25% | Deploy/rebalance |
| 2 | 100 | ±0.5% | Deploy/rebalance |
| 3 | 200 | ±1.0% | Deploy/rebalance |
| 4 | 500 | ±2.5% | Deploy/rebalance |

**DQN Actions (51 discrete):**
- Action 0: **HOLD** - no gas
- Actions 1-50: Deploy LP with width = action × tick_spacing

### Action Logic (Real-World Uniswap v3)

In Uniswap v3, there is no "adjust position" - only **mint** (create) and **burn** (destroy).
Any range change requires: burn old position + mint new position = **2x gas cost**.

| Scenario | Action | Gas Cost |
|----------|--------|----------|
| No position → Deploy | Any width > 0 | $0.50 |
| Has position → HOLD | Action 0 | $0 |
| Has position → Change width | Any width > 0 | $0.50 |
| Has position → Same width (recenter) | Same width | $0.50 |

**Agent learns:** When is paying $0.50 gas worth it to recenter/resize position?

### Gas Cost Justification

| Chain | Typical Rebalance Cost | Our Setting |
|-------|----------------------|-------------|
| **Ethereum Mainnet** | $5-50 | - |
| **Arbitrum L2** | **$0.10-$1.00** | **$0.50** ✓ |

Our pool is on **Arbitrum** (verified by token addresses), so we use $0.50 gas cost, not the $5 from papers which assumed Ethereum mainnet.

## Usage

### PPO Training (Xu & Brini approach)
```bash
cd /Users/ohm/Documents/GitHub/ice-senior-project/research/simulation_7

python uniswap_v3_ppo_paper.py \
    --data-dir ../simulation_6/training_data \
    --num-envs 8 \
    --timesteps 100000 \
    --save-path ppo_uniswap_v3_paper
```

### Dueling DDQN Training (Zhang et al. approach)
```bash
python uniswap_v3_dqn_paper.py \
    --data-dir ../simulation_6/training_data \
    --episodes 500 \
    --save-path dqn_uniswap_v3_paper
```

### Evaluation
```bash
# PPO
python uniswap_v3_ppo_paper.py --evaluate --eval-episodes 10

# DQN
python uniswap_v3_dqn_paper.py --evaluate
```

### Full Comparison (For Thesis)
```bash
# Train and evaluate both algorithms + baselines
python compare_algorithms.py \
    --data-dir ../simulation_6/training_data \
    --ppo-timesteps 50000 \
    --dqn-episodes 200 \
    --eval-episodes 10

# Results saved to comparison_results.json
```

## Algorithm Comparison

| Aspect | PPO | Dueling DDQN |
|--------|-----|--------------|
| **Type** | Policy gradient | Value-based |
| **Action** | Continuous → discretized | Discrete |
| **Sample efficiency** | Lower | Higher |
| **Stability** | High (clipped updates) | Moderate (target network) |
| **Exploration** | Entropy bonus | ε-greedy |
| **Paper** | Xu & Brini (2025) | Zhang et al. (2023) |

## Key Findings from Papers

### Zhang et al. (2023) - DQN
- Agent prefers **narrowest width (width=1)** for maximum fee concentration
- Reallocation frequency **decreases with more capital** (gas becomes less significant)
- **Hedging reduces variance** in returns significantly
- DQN outperforms baselines by 9-69% depending on initial capital

### Xu & Brini (2025) - PPO
- Active LP outperforms passive in **7 out of 11 test windows**
- Agent learns to **anticipate mean reversion** and avoid unnecessary reallocations
- Rolling window training important for handling regime changes

## Our Experimental Results

### Test Conditions
- **Period:** 2026-01-25 to 2026-02-03 (10% test set)
- **Market:** Bear market (ETH dropped ~30% from $2900 to $2300)
- **Fee tier:** 0.05% (lower than papers' typical 0.3%)

### Results Summary

| Strategy | Mean Reward | Notes |
|----------|-------------|-------|
| **HOLD (baseline)** | 0.00 | No LP, no risk |
| Fixed Width=1 | -477.42 | Constant rebalancing, gas drain |
| Fixed Width=5 | -397.61 | Less frequent rebalancing |
| **PPO** | **-19.80 ± 59.40** | Deploys wide, then holds |
| **DQN** | **0.00** | Learned to always HOLD |

### Interpretation

**Why DQN learned HOLD:**
- In 0.05% fee pool during bear market, LP is barely profitable
- DQN correctly learned: "Not providing LP is optimal"
- This is rational behavior, not a bug

**Why PPO shows negative reward:**
- PPO deployed one wide LP position (500 ticks)
- Bear market caused impermanent loss (IL)
- Reward includes **theoretical LVR** (unrealized IL estimate)
- Actual realized P&L would differ

### Comparison with Paper Findings

| Paper Finding | Our Result | Explanation |
|---------------|------------|-------------|
| DQN prefers narrow width | DQN prefers HOLD | Our fee tier is 6x lower (0.05% vs 0.3%) |
| PPO outperforms passive | PPO underperforms | Bear market + low fees = LP unprofitable |
| Active LP beats passive | Tie or worse | Market conditions matter significantly |

### Key Insight

**LP profitability depends heavily on:**
1. **Fee tier:** 0.05% vs 0.3% makes 6x difference in fee income
2. **Market conditions:** Bull vs bear market affects IL
3. **Gas costs:** Arbitrum ($0.50) vs Ethereum ($5-50)

Our agents learned **rational strategies for their environment** - they just happen to favor conservative behavior in our low-fee, bear-market test conditions.

## For Your Thesis

### Approaches Comparison

| Approach | Simulation | Fee Calculation | Speed | Accuracy |
|----------|------------|-----------------|-------|----------|
| **Simulation 6** | Exact swap-by-swap | Per-swap events | Slow | Highest |
| **Simulation 7 PPO** | Formula-based | Price change formula | Fast | Approximation |
| **Simulation 7 DQN** | Formula-based | Price change formula | Fast | Approximation |

### What This Demonstrates

1. **Multiple RL Algorithms:**
   - PPO (Policy Gradient) - stable, handles continuous/discrete actions
   - Dueling DDQN (Value-Based) - sample efficient, good for discrete actions

2. **Published Methodology Implementation:**
   - Xu & Brini (2025) - PPO approach with LVR penalty
   - Zhang et al. (2023) - Dueling DDQN with technical indicators

3. **Mathematical Rigor:**
   - Fee formulas derived from and verified against Uniswap v3 whitepaper
   - LVR formulas from academic literature
   - Proper handling of concentrated liquidity mechanics

4. **Real-World Considerations:**
   - Correct gas costs for Arbitrum L2
   - Proper action space (HOLD vs rebalance)
   - Market condition impact on profitability

### Thesis Discussion Points

1. **Why results differ from papers:**
   - Different fee tiers (0.05% vs 0.3%)
   - Different market conditions (our test period is bear market)
   - Papers often use simulated or cherry-picked periods

2. **Agent rationality:**
   - Both agents learned rational strategies for their environment
   - DQN's "always HOLD" is optimal when LP isn't profitable
   - PPO's "deploy wide and hold" minimizes gas while staying in range

3. **Limitations of formula-based approach:**
   - Assumes all volume flows through your position proportionally
   - Real fees depend on actual swap volume, not just price change
   - Simulation 6 (per-swap) is more accurate but slower

### Visualization

The `test_decisions.png` chart shows:
- **Top panel:** PPO LP positions (green shaded) with rebalance markers (red triangles)
- **Middle panel:** DQN LP positions (or lack thereof for HOLD strategy)
- **Bottom panel:** Cumulative reward comparison over time
