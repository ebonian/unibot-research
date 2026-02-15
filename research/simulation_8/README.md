# Simulation 8: Exact Per-Swap Fee/LVR Computation

## Overview

This simulation implements the **exact per-swap fee and LVR calculation** from the paper,
iterating over every individual swap within each hourly decision step.

## How It Works (Paper Method)

The agent makes decisions at **hourly intervals**, but fees and LVR are computed by
**iterating over every individual swap** that occurred within that hour:

1. During data preparation, we store the array of swap prices per hour from `swaps_*.csv`
2. At each environment step, `_compute_fee` loops over consecutive swap price pairs `(p_i, p_{i+1})`,
   clamps each to the LP's range `[p_lower, p_upper]`, and sums the fee formula (Eq 5-6)
3. `_compute_lvr` does the same — summing LVR contributions per swap, not just open-to-close

## Fee Calculation (Exact Derivation)

For each consecutive swap price pair `(p_i, p_{i+1})` within an hour, fees are computed
following the paper's Equations 5-6:

**If price increases** (√p_i ≤ √p_{i+1}):

    fee_i = (δ / (1 - δ)) × L × (√p_{i+1,c} - √p_{i,c})

**If price decreases** (√p_i > √p_{i+1}):

    fee_i = (δ / (1 - δ)) × L × (1/√p_{i+1,c} - 1/√p_{i,c}) × p_{i+1,c}

Where:
- `δ` = pool fee rate (e.g. 0.0005 for 0.05%)
- `L` = your position's liquidity
- `p_{i,c} = clamp(p_i, p_lower, p_upper)` — price clamped to your LP range
- If both `p_i` and `p_{i+1}` are outside your range on the same side, `fee_i = 0`

**Total hourly fee** = Σ fee_i over all swaps in that hour.

## LVR Calculation (Exact Per-Swap)

LVR (Loss-Versus-Rebalancing) is also summed per swap:

    LVR_i = V(p_{i+1}) - V(p_i) - x(p_i) × (p_{i+1} - p_i)

Where:
- `V(p)` = position value at price p (standard Uniswap V3 formula)
- `x(p_i)` = amount of token X held at price p_i
- LVR is always ≤ 0 (it's a cost)

**Total hourly LVR** = Σ LVR_i over all swaps in that hour.

## Reward Function

Following the paper's Equation 17:

    R_{t+1} = fee_t - LVR_t - gas_cost

Where `gas_cost` is incurred only when the agent rebalances (changes position).

## Why This Formula Gives the Correct Fee

The formula `fee = (δ/(1-δ)) × your_L × |Δ√p|` is not an approximation —
it gives the **mathematically exact** fee your position earns. Here's why:

1. Total fee from a swap: `total_fee = (δ/(1-δ)) × L_total × |Δ√p|`
2. Your share: `your_fee = (your_L / L_total) × total_fee`
3. Substituting: `your_fee = (your_L / L_total) × (δ/(1-δ)) × L_total × |Δ√p|`
4. **L_total cancels out**: `your_fee = (δ/(1-δ)) × your_L × |Δ√p|`

The historical `|Δ√p|` from swap data already encodes the actual pool depth
(more total liquidity = smaller price impact per trade = smaller `|Δ√p|`).
So the formula implicitly accounts for other LPs' liquidity through the
historical price movements.

## Comparison with Simulation 6 (Ground Truth)

Both Sim_6 and Sim_8 iterate through **every swap** and give the same fee result,
but through different computation paths:

### Simulation 8 (This Implementation — Paper Method)

```
for each swap (p_i, p_{i+1}):
    clamp prices to LP range
    fee += (δ/(1-δ)) × L × |√p_{i+1,c} - √p_{i,c}|
```

- Uses the direct formula from the paper
- ~20 lines of math
- Fast enough for RL training loops

### Simulation 6 (Full Protocol Simulation)

```
for each swap:
    while sqrt_price != target:
        find next initialized tick
        compute step to tick boundary or target
        compute gross_in = net_in / (1 - fee_rate)
        fee_amount = gross_in - net_in
        lp_fee = fee_amount - protocol_fee
        fee_growth_global += (lp_fee × 2^128) / liquidity_active
        if crossing tick: update liquidity_active from tick map
    your_fee = (your_L × Δfee_growth_inside) / 2^128
```

- Replicates Uniswap V3 Solidity contract exactly
- Integer arithmetic, `fee_growth_global` accumulators, tick crossings
- Tracks all LPs' positions and `liquidity_active` changes at tick boundaries
- Protocol fee splitting (LP share vs protocol share)
- ~800 lines of Solidity-like logic
- Too slow for RL training

### Why They Give the Same Result

Both compute `your_fee = (δ/(1-δ)) × your_L × |Δ√p_clamped|` per swap.
Sim_6 arrives at this through the full protocol path:

1. `fee_amount = gross_in - net_in` → this equals `(δ/(1-δ)) × L_total × |Δ√p|`
2. `your_share = your_L / L_total` (via fee_growth accounting)
3. Result: `your_fee = (δ/(1-δ)) × your_L × |Δ√p|` — same formula

The extra machinery in Sim_6 (tick crossings, fee_growth accumulators, protocol fees)
does not change the final answer for a small LP position. It only matters if:
- Your position is large enough to meaningfully change `liquidity_active`
- Protocol fee rates change mid-simulation
- You need exact integer-level precision matching the on-chain contract

For RL training with a $1000 position on a pool with $100M+ TVL, none of these apply.

### Summary

| Aspect | Sim_7 (TV Approx) | Sim_8 (Paper Method) | Sim_6 (Ground Truth) |
|---|---|---|---|
| Iterates swaps | No (1 scalar per hour) | Yes | Yes |
| Fee formula | `fee_mult × L × TV × frac` | `Σ fee_mult × L × \|Δ√p_c\|` | Full Solidity logic |
| In-range handling | Estimated from open-close | Exact per swap | Exact per swap |
| LVR | Endpoint only | Per swap | Per swap |
| Fee accuracy | ~92% | **Exact** | **Exact** (same result) |
| Speed | Fastest | Medium | Slowest |
| Suitable for RL | Yes (approximate) | **Yes (exact + fast)** | No (too slow) |

## Architecture

Same as Simulation 7:
- **DQN**: Dueling Double DQN (Zhang et al. 2023)
- **PPO**: PPO (Xu & Brini 2025)
- **LSTM DQN**: LSTM-based Dueling DDQN (sequence-aware)

## Data

Uses symlinked `training_data/` from `simulation_6/` (swap-level Uniswap V3 data):
- `swaps_*.csv` — individual swap transactions (~892MB)
- `ETHUSDT_hourly_data_*.csv` — hourly OHLCV candles
- `pool_config_*.csv` — pool parameters (fee tier, tick spacing)

## Usage

```bash
# Run baselines only
python compare_algorithms.py --data-dir training_data --baselines-only --eval-episodes 3

# Train LSTM DQN
python compare_algorithms.py --data-dir training_data --lstm-only --lstm-episodes 100 --lstm-seq-len 24

# Train all algorithms (PPO + DQN)
python compare_algorithms.py --data-dir training_data --ppo-timesteps 500000 --dqn-episodes 500

# Use GPU
python compare_algorithms.py --data-dir training_data --lstm-only --device cuda
```

## Baseline Results (Exact Per-Swap)

With exact fees, **concentrated LP is profitable**:

| Strategy | Mean Reward ($) |
|---|---|
| HOLD (no LP) | 0.00 |
| Fixed width=5 (narrow) | **+652.92** |
| Fixed width=25 (medium) | **+87.13** |
| Fixed width=50 (wide) | -38.87 |

This confirms that the fee underestimation in earlier simulations was the root cause
of agents learning to avoid LP entirely.
