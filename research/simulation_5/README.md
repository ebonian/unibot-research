# Uniswap V3 RL Agent (Simulation 5)

A reinforcement learning agent that learns optimal liquidity provision strategies for Uniswap V3.

## Overview

This project trains a PPO agent to manage concentrated liquidity positions on Uniswap V3. The agent learns when to:
- **ADJUST**: Create/move LP position with optimal width
- **HOLD**: Keep current position
- **BURN**: Exit position

## Files

| File | Description |
|------|-------------|
| `uniswap_v3_ppo_continuous.py` | Main training script + Gymnasium environment |
| `fee_simulator.py` | Uniswap V3 fee/IL calculation engine |
| `evaluate_agent.py` | Visualization & evaluation of trained model |
| `reward_test.py` | Quick sanity check for trained model |
| `test_fee_simulator_mock.py` | Unit tests for fee simulator |
| `test_capital_carryover.py` | Tests for capital tracking logic |
| `vec_normalize.pkl` | Trained observation normalization stats |
| `ppo_uniswap_v3_continuous.zip` | Trained PPO model (generated) |

## Environment Design

### Observation Space (7 dimensions)
```
[0] log(price)          - Current ETH price (log scale)
[1] width_pct           - Current LP range width (0-1)
[2] has_lp              - Position exists flag (0 or 1)
[3] volatility_24h      - Recent price volatility (0-1)
[4] price_change_pct    - Price change since last step (-1 to 1)
[5] in_range            - Is price within LP bounds (0 or 1)
[6] capital_ratio       - current_capital / initial_capital (0-2)
```

### Action Space (2 continuous values)
```
[0] mode (-1 to 1):     <-0.33 BURN | [-0.33, 0.33] HOLD | >0.33 ADJUST
[1] width_param (-1 to 1): Maps to width in [0.1%, 1%]
```

### Reward Function (Realistic IL)
```
On HOLD:   reward = fees_usd              (IL is unrealized)
On ADJUST: reward = fees + crystallized_IL - gas  (IL realized on rebalance)
On BURN:   reward = fees + crystallized_IL - gas  (IL realized on exit)
```

Key design: **Impermanent Loss is only crystallized when the position is closed or rebalanced**, matching real LP behavior. During HOLD, only accumulated fees count as reward.

## Training Parameters

```python
window_hours = 1          # 1-hour decision windows
gas_per_action_usd = 0.1  # Arbitrum L2 gas cost
total_usd = 1000          # Initial capital
min_width_pct = 0.001     # 0.1% minimum range
max_width_pct = 0.01      # 1% maximum range
total_timesteps = 100000  # PPO training steps
train_ratio = 0.8         # 80% train / 20% eval split
```

## Train/Eval Split

The environment supports proper data splitting to prevent overfitting:
```python
# Training (default): uses first 80% of time windows
env = UniswapV3ContinuousEnv(data_dir, mode="train")

# Evaluation: uses last 20% of time windows  
env = UniswapV3ContinuousEnv(data_dir, mode="eval")

# Full data (for final testing)
env = UniswapV3ContinuousEnv(data_dir, mode="all")
```

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Train new model
python uniswap_v3_ppo_continuous.py

# 3. Evaluate trained model with visualization
python evaluate_agent.py

# 4. Quick action test
python reward_test.py
```

## Dependencies

```bash
pip install numpy pandas gymnasium stable-baselines3 shimmy matplotlib
```

## Data Requirements

Requires CSV files in `dune_pipeline/`:
- `pool_config_eth_usdt_0p3.csv`
- `token_metadata_eth_usdt_0p3.csv`
- `slot0_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`
- `swaps_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`
- `mints_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`
- `burns_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`

Also requires hourly ETH/USD prices (local file):
- `ETHUSDT_hourly_data_20241101_20251101.csv`

## Data Fields Used

### Environment (uniswap_v3_ppo_continuous.py)

| File | Field | Purpose |
|------|-------|---------|
| `swaps` | `evt_block_time` | Build 1-hour time windows |
| `swaps` | `sqrtPriceX96` | Calculate 24h volatility (observation) |

### Fee Simulator (fee_simulator.py)

| File | Field | Purpose |
|------|-------|---------|
| `swaps` | `sqrtPriceX96`, `tick` | Track price movement |
| `swaps` | `liquidity` | Active liquidity for fee share |
| `swaps` | `amount0`, `amount1` | Fee accrual calculation |
| `slot0` | `output_sqrtPriceX96` | Initial pool state |
| `slot0` | `output_tick` | Initial tick position |
| `mints` | `tickLower`, `tickUpper` | LP range bounds |
| `mints` | `liquidity_added` | Track liquidity changes |
| `burns` | `tickLower`, `tickUpper` | LP range bounds |
| `burns` | `liquidity_removed` | Track liquidity changes |
| `pool_config` | `fee` | Fee tier (3000 = 0.3%) |
| `pool_config` | `tickSpacing` | Tick spacing for range |
| `token_metadata` | `decimals` | Price conversion (ETH=18, USDT=6) |

### External Price Data

| File | Field | Purpose |
|------|-------|---------|
| `ETHUSDT_hourly_data*.csv` | `close` | USD valuation of positions |

## Results

With 1-hour windows and $0.10 gas cost:
- Agent learns to maintain narrow positions (~0.1% width)
- ~$610/month potential profit (simulated)
- Trained on September 2025 data (first 80%)

## Trust Levels

| Component | Trust | Notes |
|-----------|-------|-------|
| Dune Data | High | On-chain event data from Dune Analytics |
| Fee Simulator | Medium | Spot-checked, not formally verified |
| Gas Estimates | Low | Based on research; Arbitrum costs vary $0.01-$0.50 |
| ETH/USD Prices | Medium | External feed; minor discrepancies possible |

## Known Limitations

1. Trained on single month of data (September 2025)
2. Evaluation on last 20% of same month (same market regime)
3. Gas cost is estimated ($0.10), actual L2 costs vary
4. Does not model slippage or position size impact on pool
