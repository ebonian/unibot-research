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
| `uniswap_v3_ppo_continuous.py` | Main training script + environment |
| `fee_simulator.py` | Uniswap V3 fee/IL calculation engine |
| `reward_test.py` | Test trained model actions |
| `vec_normalize.pkl` | Trained normalization stats |
| `ppo_uniswap_v3_continuous.zip` | Trained model (generated) |

## Environment Design

### Observation Space (6 dimensions)
```
[0] log(price)          - Current ETH price
[1] width_pct           - Current LP range width (0-1)
[2] has_lp              - Position exists flag
[3] volatility_24h      - Recent price volatility
[4] price_change_pct    - Price change since last step
[5] in_range            - Is price in LP bounds
```

### Action Space (2 continuous values)
```
[0] mode (-1 to 1):     <-0.33 BURN | [-0.33, 0.33] HOLD | >0.33 ADJUST
[1] width_param (-1 to 1): Maps to width in [0.1%, 1%]
```

### Reward Function
```
reward = fees_usd + il_usd - gas_per_action
```
- `fees_usd`: Trading fees earned
- `il_usd`: Impermanent loss (negative = loss)
- `gas_per_action`: $0.10 per rebalance (L2 cost)

## Training Parameters

```python
window_hours = 1          # 1-hour decision windows
gas_per_action_usd = 0.1  # Arbitrum L2 gas cost
total_usd = 1000          # Capital to simulate
min_width_pct = 0.001     # 0.1% minimum range
max_width_pct = 0.01      # 1% maximum range
total_timesteps = 100000  # Training steps
```

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Train new model
python uniswap_v3_ppo_continuous.py

# 3. Test trained model
python reward_test.py
```

## Dependencies

```bash
pip install numpy pandas gymnasium stable-baselines3 shimmy
```

## Data Requirements

Requires CSV files in `dune_pipeline/`:
- `pool_config_eth_usdt_0p3.csv`
- `token_metadata_eth_usdt_0p3.csv`
- `slot0_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`
- `swaps_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`
- `mints_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`
- `burns_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv`

## Results

With 1-hour windows and $0.10 gas cost:
- Agent learns to always ADJUST with 0.1% width
- ~$610/month potential profit (simulated)
- Trained on September 2025 data

## Trust Levels

This project relies on several layers of assumptions:

1. **Dune Data (High Trust)** - We trust that swap/mint/burn data from Dune Analytics is accurate and complete
2. **Fee Simulator (Medium Trust)** - `fee_simulator.py` calculates fees and IL; math has been spot-checked but not formally verified
3. **Gas Cost (Estimated)** - $0.10 per action is based on Arbitrum L2 research; actual costs vary $0.01-$0.50
4. **ETH/USD Prices** - External price feed used for valuation; minor discrepancies possible vs on-chain prices

## Known Limitations

1. Trained on single month of data
2. No out-of-sample validation
3. Gas cost is estimated ($0.10), actual may vary
