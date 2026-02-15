# Simulation 12: Extended Data Range (May 2025 – Feb 2026)

**Objective**: Same methodology as Simulation 11, but with extended data coverage.

## Changes from Simulation 11
- **Data range**: May 4, 2025 → Feb 12, 2026 (vs Nov 3, 2025 → Feb 3, 2026 in Sim 11)
- **Data source**: B2 pipeline (daily parquet → consolidated CSV)
- **Initial capital**: $1,000 USD
- **Pool**: ETH/USDT 0.05% fee tier (fee=500, tickSpacing=10) — same pool

## Training Data

| File | Description |
|---|---|
| `swaps_20250504_to_20260212_eth_usdt_0p3.csv` | ~1.2 GB swap data (~285 days) |
| `pool_config_eth_usdt_0p3.csv` | Fee=500 (0.05%), tickSpacing=10 |
| `token_metadata_eth_usdt_0p3.csv` | WETH (18 dec) / USDT (6 dec) |

## How to Run

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy gymnasium stable-baselines3 torch matplotlib

# Verify data integrity
python verify_data_integrity.py

# Compare all three algorithms
python compare_algorithms.py --data-dir training_data --run-dir run_001 \
  --ppo-timesteps 200000 --dqn-episodes 800 --lstm-episodes 1000
```
