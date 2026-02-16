# Kongtrae – Uniswap V3 LP Strategy

Trained RL models for hourly LP decisions on Uniswap V3 ETH/USDT 0.05% pool.

## Models

| Model | File | Description |
|---|---|---|
| **PPO** | `comparison_ppo.zip` | Proximal Policy Optimization |
| **DQN** | `comparison_dqn_best.pth` | Dueling Double DQN |
| **LSTM-DQN** | `comparison_lstm_dqn_best.pth` | LSTM-enhanced DQN (24h memory) |

## How to Use (Hourly Workflow)

Run this **every hour** to get the model's LP decision:

### Step 1: Pull the latest ~200 hours of swap data

From the chain (or your indexer). Save as CSV with columns: `evt_block_time`, `sqrtPriceX96`.

```csv
evt_block_time,sqrtPriceX96,amount0,amount1,liquidity,tick
2026-02-15 00:00:00,345...000,...
...
```

### Step 2: Run the model

The tool naturally understands raw swaps and will convert them to OHLCV features automatically:

```bash
python kongtrae/inference.py \
  --model dqn \
  --swap-csv my_swaps.csv \
  --has-position \
  --current-width 1 \
  --in-range \
  --hours-since-rebalance 3
```

### Step 3: Read the output and act

```
🤖 Kongtrae Decision
  Model:   DQN
  Price:   $3,000.00
  Action:  WIDTH-3
  →  Set LP range: $2,991.00 – $3,009.00 (±0.3%)
```

- **HOLD** → do nothing, check again next hour
- **WIDTH-X** → remove current LP, set new LP at the printed range

### Example flow

```
Hour 1:  Model → WIDTH-1  → Set LP at $2,997–$3,003
Hour 2:  Model → HOLD     → Do nothing
Hour 3:  Model → WIDTH-3  → Rebalance LP to $2,991–$3,009
...
```

> The model only tells you **what to do**. You execute the LP operations on-chain yourself.

## Quick Start

### Prerequisites

```bash
pip install torch stable-baselines3 pandas numpy gymnasium
```

### Get a Decision (Simple Mode)

Just pass the current ETH price:

```bash
python kongtrae/inference.py --model dqn --price 3000
```

Output:
```
==================================================
  🤖 Kongtrae Decision
==================================================
  Model:   DQN
  Price:   $3,000.00
  Action:  WIDTH-1
  →  Set LP range: $2,997.00 – $3,003.00 (±0.1%)
  Range:   [80151, 80171]
  Width:   ±0.1%
==================================================
```

### Get a Decision (Accurate Mode)

Provide 200+ hours of OHLCV data for full technical indicators:

```bash
python kongtrae/inference.py --model dqn --ohlcv-csv hourly_eth.csv
```

CSV format:

```csv
timestamp,open,high,low,close,volume
2026-02-15 00:00:00,2950,2970,2940,2960,8000000
2026-02-15 01:00:00,2960,2965,2950,2955,6500000
...
```

### With Current Position State

If you already have an LP position, tell the model about it:

```bash
python kongtrae/inference.py \
  --model dqn \
  --price 3000 \
  --has-position \
  --current-width 1 \
  --in-range \
  --hours-since-rebalance 3
```

### Save to JSON

```bash
python kongtrae/inference.py --model dqn --price 3000 -o decision.json
```

## What Data to Pull from Chain

The model needs **5 raw values per hour**. Everything else is computed automatically.

### Raw Data Required (per hour)

| Column | What it is | Where to get it |
|---|---|---|
| `open` | ETH price at hour start | First swap's `sqrtPriceX96` in that hour |
| `high` | Highest ETH price in hour | Max price from swaps in that hour |
| `low` | Lowest ETH price in hour | Min price from swaps in that hour |
| `close` | ETH price at hour end | Last swap's `sqrtPriceX96` in that hour |
| `volume` | Total trade volume in USD | Sum of `|amount1| / 10^6` (USDT) per hour |

> **Tip:** You can also just use Binance 1h candles for ETH/USDT instead of pulling from chain. The model doesn't care where the OHLCV comes from.

### How to convert `sqrtPriceX96` to USD price

```python
price_usd = (sqrtPriceX96 / 2**96) ** 2 * 10 ** (decimals0 - decimals1)
# For ETH/USDT: decimals0=18 (WETH), decimals1=6 (USDT)
# price_usd = (sqrtPriceX96 / 2**96) ** 2 * 10**12
```

### How much history?

| Indicator | Hours needed | Why |
|---|---|---|
| RSI(14), ATR(14) | 14 | 14-period lookback |
| MA(50) | 50 | 50-hour moving average |
| **MA(200)** | **200** | 200-hour moving average |
| 7-day return | 168 | 7 × 24 hours |

**→ Pull at least 200 hourly candles for full accuracy.**

### Features computed automatically (31 total)

These are all computed from your 5 OHLCV columns — you do NOT need to compute them:

| # | Feature | From |
|---|---|---|
| 1 | `high_open_ratio` | high / open |
| 2 | `low_open_ratio` | low / open |
| 3 | `close_open_ratio` | close / open |
| 4 | `dema_ratio` | Double EMA ratio |
| 5 | `momentum_12` | 12h price momentum |
| 6 | `roc_12` | 12h rate of change |
| 7 | `atr_14` | Average True Range (14h) |
| 8 | `natr_14` | Normalized ATR (14h) |
| 9 | `adx_14` | Avg Directional Index (14h) |
| 10 | `plus_di` | +DI (directional indicator) |
| 11 | `minus_di` | −DI (directional indicator) |
| 12 | `cci_20` | Commodity Channel Index (20h) |
| 13 | `rsi_14` | Relative Strength Index (14h) |
| 14 | `macd` | MACD line |
| 15 | `macd_signal` | MACD signal line |
| 16 | `macd_hist` | MACD histogram |
| 17 | `bb_upper` | Bollinger Band upper |
| 18 | `bb_lower` | Bollinger Band lower |
| 19 | `bb_width` | Bollinger Band width |
| 20 | `stoch_k` | Stochastic %K |
| 21 | `stoch_d` | Stochastic %D |
| 22 | `volume_sma_ratio` | Volume / SMA(volume) |
| 23 | `return_1h` | 1-hour return |
| 24 | `return_24h` | 24-hour return |
| 25 | `return_7d` | 7-day return |
| 26 | `price_vs_ma50` | Price / MA(50) |
| 27 | `price_vs_ma200` | Price / MA(200) |
| 28 | `ma50_vs_ma200` | MA(50) / MA(200) |
| 29 | `market_regime` | −1 bear, 0 sideways, +1 bull |
| 30 | `trend_strength_24h` | 24h trend strength |
| 31 | `trend_strength_7d` | 7d trend strength |

Plus **7 position features** (current LP state — passed via CLI flags).

### CSV Format

```csv
timestamp,open,high,low,close,volume
2026-02-16 00:00:00,2950,2970,2940,2960,8000000
2026-02-16 01:00:00,2960,2965,2950,2955,6500000
...at least 200 rows...
2026-02-24 08:00:00,3000,3010,2990,3005,7200000
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | *required* | `ppo`, `dqn`, or `lstm` |
| `--price` | — | Current ETH price (quick mode, less accurate) |
| `--ohlcv-csv` | — | CSV with 200+ hourly OHLCV rows (accurate mode) |
| `--volume` | 5M | Hourly volume USD (quick mode only) |
| `--has-position` | false | Currently have an LP position? |
| `--current-width` | — | Current width (1,3,5,10,20,40) |
| `--in-range` | false | Position currently in range? |
| `--hours-since-rebalance` | 0 | Hours since last rebalance |
| `--device` | cpu | `cpu` or `cuda` |
| `-o` | — | Save output to JSON |

## Action Space

| Action | Width | LP Range |
|---|---|---|
| 0 | HOLD | Keep current position |
| 1 | ±1 tick spacing | ~±0.1% |
| 2 | ±3 tick spacings | ~±0.3% |
| 3 | ±5 tick spacings | ~±0.5% |
| 4 | ±10 tick spacings | ~±1.0% |
| 5 | ±20 tick spacings | ~±2.0% |
| 6 | ±40 tick spacings | ~±4.0% |

## ⚠️ Notes

- **Quick mode** (`--price` only) uses zero-valued indicators — less accurate but instant
- **Accurate mode** (`--ohlcv-csv`) needs 200+ hourly candles for full technical indicators
- **Gas costs**: Training assumed $0.05 gas (Arbitrum L2). Factor in real gas costs
- **Not financial advice**: Research models only
