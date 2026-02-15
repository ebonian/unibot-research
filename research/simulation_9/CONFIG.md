# Current Configuration (All Models)

## Environment (Shared by All Models)

### Data
| Parameter | Value | Notes |
|---|---|---|
| Data source | `swaps_*.csv` + `ETHUSDT_hourly_data_*.csv` | From simulation_6 |
| Decision interval | 1 hour | Agent acts once per hour |
| Fee calculation | Exact per-swap | Iterates every swap, paper Eq 5-6 |
| LVR calculation | Exact per-swap | Iterates every swap |
| Data split | 80/10/10 | Train / Eval / Test |
| Pool | ETH/USDT 0.05% fee tier | tick_spacing = 10 |

### Reward Function
| Component | Formula | Notes |
|---|---|---|
| Fee | `Σ (δ/(1-δ)) × L × \|Δ√p_c\|` per swap | Exact, clamped to LP range |
| LVR | `Σ (ΔV - x × Δp)` per swap | Always ≤ 0 |
| Gas cost | -$0.50 per rebalance | Arbitrum L2 |
| Swap fee cost | -0.5 × 0.05% × position_value per rebalance | ~50% of position swapped during rebalance |
| Opportunity cost | -0.00057% of position per hour | 5% APY / 8760 hrs, when out of LP range |
| **Total** | `R = fee + LVR - gas - swap_fee - opportunity` | |

### Capital
| Parameter | Value |
|---|---|
| Initial capital (all models) | $1,000 USD |

---

## DQN (Dueling Double DQN)

### State Space (37 dimensions)

**Technical indicators (31 features):**

| # | Feature | Source |
|---|---|---|
| 1 | high_open_ratio | OHLC |
| 2 | low_open_ratio | OHLC |
| 3 | close_open_ratio | OHLC |
| 4 | dema_ratio | Double EMA(12) / open |
| 5 | momentum_12 | close[t] - close[t-12] |
| 6 | roc_12 | Rate of change (12h) |
| 7 | atr_14 | Average True Range (14h) |
| 8 | natr_14 | Normalized ATR |
| 9 | adx_14 | Average Directional Index |
| 10 | plus_di | +DI (directional indicator) |
| 11 | minus_di | -DI (directional indicator) |
| 12 | cci_20 | Commodity Channel Index |
| 13 | rsi_14 | Relative Strength Index |
| 14 | macd | MACD / price |
| 15 | macd_signal | Signal line / price |
| 16 | macd_hist | MACD histogram / price |
| 17 | bb_upper | Bollinger upper / price |
| 18 | bb_lower | Bollinger lower / price |
| 19 | bb_width | Bollinger width / price |
| 20 | stoch_k | Stochastic %K |
| 21 | stoch_d | Stochastic %D |
| 22 | volume_sma_ratio | Volume / SMA(20) volume |
| 23 | return_1h | 1-hour return |
| 24 | return_24h | 24-hour return |
| 25 | return_7d | 7-day return |
| 26 | price_vs_ma50 | (price - MA50) / MA50 |
| 27 | price_vs_ma200 | (price - MA200) / MA200 |
| 28 | ma50_vs_ma200 | (MA50 - MA200) / MA200 |
| 29 | market_regime | +1 bull / 0 sideways / -1 bear |
| 30 | trend_strength_24h | |return_24h| |
| 31 | trend_strength_7d | |return_7d| |

**Position features (6 features):**

| # | Feature | Description |
|---|---|---|
| 32 | cash_ratio | cash / initial_capital |
| 33 | width_normalized | position_width / max_width |
| 34 | in_range | 1.0 if price in LP range, else 0.0 |
| 35 | position_value_ratio | position_value / initial_capital |
| 36 | position_offset | (center_tick - current_tick) / (max_width × tick_spacing) |
| 37 | price_momentum | (price - prev_price) / prev_price |

### Action Space (21 actions)

| Action | Description |
|---|---|
| 0 | HOLD (keep current position) |
| 1-5 | Width=5 ticks (~0.5%), offsets [-2, -1, 0, +1, +2] |
| 6-10 | Width=10 ticks (~1%), offsets [-2, -1, 0, +1, +2] |
| 11-15 | Width=25 ticks (~2.5%), offsets [-2, -1, 0, +1, +2] |
| 16-20 | Width=50 ticks (~5%), offsets [-2, -1, 0, +1, +2] |

Offset is in half-width units: -2 = far below price (bearish), 0 = centered, +2 = far above (bullish)

### Network Architecture

```
Input (37) → Linear(64) → ReLU → Linear(64) → ReLU
                                       ↓
                              ┌────────┴────────┐
                              ↓                  ↓
                     Value Stream          Advantage Stream
                     Linear(32)→ReLU       Linear(32)→ReLU
                     Linear(1)             Linear(21)
                              ↓                  ↓
                              └───── Q = V + (A - mean(A)) ─────→ Output (21)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Gamma (discount) | 0.9 |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay | 0.99 per episode |
| Target update rate (tau) | 0.01 (soft update) |
| Replay buffer size | 100,000 |
| Batch size | 256 |
| Grad clip | 0.7 |
| Optimizer | Adam |

---

## LSTM DQN (LSTM-based Dueling Double DQN)

### State Space

Same 37 features as DQN, but processed as a **sequence of 24 hourly states**.

Input shape: `(24, 37)` — 24 hours of history, 37 features each.

### Action Space

Same as DQN (21 actions).

### Network Architecture

```
Input (24, 37) → LSTM(hidden=64, layers=1) → last hidden state (64)
                                                      ↓
                                              Linear(64) → ReLU
                                                      ↓
                                             ┌────────┴────────┐
                                             ↓                  ↓
                                    Value Stream          Advantage Stream
                                    Linear(32)→ReLU       Linear(32)→ReLU
                                    Linear(1)             Linear(21)
                                             ↓                  ↓
                                             └── Q = V + (A - mean(A)) ──→ Output (21)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Gamma (discount) | 0.9 |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay | 0.99 per episode |
| Target update rate (tau) | 0.01 (soft update) |
| Replay buffer size | 50,000 (smaller — sequences use more memory) |
| Batch size | 64 (smaller — sequences use more memory) |
| Sequence length | 24 hours |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| FC hidden size | 64 |
| Grad clip | 0.7 |
| Optimizer | Adam |

---

## PPO (Proximal Policy Optimization)

### State Space (8 dimensions)

| # | Feature | Description |
|---|---|---|
| 1 | log(price) | Log of current ETH/USDT price |
| 2 | tick_index | Current tick (normalized) |
| 3 | width | LP interval width (in tick_spacing units) |
| 4 | liquidity_ratio | Current L / initial L |
| 5 | volatility | EWMA volatility (σ) |
| 6 | ma_24h_ratio | 24h moving average / price |
| 7 | ma_168h_ratio | 168h moving average / price |
| 8 | in_range | 1.0 if price in LP range, else 0.0 |

### Action Space (5 actions)

| Action | Description |
|---|---|
| 0 | HOLD (do nothing) |
| 1 | Width = 50 ticks (~0.5%) |
| 2 | Width = 100 ticks (~1%) |
| 3 | Width = 200 ticks (~2%) |
| 4 | Width = 500 ticks (~5%) |

No offset support — PPO always centers at current price.

### Network Architecture

```
Input (8) → Linear(64) → ReLU → Linear(64) → ReLU
                                       ↓
                              ┌────────┴────────┐
                              ↓                  ↓
                         Policy Head         Value Head
                         Linear(5)           Linear(1)
                         Softmax                 ↓
                              ↓              State Value
                         Action Probs
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 |
| Gamma (discount) | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| N epochs per update | 10 |
| N steps per env | 512 (2048 / num_envs) |
| Batch size | 256 |
| Num parallel envs | 4 |
| Network | MlpPolicy [64, 64] |
| VecNormalize | Yes (observations + rewards) |

---

## Key Differences Between Models

| Aspect | DQN | LSTM DQN | PPO |
|---|---|---|---|
| **State dim** | 37 | 37 × 24 = 888 | 8 |
| **Action dim** | 21 | 21 | 5 |
| **Offsets** | Yes (5 options) | Yes (5 options) | No |
| **Temporal** | No (single state) | Yes (24h sequence) | No (single state) |
| **Algorithm** | Off-policy (replay buffer) | Off-policy (sequence replay) | On-policy |
| **Exploration** | ε-greedy | ε-greedy | Entropy bonus |
| **Paper** | Zhang et al. 2023 | Extension | Xu & Brini 2025 |
