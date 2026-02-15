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
| Fee | `Σ (δ/(1-δ)) × L × |Δ√p_c|` per swap | Exact, clamped to LP range |
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

## Unified Action Space (All Models — 7 actions)

| Action | Width | Range (±) | Approx % |
|---|---|---|---|
| 0 | HOLD | — | Do nothing |
| 1 | 1 | ±10 ticks | ~0.1% |
| 2 | 3 | ±30 ticks | ~0.3% |
| 3 | 5 | ±50 ticks | ~0.5% |
| 4 | 10 | ±100 ticks | ~1.0% |
| 5 | 20 | ±200 ticks | ~2.0% |
| 6 | 40 | ±400 ticks | ~4.0% |

All non-HOLD actions center the LP range at the current price. No directional offsets.

---

## Unified State Space (All Models — 36 dimensions)

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

**Position features (5 features):**

| # | Feature | Description |
|---|---|---|
| 32 | cash_ratio | cash / initial_capital |
| 33 | width_normalized | position_width / max_width |
| 34 | in_range | 1.0 if price in LP range, else 0.0 |
| 35 | position_value_ratio | position_value / initial_capital |
| 36 | price_momentum | (price - prev_price) / prev_price |

---

## DQN (Dueling Double DQN)

### Network Architecture

```
Input (36) → Linear(64) → ReLU → Linear(64) → ReLU
                                       ↓
                              ┌────────┴────────┐
                              ↓                  ↓
                     Value Stream          Advantage Stream
                     Linear(32)→ReLU       Linear(32)→ReLU
                     Linear(1)             Linear(7)
                              ↓                  ↓
                              └───── Q = V + (A - mean(A)) ─────→ Output (7)
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

### Input

Same 36 features as DQN, but processed as a **sequence of 24 hourly states**.
Input shape: `(24, 36)` — 24 hours of history, 36 features each.

### Network Architecture

```
Input (24, 36) → LSTM(hidden=64, layers=1) → last hidden state (64)
                                                      ↓
                                              Linear(64) → ReLU
                                                      ↓
                                             ┌────────┴────────┐
                                             ↓                  ↓
                                    Value Stream          Advantage Stream
                                    Linear(32)→ReLU       Linear(32)→ReLU
                                    Linear(1)             Linear(7)
                                             ↓                  ↓
                                             └── Q = V + (A - mean(A)) ──→ Output (7)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Gamma (discount) | 0.9 |
| Epsilon start/end | 1.0 → 0.05 |
| Replay buffer size | 50,000 |
| Batch size | 64 |
| Sequence length | 24 hours |
| LSTM hidden size | 64 |

---

## PPO (Proximal Policy Optimization)

### Network Architecture

```
Input (36) → Linear(64) → ReLU → Linear(64) → ReLU
                                       ↓
                              ┌────────┴────────┐
                              ↓                  ↓
                         Policy Head         Value Head
                         Linear(7)           Linear(1)
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
| Batch size | 256 |
| Num parallel envs | 4 |
| VecNormalize | Yes (observations + rewards) |

---

## Key Differences Between Models

| Aspect | DQN | LSTM DQN | PPO |
|---|---|---|---|
| **State dim** | 36 | 36 × 24 = 864 | 36 |
| **Action dim** | 7 | 7 | 7 |
| **Temporal** | No (single state) | Yes (24h sequence) | No (single state) |
| **Algorithm** | Off-policy (replay buffer) | Off-policy (sequence replay) | On-policy |
| **Exploration** | ε-greedy | ε-greedy | Entropy bonus |
| **Capital model** | Cash tracking | Cash tracking | Re-deploys $1000 each rebalance |
