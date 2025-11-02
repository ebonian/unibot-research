## Uniswap v3 Fee Simulator (simulation_2)

This simulator estimates historical fees for a hypothetical Uniswap v3 position over a time window and price range. It supports two accounting modes:

- Direct pro‑rata accounting (recommended with current datasets)
- FeeGrowthInside ("growth") accounting as in the Uniswap v3 whitepaper (requires full tick state at the start time)

### Why two modes?

- Direct pro‑rata accounting distributes each swap’s LP fee to your position by its pro‑rata share of active liquidity when your range is in-range. This matches how v3 accrues fees in real time and does not require a full tick snapshot at the start time.
- FeeGrowthInside (whitepaper) requires the pool’s tick state at the start time t0 (feeGrowthOutside and liquidityNet for all initialized ticks). Without that snapshot, growth-based results can be inaccurate.

### Current data and accuracy

With the provided CSVs (slot0, swaps, mints, burns, token metadata, pool config):

- Direct pro‑rata accounting is accurate. Use it for RL training and historical evaluation now.
- Growth accounting is NOT exact without a full tick snapshot at t0. The simulator can compute growth deltas, but they will deviate if historical tick state before t0 is missing.

### Data needed for exact whitepaper parity (growth mode)

To use the whitepaper’s feeGrowthInside method exactly, you must have at time t0:

- Global state:
  - feeGrowthGlobal0X128, feeGrowthGlobal1X128
  - slot0.tick, slot0.sqrtPriceX96
  - feeProtocol
- Per initialized tick (for all initialized ticks):
  - liquidityNet
  - feeGrowthOutside0X128, feeGrowthOutside1X128

How to obtain:

- Query an archive node at t0: iterate the tick bitmap and call `UniswapV3Pool.ticks(int24)` for every initialized tick; read the pool’s globals.
- Or reconstruct from genesis by replaying all events up to t0 (heavy and slow).

### CLI usage

Direct (recommended now):

```bash
python research/simulation_2/fee_simulator.py \
  --price-lower 3500 --price-upper 5000 \
  --total-usd 1000 \
  --start 2025-09-01T00:00:00Z --end 2025-09-01T06:00:00Z \
  --validate \
  --use-swap-liquidity \
  --accounting-mode direct
```

- `--use-swap-liquidity` uses the per-swap pool liquidity column for fee allocation. This avoids relying on an incomplete tick snapshot.
- `--validate` prints the direct accounting totals separately; in direct mode, reported fees match the validator by construction.

Growth (whitepaper) mode:

```bash
python research/simulation_2/fee_simulator.py \
  --price-lower 3500 --price-upper 5000 \
  --total-usd 1000 \
  --start 2025-09-01T00:00:00Z --end 2025-09-01T06:00:00Z \
  --accounting-mode growth
```

Note: Only use growth mode if you have populated all tick states at t0 (see Data needed above). Otherwise results can under/over count.

### Implementation notes

- Tick mapping: price-to-tick rounds down (floor) and ticks are rounded to tick spacing.
- Fees are charged on the input token per swap, net LP fees are pro‑rated by active liquidity.
- Protocol fees (if any) are accounted for before LP distribution.
- Time window: events are filtered by `[start, end]` and processed in `(time, block)` order, interleaving mints/burns with swaps.

### Outputs

The simulator prints:

- Pool and ticks crossed
- Deposit liquidity and token amounts
- Fees in token0/token1 minimal units and human units
- (With `--validate`) direct accounting totals for cross‑checking

### Recommendations

- For RL experiments: use `--accounting-mode direct --use-swap-liquidity`.
- For exact whitepaper growth parity: add a tick snapshot at t0 (globals + all initialized ticks’ feeGrowthOutside and liquidityNet) from an archive node or full reconstruction.
