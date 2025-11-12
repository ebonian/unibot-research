## Uniswap v3 Fee Simulator (Simulation 3)

This document explains, step by step, how the fee simulator in `research/simulation_3/fee_simulator.py` performs a historical simulation of Uniswap v3 fees and position valuation. It is designed to closely follow the Uniswap v3 whitepaper’s mechanics for concentrated liquidity, fee growth, and tick crossing behavior, including the handling of global fee growth and fee growth outside/inside a range of ticks.

The simulator produces a structured summary that can be serialized to JSON and includes principal vs. fee components, valuation in token units and USD, and impermanent loss metrics.

### Scope and alignment with the whitepaper

- **Concentrated liquidity and ticks**: Prices live on a tick grid with base 1.0001. Liquidity is only active inside a chosen price interval. The simulator rounds user-provided price bounds to valid ticks using the pool’s `tickSpacing`.
- **Fee growth accounting**: The simulator updates global fee growth per token (X128 fixed-point) and derives in-range fee growth via the standard “inside = global − below − above” identity using fee growth outside recorded at boundary ticks, as described in the whitepaper.
- **Tick crossing semantics**: When crossing initialized ticks, fee growth outside is flipped (subtracted from global), and active liquidity is updated by the tick’s net liquidity delta.
- **Protocol fee**: The protocol cut is taken from swap fees before they are added to the LP fee growth, using the same base-256 encoding used by v3 core.

## Inputs

You will typically load Dune-exported or node-indexed datasets with the following schemas (column names are what the simulator expects):

- **Pool config (`pool_cfg`)**

  - `pool`: pool address (string)
  - `token0`, `token1`: token contract addresses (lower/upper ordering per v3)
  - `fee`: pool fee in hundredths of a bip (e.g., 3000 for 0.3%)
  - `tickSpacing`: pool tick spacing (e.g., 60 for 0.3% fee tier)

- **Token metadata (`tokens`)**

  - `contract_address`: token address (lowercased internally)
  - `symbol`: token symbol (informational)
  - `decimals`: ERC-20 decimals

- **Slot0 snapshots (`slot0`)**

  - `call_block_time`: timestamp (UTC)
  - `output_feeProtocol`: packed protocol fee (base-256 nibble encoding)
  - `output_sqrtPriceX96`: current sqrt price Q64.96
  - `output_tick`: current tick

- **Swaps (`swaps`)**

  - `evt_block_time`: swap time (UTC)
  - `evt_block_number`: block number (for tie-breaking)
  - `sqrtPriceX96`: post-swap sqrt price Q64.96
  - `tick`: post-swap tick
  - `amount0`, `amount1`: signed swap amounts (conventionally, positive is received by the pool)
  - `liquidity` (optional): active liquidity at the swap (required if `use_swap_liquidity=True`)

- **Mints (`mints`)**

  - `evt_block_time`, `evt_block_number`
  - `tickLower`, `tickUpper`
  - `liquidity_added`: liquidity added by the mint

- **Burns (`burns`)**

  - `evt_block_time`, `evt_block_number`
  - `tickLower`, `tickUpper`
  - `liquidity_removed`: liquidity removed by the burn

- **Optional ETH/USD prices (`eth_usdt_prices`)**
  - `open_time`: candle open time (UTC)
  - `close`: ETH/USDT close price; used to express token0 (if ETH) in USD

All timestamps are parsed as timezone-aware UTC. Addresses are normalized to lowercase.

## Quick start

```python
import pandas as pd
from research.simulation_3.fee_simulator import UniswapV3FeeSimulator

pool_cfg = pd.read_csv('dune_pipeline/pool_config_eth_usdt_0p3.csv')
tokens = pd.read_csv('dune_pipeline/token_metadata_eth_usdt_0p3.csv')
slot0 = pd.read_csv('dune_pipeline/slot0_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
swaps = pd.read_csv('dune_pipeline/swaps_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
mints = pd.read_csv('dune_pipeline/mints_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
burns = pd.read_csv('dune_pipeline/burns_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
eth_usdt = pd.read_csv('dune_pipeline/token_metadata_eth_usdt_0p3.csv')  # replace with your ETH/USDT OHLCV

sim = UniswapV3FeeSimulator(pool_cfg, tokens, slot0, swaps, mints, burns, eth_usdt_prices=None)

result = sim.simulate(
    price_lower=2000,
    price_upper=2200,
    start='2025-09-01T00:00:00Z',
    end='2025-09-13T06:00:00Z',
    total_usd=1000.0,                # or provide liquidity=int(...)
    validate=False,                  # True enables direct accounting cross-check
    use_swap_liquidity=True,         # infer active L from swaps if available
    accounting_mode='growth',        # 'growth' (via feeGrowth) or 'direct' (validator)
)

print(result.to_dict())
```

## Step-by-step algorithm

This section mirrors the internal flow of `UniswapV3FeeSimulator.simulate` and related helpers.

### 1) Configuration and normalization

- Token addresses are lowercased; pool config is converted to a `PoolConfig` with `fee`, `tickSpacing`, token decimals, and symbols.
- Input tables are time-sorted: swaps by `(evt_block_time, evt_block_number)`, mints/burns similarly.

### 2) Initialize pool state at start time t0

- The simulator picks the slot0 snapshot at or before `t0` and decodes the protocol fee into per-token nibble values.
- If there is a swap strictly before `t0`, `sqrtPriceX96` and `tick` are aligned to the last pre-window swap; otherwise the slot0 snapshot is used.
- The active liquidity is initialized via one of two modes:
  - If `use_swap_liquidity=True`, use the `liquidity` column of the first in-window swap.
  - Else, compute active liquidity from pre-window events by summing `liquidity_net` across initialized ticks ≤ current tick.

### 3) Build tick map from pre-window events

- Using all mints and burns with `evt_block_time ≤ t0`, compute a map `tick -> TickInfo{liquidity_net, fee_growth_outsideX128, initialized}` where:
  - Mint at `[tickLower, tickUpper)` adds `+L` at `tickLower` and `−L` at `tickUpper`.
  - Burn does the opposite.
- This yields the net-liquidity step function across the tick grid.

### 4) Convert user price bounds to ticks

- Convert prices to raw units accounting for token decimal difference.
- Convert to tick via: `tick = floor( ln(price_raw) / ln(1.0001) )`.
- Enforce tick spacing:
  - Lower bound: floor to spacing multiple.
  - Upper bound: ceil to spacing multiple.
- Validate `lower_tick < upper_tick`.

### 5) Choose position liquidity L and starting token amounts

Let `a = sqrt(price_lower_raw) * Q96`, `b = sqrt(price_upper_raw) * Q96`, `p = sqrt(price_current_raw) * Q96`.

- If `liquidity` is provided, use it directly and compute starting token amounts using the standard v3 formulas:
  - For `p < b`: `amount0 = Δx = get_amount0_delta(max(p, a), b, L)`
  - For `p > a`: `amount1 = Δy = get_amount1_delta(a, min(p, b), L)`
- Else, for a USD budget `B` (denominated as token1 units), solve for `L` using a piecewise-linear cost per unit-L:
  - If `p ≤ a`: `amt0_per_L = ((b − a) << 96) / (b·a)` and `k = price_current_raw · amt0_per_L`
  - If `p ≥ b`: `amt1_per_L = (b − a) / Q96` and `k = amt1_per_L`
  - Else: `amt0_per_L = ((b − p) << 96) / (b·p)`, `amt1_per_L = (p − a) / Q96`, `k = amt1_per_L + price_current_raw · amt0_per_L`
  - Then `L = floor(B · 10^{decimals1} / k)` and amounts as above.

These are equivalent to the whitepaper’s liquidity and amounts relations in Q64.96 form.

### 6) Seed feeGrowthOutside at boundary ticks

To later compute fee growth “inside” the range, the simulator ensures both `lower_tick` and `upper_tick` are initialized and sets their `fee_growth_outsideX128` consistent with the start tick:

- If `tick_start ≥ lower_tick`, set `lower.outside = global_start`; else `lower.outside = 0`.
- If `tick_start < upper_tick`, set `upper.outside = 0`; else `upper.outside = global_start`.

This matches the invariant that “outside” accumulators reflect fee growth seen on the opposite side of the tick.

### 7) Process swaps in-window with interleaved events

For each swap within `[t0, t1]`, the simulator first applies any mints/burns strictly before the swap (ordered by time and block). Each update to ticks may change `liquidity_net` at tick boundaries and, if not using swap-provided liquidity, also adjusts `state.liquidity_active` when the position’s current tick lies within an added/removed range.

Then the swap price move from current `sqrtPriceX96` to the swap’s target `sqrtPriceX96` is broken into one or more “steps” that end either at the next initialized tick in the swap direction or at the final target price. For each step:

1. Determine direction: zero-for-one if price decreases; one-for-zero otherwise.
2. Compute the net input needed to move from current to step target using:
   - `Δx = get_amount0_delta(step_target, current, L_active)` for zero-for-one
   - `Δy = get_amount1_delta(current, step_target, L_active)` for one-for-zero
3. Convert net input to gross input with fee `f = fee_bps / 1_000_000`:
   - `gross_in = ceil( net_in / (1 − f) )`
   - `fee_amount = gross_in − net_in`
4. Apply protocol fee split using the pool’s nibble for the input token (base-256):
   - `lp_fee, proto_fee = apply_protocol_fee(fee_amount, proto_nibble)`
5. Update global fee growth for the input token:
   - `feeGrowthGlobalX128 += floor(lp_fee * Q128 / L_active)`
6. If a validation `FeeAccrualValidator` is active and the position is in-range, accrue the position’s pro-rata share:
   - `share = floor(L_position * lp_fee / L_active)`
7. Move the price to `step_target`. If a tick boundary is reached and it is initialized, flip its `fee_growth_outsideX128` by `outside = global − outside`, update `state.tick`, and adjust active liquidity by the tick’s `liquidity_net` sign depending on direction (unless using swap-provided liquidity).

This continues until the swap’s final target price and tick are reached.

### 8) Compute in-range fee growth and fees owed

After all swaps are processed, compute the fee growth inside the range between start and end using the standard identities.

For each token:

- Define helper `inside(global, lower_out, upper_out, tick)` as: `inside = global − below − above`, where
  - `below = lower_out` if `tick ≥ lower_tick`, else `global − lower_out`
  - `above = upper_out` if `tick < upper_tick`, else `global − upper_out`
- Then `ΔfeeGrowthInside = inside_end − inside_start` for each token.

Fees owed are computed by one of two accounting modes:

- **growth (default)**: `fees0 = floor(L * ΔfeeGrowthInside0 / Q128)`, `fees1 = floor(L * ΔfeeGrowthInside1 / Q128)`
- **direct**: use the validator’s running accrual of `lp_fee` when in-range (numerically very close and useful for cross-checking)

### 9) Separate principal from fees at start and end

Principal token amounts at a given price are computed for `(L, a, b, p)` using the piecewise formulas:

- If `p ≤ a`: all principal in token0: `x = Δx(a, b, L)`, `y = 0`
- If `p ≥ b`: all principal in token1: `x = 0`, `y = Δy(a, b, L)`
- Else: in-range split: `x = Δx(p, b, L)`, `y = Δy(a, p, L)`

Apply at the start and end prices to obtain:

- Start principal: `(x_start, y_start)`
- End principal: `(x_end, y_end)`
- End totals (including fees): `(x_end + fees0, y_end + fees1)`

### 10) Valuation, IL, and baselines

Let `raw_price = price_token1_per_token0 · 10^{decimals1 - decimals0}`. Define token1-valuation units:

- `value_token1_units(x, y) = y + x · raw_price`

Compute:

- `start_value = value_token1_units(x_start, y_start)`
- `end_value_principal = value_token1_units(x_end, y_end)`
- `end_value_total = value_token1_units(x_end + fees0, y_end + fees1)`
- `hodl_end_value = value_token1_units(x_start, y_start)` but at the end price
- **Impermanent loss (IL)**:
  - `IL = end_value_principal − hodl_end_value`
  - `IL_pct = 0 if hodl_end_value == 0 else IL / hodl_end_value`

For USD values, the simulator optionally uses `eth_usdt_prices` to convert token0 (e.g., ETH) to USD at the start and end timestamps. Token1 amounts are already in USD-like units when token1 is a USD stablecoin.

## Options and flags

- **`use_swap_liquidity`**: When true, the active liquidity comes from each swap row’s `liquidity` column (typical for Dune exports). When false, active liquidity is derived from net-liquidity at the current tick via pre-window mints/burns and updated on tick crossings.
- **`accounting_mode`**: `'growth'` uses the feeGrowth accumulators; `'direct'` uses a validator that accrues the LP’s pro-rata share per step when in-range (useful for validation and sensitivity checks).
- **`validate`**: Shortcut to enable the validator regardless of accounting mode.
- **`protocol_fee_encoding`**: `'base256'` (default, matches v3 core) or `'one_over'` (experimental); affects how the protocol cut is applied to fees.

## Output schema

`simulate(...)` returns a `SimulationSummaryOut`. Key fields:

- **`pool`**, **`fee_bps`**: identifiers
- **`tokens`**: `{token0: {address, symbol, decimals}, token1: {…}}`
- **`window_utc`**: `{start, end}` ISO timestamps
- **`position_ticks`**: `{lower, upper}`
- **`price`**: `{start, end}` as token1 per token0
- **`start_tokens`**: deposit at t0 as amounts and USD (if token0 USD is available)
- **`start_principal_tokens`**: principal only at t0
- **`end_tokens`**: end-of-window totals (principal + fees)
- **`end_principal_tokens`**: principal only at t1
- **`fees_usd`**: fees valued in USD-equivalent units (token1), plus token0 USD if a price feed is provided
- **`impermanent_loss`**: `{usd, pct}` relative to HODL baseline at end price
- **`valuation_baselines`**: `{hodl_end_usd, end_principal_usd}`

All numeric values are converted to floats for readability in the summary; underlying math uses high precision.

## Numerical precision and units

- The simulator uses Python `decimal` with precision 80 for price/tick conversions.
- Fixed-point constants: `Q96 = 2^96`, `Q128 = 2^128` for Uniswap v3 formulas.
- Fees are expressed with `FEE_DENOM = 1_000_000` (hundredths of a bip), matching v3.
- Amounts are computed in raw token units (wei-like). Display fields scale to human units.

## Assumptions, caveats, and limitations

- Swaps must include accurate `sqrtPriceX96`, `tick`, and preferably `liquidity` if `use_swap_liquidity=True`.
- Pre-window mints/burns are used to reconstruct initialized ticks and active liquidity when not using swap-provided liquidity; incomplete event history can bias results.
- Interleaving of mints/burns and swaps is resolved by `(time, block_number)` ordering; only events strictly before a swap are applied before that swap.
- The simulator treats fees as accrued but not reinvested into `L`; principal and fees are tracked separately.
- USD valuation of token0 depends on an external price feed; if absent, token0-USD fields are `null`.
- Time window is inclusive of both `start` and `end` swap timestamps.

## Troubleshooting

- “Swaps dataset missing liquidity column; cannot use use_swap_liquidity”: either provide swaps with a `liquidity` column or set `use_swap_liquidity=False`.
- “No swaps in selected window to infer liquidity”: widen the simulation window or disable `use_swap_liquidity`.
- “lower_tick must be < upper_tick”: check price bounds and pool `tickSpacing`.

## References

- Uniswap v3 whitepaper (concentrated liquidity, ticks, fee growth inside/outside, protocol fee encoding)
- Uniswap v3 core mathematical formulas for `Δx`, `Δy`, fee growth, and tick-to-price conversions

## File map

- `research/simulation_3/fee_simulator.py`: implementation
- `dune_pipeline/*.csv`: example data export format (pool config, slot0, swaps, mints, burns, token metadata)
