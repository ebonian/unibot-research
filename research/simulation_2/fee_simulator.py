from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, Tuple

import argparse
import pandas as pd


# High precision for tick/price math
getcontext().prec = 80

Q96 = 2 ** 96
Q128 = 2 ** 128
FEE_DENOM = 1_000_000  # Uniswap v3 fee in hundredths of a bip


@dataclass
class PoolConfig:
    pool: str
    token0: str
    token1: str
    fee: int
    tick_spacing: int
    decimals0: int
    decimals1: int


@dataclass
class PoolState:
    sqrt_price_x96: int
    tick: int
    liquidity_active: int
    fee_protocol_token0: int
    fee_protocol_token1: int
    fee_growth_global0_x128: int = 0
    fee_growth_global1_x128: int = 0


@dataclass
class TickInfo:
    liquidity_net: int
    fee_growth_outside0_x128: int = 0
    fee_growth_outside1_x128: int = 0
    initialized: bool = False


def decode_fee_protocol(fp: int) -> Tuple[int, int]:
    t0 = fp & 0x0F
    t1 = (fp >> 4) & 0x0F
    return t0, t1


def tick_to_sqrt_price_x96(tick: int) -> int:
    sqrt_price = (Decimal('1.0001') ** Decimal(tick)).sqrt()
    return int((sqrt_price * Q96).to_integral_value(rounding='ROUND_FLOOR'))


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> Decimal:
    # Human price token1 per token0 (matches earlier visualization)
    p = Decimal(sqrt_price_x96) / Decimal(Q96)
    return (p * p) * (Decimal(10) ** Decimal(decimals0 - decimals1))


def price_to_tick(price_token1_per_token0: Decimal, decimals0: int, decimals1: int) -> int:
    if price_token1_per_token0 <= 0:
        raise ValueError('price must be positive')
    # Convert to raw ratio before tick mapping
    price_raw = price_token1_per_token0 * (Decimal(10) ** Decimal(decimals1 - decimals0))
    return int((price_raw.ln() / Decimal('1.0001').ln()).to_integral_value(rounding='ROUND_HALF_EVEN'))


def round_tick_to_spacing(tick: int, spacing: int) -> int:
    # Round to nearest initialized tick per spacing (downwards like UI)
    if tick >= 0:
        return tick - (tick % spacing)
    # For negative ticks, ensure multiples of spacing
    mod = (-tick) % spacing
    return tick - (spacing - mod) if mod != 0 else tick


def get_amount0_delta(sqrt_pa_x96: int, sqrt_pb_x96: int, liquidity: int, round_up: bool) -> int:
    if sqrt_pa_x96 > sqrt_pb_x96:
        sqrt_pa_x96, sqrt_pb_x96 = sqrt_pb_x96, sqrt_pa_x96
    num = (int(liquidity) * (sqrt_pb_x96 - sqrt_pa_x96)) << 96
    denom = sqrt_pb_x96 * sqrt_pa_x96
    if round_up:
        return (num + denom - 1) // denom
    else:
        return num // denom


def get_amount1_delta(sqrt_pa_x96: int, sqrt_pb_x96: int, liquidity: int, round_up: bool) -> int:
    if sqrt_pa_x96 > sqrt_pb_x96:
        sqrt_pa_x96, sqrt_pb_x96 = sqrt_pb_x96, sqrt_pa_x96
    num = int(liquidity) * (sqrt_pb_x96 - sqrt_pa_x96)
    amount = num // Q96
    if round_up and (num % Q96 != 0):
        return amount + 1
    return amount


def apply_protocol_fee(fee_amount: int, proto_b4: int) -> Tuple[int, int]:
    if proto_b4 == 0:
        return fee_amount, 0
    lp = (fee_amount * (256 - proto_b4)) // 256
    return lp, fee_amount - lp


def build_liquidity_net_from_events(mints: pd.DataFrame, burns: pd.DataFrame) -> Dict[int, TickInfo]:
    ticks: Dict[int, TickInfo] = {}
    def add_tick(t: int, delta: int):
        info = ticks.get(t)
        if info is None:
            info = TickInfo(liquidity_net=0, initialized=True)
            ticks[t] = info
        info.liquidity_net += delta
        info.initialized = True
    for _, r in mints.iterrows():
        tl = int(r['tickLower'])
        tu = int(r['tickUpper'])
        L = int(r['liquidity_added'])
        add_tick(tl, +L)
        add_tick(tu, -L)
    for _, r in burns.iterrows():
        tl = int(r['tickLower'])
        tu = int(r['tickUpper'])
        L = int(r['liquidity_removed'])
        add_tick(tl, -L)
        add_tick(tu, +L)
    return ticks


def init_active_liquidity_from_ticks(current_tick: int, ticks: Dict[int, TickInfo]) -> int:
    total = 0
    for t, info in ticks.items():
        if info.initialized and t <= current_tick:
            total += info.liquidity_net
    return max(total, 0)


class FeeAccrualValidator:
    def __init__(self, tick_lower: int, tick_upper: int, liquidity_L: int):
        self.tick_lower = tick_lower
        self.tick_upper = tick_upper
        self.L = int(liquidity_L)
        self.accr_token0: int = 0
        self.accr_token1: int = 0


def simulate_swaps(swaps: pd.DataFrame,
                   pool_fee_bps: int,
                   init_state: PoolState,
                   ticks: Dict[int, TickInfo],
                   validator: FeeAccrualValidator | None = None) -> Tuple[PoolState, Dict[int, TickInfo]]:
    state = PoolState(
        sqrt_price_x96=init_state.sqrt_price_x96,
        tick=init_state.tick,
        liquidity_active=init_state.liquidity_active,
        fee_protocol_token0=init_state.fee_protocol_token0,
        fee_protocol_token1=init_state.fee_protocol_token1,
        fee_growth_global0_x128=init_state.fee_growth_global0_x128,
        fee_growth_global1_x128=init_state.fee_growth_global1_x128,
    )

    for _, s in swaps.iterrows():
        target_sqrt = int(s['sqrtPriceX96'])
        target_tick = int(s['tick'])
        amt0 = int(s['amount0'])
        amt1 = int(s['amount1'])
        if amt0 == 0 and amt1 == 0:
            continue
        if amt0 > 0 and amt1 < 0:
            input_token = 0
            zero_for_one = True
        elif amt1 > 0 and amt0 < 0:
            input_token = 1
            zero_for_one = False
        else:
            zero_for_one = target_sqrt < state.sqrt_price_x96
            input_token = 0 if zero_for_one else 1

        while state.sqrt_price_x96 != target_sqrt:
            if zero_for_one:
                next_tick = state.tick - 1
                next_sqrt = tick_to_sqrt_price_x96(next_tick)
                step_target = max(next_sqrt, target_sqrt)
                in_net = get_amount0_delta(step_target, state.sqrt_price_x96, state.liquidity_active, round_up=True)
            else:
                next_tick = state.tick + 1
                next_sqrt = tick_to_sqrt_price_x96(next_tick)
                step_target = min(next_sqrt, target_sqrt)
                in_net = get_amount1_delta(state.sqrt_price_x96, step_target, state.liquidity_active, round_up=True)

            gross_in = (in_net * FEE_DENOM + (FEE_DENOM - pool_fee_bps) - 1) // (FEE_DENOM - pool_fee_bps)
            fee_amount = gross_in - in_net

            if input_token == 0:
                lp_fee, _ = apply_protocol_fee(fee_amount, state.fee_protocol_token0)
                if state.liquidity_active > 0 and lp_fee > 0:
                    state.fee_growth_global0_x128 += (lp_fee * Q128) // state.liquidity_active
            else:
                lp_fee, _ = apply_protocol_fee(fee_amount, state.fee_protocol_token1)
                if state.liquidity_active > 0 and lp_fee > 0:
                    state.fee_growth_global1_x128 += (lp_fee * Q128) // state.liquidity_active

            # Optional: direct accrual validator using pro-rata share L/L_active when in range
            if validator is not None and state.liquidity_active > 0 and lp_fee > 0:
                in_range = (validator.tick_lower <= state.tick < validator.tick_upper)
                if in_range and validator.L > 0:
                    share_num = validator.L * lp_fee
                    share = share_num // state.liquidity_active
                    if input_token == 0:
                        validator.accr_token0 += share
                    else:
                        validator.accr_token1 += share

            state.sqrt_price_x96 = step_target
            if state.sqrt_price_x96 == next_sqrt:
                info = ticks.get(next_tick)
                if info is None:
                    info = TickInfo(liquidity_net=0, initialized=False)
                    ticks[next_tick] = info
                info.fee_growth_outside0_x128 = state.fee_growth_global0_x128
                info.fee_growth_outside1_x128 = state.fee_growth_global1_x128
                if zero_for_one:
                    state.liquidity_active -= info.liquidity_net
                else:
                    state.liquidity_active += info.liquidity_net
                state.tick = next_tick
            else:
                state.tick = target_tick

    return state, ticks


def fee_growth_inside_delta(ticks: Dict[int, TickInfo],
                            tick_lower: int,
                            tick_upper: int,
                            global0_x128_start: int,
                            global1_x128_start: int,
                            global0_x128_end: int,
                            global1_x128_end: int,
                            tick_at_start: int,
                            tick_at_end: int) -> Tuple[int, int]:
    lower = ticks.get(tick_lower, TickInfo(0, 0, 0, False))
    upper = ticks.get(tick_upper, TickInfo(0, 0, 0, False))

    def inside(global_start, global_end, lower_out, upper_out, tick_start, tick_end):
        below_start = lower_out if tick_start >= tick_lower else global_start - lower_out
        above_start = upper_out if tick_start < tick_upper else global_start - upper_out
        inside_start = global_start - below_start - above_start

        below_end = lower_out if tick_end >= tick_lower else global_end - lower_out
        above_end = upper_out if tick_end < tick_upper else global_end - upper_out
        inside_end = global_end - below_end - above_end
        return inside_end - inside_start

    d0 = inside(global0_x128_start, global0_x128_end,
                lower.fee_growth_outside0_x128, upper.fee_growth_outside0_x128,
                tick_at_start, tick_at_end)
    d1 = inside(global1_x128_start, global1_x128_end,
                lower.fee_growth_outside1_x128, upper.fee_growth_outside1_x128,
                tick_at_start, tick_at_end)
    return d0, d1


def compute_L_from_deposit(price_lower: Decimal, price_upper: Decimal, price_current: Decimal,
                           amount0: int | None, amount1: int | None,
                           decimals0: int, decimals1: int) -> Tuple[int, int, int]:
    # Convert human prices to raw ratios before sqrt
    raw_lower = price_lower * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_upper = price_upper * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_current = price_current * (Decimal(10) ** Decimal(decimals1 - decimals0))
    a = int((raw_lower.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    b = int((raw_upper.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    p = int((raw_current.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    if amount0 is None and amount1 is None:
        L = 10 ** 18
    elif amount0 is not None and amount1 is None:
        # L = amount0 * (a*b) / (b - a) with proper scaling
        L = (int(amount0) * (a * b // Q96)) // (b - a)
    elif amount1 is not None and amount0 is None:
        # L = amount1 / (b - a)
        L = (int(amount1) * Q96) // (b - a)
    else:
        L = 10 ** 18
    amt0 = get_amount0_delta(p if p < b else a, b, L, round_up=True) if p < b else 0
    amt1 = get_amount1_delta(a, p if p > a else b, L, round_up=True) if p > a else 0
    return L, int(amt0), int(amt1)


def compute_L_for_budget_usd(price_lower: Decimal, price_upper: Decimal, price_current: Decimal,
                             total_usd: Decimal, decimals0: int, decimals1: int) -> Tuple[int, int, int]:
    # Convert to raw sqrt prices
    raw_lower = price_lower * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_upper = price_upper * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_current = price_current * (Decimal(10) ** Decimal(decimals1 - decimals0))
    a = int((raw_lower.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    b = int((raw_upper.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    p = int((raw_current.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))

    # Value conversion: token0 minimal units * raw_current => token1 minimal units (USD cents)

    if p <= a:
        # Pure token0 deposit: USD per unit L
        num = (b - a) << 96
        denom = b * a
        amt0_per_L = Decimal(num) / Decimal(denom)
        k = raw_current * amt0_per_L
    elif p >= b:
        # Pure token1 deposit
        amt1_per_L = Decimal(b - a) / Decimal(Q96)
        k = amt1_per_L
    else:
        # Mixed region
        amt0_per_L = Decimal(((b - p) << 96)) / Decimal(b * p)
        amt1_per_L = Decimal(p - a) / Decimal(Q96)
        k = amt1_per_L + raw_current * amt0_per_L

    # Convert USD budget to token1 minimal units (USDT has decimals1)
    total_usd_units = total_usd * (Decimal(10) ** Decimal(decimals1))
    if k == 0:
        L = 0
    else:
        L = int((total_usd_units / k).to_integral_value(rounding='ROUND_FLOOR'))

    # Compute final amounts with integer functions
    amt0 = get_amount0_delta(p if p < b else a, b, L, round_up=True) if p < b else 0
    amt1 = get_amount1_delta(a, p if p > a else b, L, round_up=True) if p > a else 0
    return L, int(amt0), int(amt1)


def load_pool_context(base: Path) -> Tuple[PoolConfig, PoolState]:
    pool_cfg = pd.read_csv(base / 'dune_pipeline' / 'pool_config_eth_usdt_0p3.csv')
    tokens = pd.read_csv(base / 'dune_pipeline' / 'token_metadata_eth_usdt_0p3.csv')
    tokens['contract_address'] = tokens['contract_address'].str.lower()
    t0 = tokens.set_index('contract_address').loc[pool_cfg.loc[0, 'token0'].lower()]
    t1 = tokens.set_index('contract_address').loc[pool_cfg.loc[0, 'token1'].lower()]
    cfg = PoolConfig(
        pool=pool_cfg.loc[0, 'pool'],
        token0=pool_cfg.loc[0, 'token0'].lower(),
        token1=pool_cfg.loc[0, 'token1'].lower(),
        fee=int(pool_cfg.loc[0, 'fee']),
        tick_spacing=int(pool_cfg.loc[0, 'tickSpacing']),
        decimals0=int(t0['decimals']),
        decimals1=int(t1['decimals']),
    )
    slot0 = pd.read_csv(base / 'dune_pipeline' / 'slot0_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
    slot0 = slot0.sort_values('call_block_time').iloc[0]
    fee_proto0, fee_proto1 = decode_fee_protocol(int(slot0['output_feeProtocol']))
    init = PoolState(
        sqrt_price_x96=int(slot0['output_sqrtPriceX96']),
        tick=int(slot0['output_tick']),
        liquidity_active=0,
        fee_protocol_token0=fee_proto0,
        fee_protocol_token1=fee_proto1,
    )
    return cfg, init


def load_events(base: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    swaps = pd.read_csv(base / 'dune_pipeline' / 'swaps_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
    swaps['evt_block_time'] = pd.to_datetime(swaps['evt_block_time'], utc=True)
    swaps = swaps.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
    mints = pd.read_csv(base / 'dune_pipeline' / 'mints_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
    mints['evt_block_time'] = pd.to_datetime(mints['evt_block_time'], utc=True)
    mints = mints.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
    burns = pd.read_csv(base / 'dune_pipeline' / 'burns_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
    burns['evt_block_time'] = pd.to_datetime(burns['evt_block_time'], utc=True)
    burns = burns.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
    return swaps, mints, burns


def main():
    parser = argparse.ArgumentParser(description='Uniswap v3 high-accuracy fee simulator')
    parser.add_argument('--price-lower', type=float, required=True, help='Lower price (token1 per token0)')
    parser.add_argument('--price-upper', type=float, required=True, help='Upper price (token1 per token0)')
    parser.add_argument('--start', type=str, required=True, help='Start time (ISO, UTC)')
    parser.add_argument('--end', type=str, required=True, help='End time (ISO, UTC)')
    parser.add_argument('--liquidity', type=int, default=None, help='Optional liquidity units (overrides USD budget)')
    parser.add_argument('--total-usd', type=float, default=None, help='Total USD budget to deposit (token1 is USD for ETH/USDT)')
    parser.add_argument('--validate', action='store_true', help='Enable fee accrual validation via direct pro-rata accounting')
    args = parser.parse_args()

    base = Path('/home/poon/developments/ice-senior-project')
    cfg, init = load_pool_context(base)
    swaps, mints, burns = load_events(base)

    # Filter window
    t0 = pd.to_datetime(args.start, utc=True)
    t1 = pd.to_datetime(args.end, utc=True)
    swaps_w = swaps[(swaps['evt_block_time'] >= t0) & (swaps['evt_block_time'] <= t1)].copy()
    mints_w = mints[(mints['evt_block_time'] <= t1)].copy()
    burns_w = burns[(burns['evt_block_time'] <= t1)].copy()

    # Build tick map to t1 (best we can without a genesis snapshot)
    tick_map = build_liquidity_net_from_events(mints_w, burns_w)
    init.liquidity_active = init_active_liquidity_from_ticks(init.tick, tick_map)

    # Capture start snapshot
    global0_start = init.fee_growth_global0_x128
    global1_start = init.fee_growth_global1_x128
    tick_start = init.tick

    # Optional validator
    validator = None
    # Prepare range
    lower_px = Decimal(str(args.price_lower))
    upper_px = Decimal(str(args.price_upper))
    lower_tick = round_tick_to_spacing(price_to_tick(lower_px, cfg.decimals0, cfg.decimals1), cfg.tick_spacing)
    upper_tick = round_tick_to_spacing(price_to_tick(upper_px, cfg.decimals0, cfg.decimals1), cfg.tick_spacing)
    if lower_tick >= upper_tick:
        raise ValueError('lower_tick must be < upper_tick')
    # Compute deposit from range at start price
    start_price = sqrt_price_x96_to_price(init.sqrt_price_x96, cfg.decimals0, cfg.decimals1)
    if args.liquidity is not None:
        L, amount0, amount1 = compute_L_from_deposit(lower_px, upper_px, start_price,
                                                     amount0=None, amount1=None,
                                                     decimals0=cfg.decimals0, decimals1=cfg.decimals1)
        L = int(args.liquidity)
        a = Decimal(args.price_lower) * (Decimal(10) ** Decimal(cfg.decimals1 - cfg.decimals0))
        b = Decimal(args.price_upper) * (Decimal(10) ** Decimal(cfg.decimals1 - cfg.decimals0))
        p = start_price * (Decimal(10) ** Decimal(cfg.decimals1 - cfg.decimals0))
        sqrt_a = int((a.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
        sqrt_b = int((b.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
        sqrt_p = int((p.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
        amount0 = get_amount0_delta(sqrt_p if sqrt_p < sqrt_b else sqrt_a, sqrt_b, L, round_up=True) if sqrt_p < sqrt_b else 0
        amount1 = get_amount1_delta(sqrt_a, sqrt_p if sqrt_p > sqrt_a else sqrt_b, L, round_up=True) if sqrt_p > sqrt_a else 0
    else:
        if args.total_usd is None:
            raise SystemExit('Either --liquidity or --total-usd must be provided')
        L, amount0, amount1 = compute_L_for_budget_usd(lower_px, upper_px, start_price,
                                                       Decimal(str(args.total_usd)),
                                                       cfg.decimals0, cfg.decimals1)
    if args.validate:
        validator = FeeAccrualValidator(lower_tick, upper_tick, L)
    # Run simulation
    final_state, tick_map_out = simulate_swaps(swaps_w, cfg.fee, init, tick_map, validator)

    

    # Fee growth inside delta over [t0, t1]
    d0, d1 = fee_growth_inside_delta(
        tick_map_out,
        lower_tick,
        upper_tick,
        global0_start,
        global1_start,
        final_state.fee_growth_global0_x128,
        final_state.fee_growth_global1_x128,
        tick_start,
        final_state.tick,
    )
    fees0 = (L * d0) // Q128
    fees1 = (L * d1) // Q128

    print({'pool': cfg.pool,
           'start_tick': tick_start,
           'end_tick': final_state.tick,
           'range_ticks': (lower_tick, upper_tick)})
    print({'start_price_token1_per_token0': str(start_price)})
    print({'deposit_liquidity': L, 'amount0_wei': amount0, 'amount1_wei': amount1})
    print({'fees_token0_wei': int(fees0), 'fees_token1_wei': int(fees1)})
    print({'fees_token0_eth': float(Decimal(fees0) / Decimal(10 ** cfg.decimals0)),
           'fees_token1_eth': float(Decimal(fees1) / Decimal(10 ** cfg.decimals1))})

    if args.validate and validator is not None:
        print({'validation_direct_pro_rata_token0_wei': validator.accr_token0,
               'validation_direct_pro_rata_token1_wei': validator.accr_token1})


if __name__ == '__main__':
    main()


