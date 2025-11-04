from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, Tuple

import argparse
import pandas as pd
from bisect import bisect_left, bisect_right, insort
import json


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
    symbol0: str = ''
    symbol1: str = ''


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
    return int((price_raw.ln() / Decimal('1.0001').ln()).to_integral_value(rounding='ROUND_FLOOR'))


def round_tick_to_spacing(tick: int, spacing: int) -> int:
    # Round down to nearest initialized tick per spacing
    if tick >= 0:
        return tick - (tick % spacing)
    # For negative ticks, ensure multiples of spacing
    mod = (-tick) % spacing
    return tick - (spacing - mod) if mod != 0 else tick


def round_up_to_spacing(tick: int, spacing: int) -> int:
    # Round up to next initialized tick per spacing
    rem = tick % spacing
    if rem == 0:
        return tick
    return tick + (spacing - rem)


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


def apply_protocol_fee(fee_amount: int, proto: int, encoding: str = 'base256') -> Tuple[int, int]:
    if proto == 0:
        return fee_amount, 0
    if encoding == 'base256':
        protocol_cut = (fee_amount * int(proto)) // 256
        lp_fee = fee_amount - protocol_cut
        return lp_fee, protocol_cut
    elif encoding == 'one_over':
        # Legacy behavior: protocol takes 1/proto share
        protocol_cut = fee_amount // int(proto)
        lp_fee = fee_amount - protocol_cut
        return lp_fee, protocol_cut
    else:
        # Default to base256 if unknown
        protocol_cut = (fee_amount * int(proto)) // 256
        lp_fee = fee_amount - protocol_cut
        return lp_fee, protocol_cut


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
                   validator: FeeAccrualValidator | None = None,
                   mints_post: pd.DataFrame | None = None,
                   burns_post: pd.DataFrame | None = None,
                   use_swap_liquidity: bool = False,
                   protocol_fee_encoding: str = 'base256') -> Tuple[PoolState, Dict[int, TickInfo]]:
    state = PoolState(
        sqrt_price_x96=init_state.sqrt_price_x96,
        tick=init_state.tick,
        liquidity_active=init_state.liquidity_active,
        fee_protocol_token0=init_state.fee_protocol_token0,
        fee_protocol_token1=init_state.fee_protocol_token1,
        fee_growth_global0_x128=init_state.fee_growth_global0_x128,
        fee_growth_global1_x128=init_state.fee_growth_global1_x128,
    )

    # Maintain sorted set of initialized ticks
    initialized_ticks_sorted = sorted([t for t, info in ticks.items() if info.initialized])

    def ensure_tick_initialized(tick_key: int):
        info = ticks.get(tick_key)
        if info is None:
            info = TickInfo(liquidity_net=0, initialized=True)
            # Initialize outside based on relation to current tick per spec
            if tick_key <= state.tick:
                info.fee_growth_outside0_x128 = state.fee_growth_global0_x128
                info.fee_growth_outside1_x128 = state.fee_growth_global1_x128
            else:
                info.fee_growth_outside0_x128 = 0
                info.fee_growth_outside1_x128 = 0
            ticks[tick_key] = info
            insort(initialized_ticks_sorted, tick_key)
        elif not info.initialized:
            info.initialized = True
            if tick_key <= state.tick:
                info.fee_growth_outside0_x128 = state.fee_growth_global0_x128
                info.fee_growth_outside1_x128 = state.fee_growth_global1_x128
            else:
                info.fee_growth_outside0_x128 = 0
                info.fee_growth_outside1_x128 = 0
            insort(initialized_ticks_sorted, tick_key)
        return info

    def next_initialized_left(current_tick: int) -> int | None:
        idx = bisect_left(initialized_ticks_sorted, current_tick) - 1
        if idx >= 0:
            return initialized_ticks_sorted[idx]
        return None

    def next_initialized_right(current_tick: int) -> int | None:
        idx = bisect_right(initialized_ticks_sorted, current_tick)
        if idx < len(initialized_ticks_sorted):
            return initialized_ticks_sorted[idx]
        return None

    mi = 0
    bi = 0
    mlen = len(mints_post) if mints_post is not None else 0
    blen = len(burns_post) if burns_post is not None else 0

    for _, s in swaps.iterrows():
        # Process all mints/burns up to this swap (by time, then block number)
        if mints_post is not None or burns_post is not None:
            swap_time = s['evt_block_time']
            swap_bn = int(s['evt_block_number'])

            while True:
                did_one = False
                # choose next event among mint/burn with earliest (time, block)
                next_m = None
                next_b = None
                if mints_post is not None and mi < mlen:
                    rm = mints_post.iloc[mi]
                    next_m = (rm['evt_block_time'], int(rm['evt_block_number']), 'mint', rm)
                if burns_post is not None and bi < blen:
                    rb = burns_post.iloc[bi]
                    next_b = (rb['evt_block_time'], int(rb['evt_block_number']), 'burn', rb)
                nxt = None
                if next_m is not None and (next_b is None or (next_m[0], next_m[1]) <= (next_b[0], next_b[1])):
                    nxt = next_m
                elif next_b is not None:
                    nxt = next_b

                if nxt is None:
                    break

                n_time, n_bn, ev_type, row = nxt
                if (n_time, n_bn) >= (swap_time, swap_bn):
                    break

                tl = int(row['tickLower'])
                tu = int(row['tickUpper'])
                if ev_type == 'mint':
                    Lchg = int(row['liquidity_added'])
                    # initialize ticks on first sight and adjust liquidity nets
                    info_l = ticks.get(tl)
                    if info_l is None or not info_l.initialized:
                        info_l = ensure_tick_initialized(tl)
                    info_l.liquidity_net += Lchg

                    info_u = ticks.get(tu)
                    if info_u is None or not info_u.initialized:
                        info_u = ensure_tick_initialized(tu)
                    info_u.liquidity_net -= Lchg

                    if (not use_swap_liquidity) and (tl <= state.tick < tu):
                        state.liquidity_active += Lchg
                    mi += 1
                    did_one = True
                else:
                    Lchg = int(row['liquidity_removed'])
                    info_l = ticks.get(tl)
                    if info_l is None or not info_l.initialized:
                        info_l = ensure_tick_initialized(tl)
                    info_l.liquidity_net -= Lchg

                    info_u = ticks.get(tu)
                    if info_u is None or not info_u.initialized:
                        info_u = ensure_tick_initialized(tu)
                    info_u.liquidity_net += Lchg

                    if (not use_swap_liquidity) and (tl <= state.tick < tu):
                        state.liquidity_active = max(0, state.liquidity_active - Lchg)
                    bi += 1
                    did_one = True

                if not did_one:
                    break
        target_sqrt = int(s['sqrtPriceX96'])
        target_tick = int(s['tick'])
        amt0 = int(s['amount0'])
        amt1 = int(s['amount1'])
        # Choose active liquidity basis
        swap_L = int(s['liquidity']) if 'liquidity' in s and pd.notna(s['liquidity']) else state.liquidity_active
        if use_swap_liquidity:
            state.liquidity_active = int(swap_L)
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
            active_L = int(swap_L) if use_swap_liquidity else state.liquidity_active
            if zero_for_one:
                # move left to next initialized tick or target
                nxt_tick = next_initialized_left(state.tick)
                next_sqrt = tick_to_sqrt_price_x96(nxt_tick) if nxt_tick is not None else target_sqrt
                step_target = max(next_sqrt, target_sqrt)
                in_net = get_amount0_delta(step_target, state.sqrt_price_x96, active_L, round_up=True)
            else:
                # move right to next initialized tick or target
                nxt_tick = next_initialized_right(state.tick)
                next_sqrt = tick_to_sqrt_price_x96(nxt_tick) if nxt_tick is not None else target_sqrt
                step_target = min(next_sqrt, target_sqrt)
                in_net = get_amount1_delta(state.sqrt_price_x96, step_target, active_L, round_up=True)

            gross_in = (in_net * FEE_DENOM + (FEE_DENOM - pool_fee_bps) - 1) // (FEE_DENOM - pool_fee_bps)
            fee_amount = gross_in - in_net

            if input_token == 0:
                lp_fee, _ = apply_protocol_fee(fee_amount, state.fee_protocol_token0, protocol_fee_encoding)
                if active_L > 0 and lp_fee > 0:
                    state.fee_growth_global0_x128 += (lp_fee * Q128) // active_L
            else:
                lp_fee, _ = apply_protocol_fee(fee_amount, state.fee_protocol_token1, protocol_fee_encoding)
                if active_L > 0 and lp_fee > 0:
                    state.fee_growth_global1_x128 += (lp_fee * Q128) // active_L

            # Optional: direct accrual validator using pro-rata share L/L_active when in range
            if validator is not None and active_L > 0 and lp_fee > 0:
                in_range = (validator.tick_lower <= state.tick < validator.tick_upper)
                if in_range and validator.L > 0:
                    share_num = validator.L * lp_fee
                    share = share_num // active_L
                    if input_token == 0:
                        validator.accr_token0 += share
                    else:
                        validator.accr_token1 += share

            state.sqrt_price_x96 = step_target
            if state.sqrt_price_x96 == next_sqrt and nxt_tick is not None:
                info = ticks.get(nxt_tick)
                if info is not None and info.initialized:
                    # toggle outside values at boundary
                    info.fee_growth_outside0_x128 = state.fee_growth_global0_x128 - info.fee_growth_outside0_x128
                    info.fee_growth_outside1_x128 = state.fee_growth_global1_x128 - info.fee_growth_outside1_x128
                    # update active liquidity
                    if not use_swap_liquidity:
                        if zero_for_one:
                            state.liquidity_active -= info.liquidity_net
                        else:
                            state.liquidity_active += info.liquidity_net
                state.tick = nxt_tick
            else:
                state.tick = target_tick

    return state, ticks


def fee_growth_inside_delta(
    ticks_start: Dict[int, TickInfo],
    ticks_end: Dict[int, TickInfo],
    tick_lower: int,
    tick_upper: int,
    global0_x128_start: int,
    global1_x128_start: int,
    global0_x128_end: int,
    global1_x128_end: int,
    tick_at_start: int,
    tick_at_end: int,
) -> Tuple[int, int]:
    lower_start = ticks_start.get(tick_lower, TickInfo(0, 0, 0, False))
    upper_start = ticks_start.get(tick_upper, TickInfo(0, 0, 0, False))
    lower_end = ticks_end.get(tick_lower, TickInfo(0, 0, 0, False))
    upper_end = ticks_end.get(tick_upper, TickInfo(0, 0, 0, False))

    def inside(global_val, lower_out, upper_out, tick_current):
        below = lower_out if tick_current >= tick_lower else global_val - lower_out
        above = upper_out if tick_current < tick_upper else global_val - upper_out
        return global_val - below - above

    inside0_start = inside(
        global0_x128_start,
        lower_start.fee_growth_outside0_x128,
        upper_start.fee_growth_outside0_x128,
        tick_at_start,
    )
    inside0_end = inside(
        global0_x128_end,
        lower_end.fee_growth_outside0_x128,
        upper_end.fee_growth_outside0_x128,
        tick_at_end,
    )
    inside1_start = inside(
        global1_x128_start,
        lower_start.fee_growth_outside1_x128,
        upper_start.fee_growth_outside1_x128,
        tick_at_start,
    )
    inside1_end = inside(
        global1_x128_end,
        lower_end.fee_growth_outside1_x128,
        upper_end.fee_growth_outside1_x128,
        tick_at_end,
    )

    return inside0_end - inside0_start, inside1_end - inside1_start


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


def principal_amounts_at_price(L: int, sqrt_a: int, sqrt_b: int, sqrt_p: int) -> Tuple[int, int]:
    # Position principal token amounts at a given sqrt price (excluding fees)
    if sqrt_p <= sqrt_a:
        # all token0
        amt0 = get_amount0_delta(sqrt_a, sqrt_b, L, round_up=False)
        return int(amt0), 0
    elif sqrt_p >= sqrt_b:
        # all token1
        amt1 = get_amount1_delta(sqrt_a, sqrt_b, L, round_up=False)
        return 0, int(amt1)
    else:
        amt0 = get_amount0_delta(sqrt_p, sqrt_b, L, round_up=False)
        amt1 = get_amount1_delta(sqrt_a, sqrt_p, L, round_up=False)
        return int(amt0), int(amt1)


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
        symbol0=str(t0['symbol']),
        symbol1=str(t1['symbol']),
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
    parser.add_argument('--use-swap-liquidity', action='store_true', help='Use per-swap pool liquidity for fee growth (approximation when full tick snapshot is unavailable)')
    parser.add_argument('--accounting-mode', choices=['growth', 'direct'], default='growth', help='Compute fees from feeGrowthInside (growth) or direct pro-rata integral (direct)')
    parser.add_argument('--protocol-fee-encoding', choices=['base256', 'one_over'], default='base256', help='Protocol fee encoding; default base256')
    parser.add_argument('--ethusdt-csv', type=str, default=None, help='Path to ETHUSDT hourly CSV for USD conversions (uses close price)')
    args = parser.parse_args()

    base = Path('/home/poon/developments/ice-senior-project')
    cfg, init = load_pool_context(base)
    swaps, mints, burns = load_events(base)
    # Load ETHUSDT for USD conversions if provided or default path exists
    eth_usdt_df = None
    default_ethusdt = base / 'research' / 'simulation_2' / 'ETHUSDT_hourly_data_20241101_20251101.csv'
    ethusdt_path = Path(args.ethusdt_csv) if args.ethusdt_csv is not None else default_ethusdt
    try:
        if ethusdt_path.exists():
            eth_usdt_df = pd.read_csv(ethusdt_path)
            eth_usdt_df['open_time'] = pd.to_datetime(eth_usdt_df['open_time'], utc=True)
            eth_usdt_df = eth_usdt_df.sort_values('open_time').reset_index(drop=True)
    except Exception:
        eth_usdt_df = None

    # Filter window
    t0 = pd.to_datetime(args.start, utc=True)
    t1 = pd.to_datetime(args.end, utc=True)

    # Align initial pool state to on-chain state at or before t0
    try:
        # Prefer last swap strictly before t0 for price/tick
        swaps_before = swaps[swaps['evt_block_time'] < t0]
        if not swaps_before.empty:
            prev = swaps_before.sort_values(['evt_block_time', 'evt_block_number']).iloc[-1]
            init.sqrt_price_x96 = int(prev['sqrtPriceX96'])
            init.tick = int(prev['tick'])
        # Always (re)sample feeProtocol at/<= t0 from slot0 snapshots
        slot0_df = pd.read_csv(base / 'dune_pipeline' / 'slot0_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv')
        if 'call_block_time' in slot0_df.columns:
            slot0_df['call_block_time'] = pd.to_datetime(slot0_df['call_block_time'], utc=True)
            slot0_at_or_before = slot0_df[slot0_df['call_block_time'] <= t0]
            if not slot0_at_or_before.empty:
                row = slot0_at_or_before.sort_values('call_block_time').iloc[-1]
            else:
                row = slot0_df.sort_values('call_block_time').iloc[0]
            fee_proto0, fee_proto1 = decode_fee_protocol(int(row['output_feeProtocol']))
            init.fee_protocol_token0 = fee_proto0
            init.fee_protocol_token1 = fee_proto1
            # If no swap before t0, use this snapshot for sqrtPrice/tick as a fallback
            if swaps_before.empty:
                init.sqrt_price_x96 = int(row['output_sqrtPriceX96'])
                init.tick = int(row['output_tick'])
    except Exception:
        # If alignment fails for any reason, proceed with existing init
        pass
    swaps_w = swaps[(swaps['evt_block_time'] >= t0) & (swaps['evt_block_time'] <= t1)].copy()
    swaps_w = swaps_w.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
    # Build initial tick map from events up to t0
    mints_before = mints[(mints['evt_block_time'] <= t0)].copy()
    burns_before = burns[(burns['evt_block_time'] <= t0)].copy()
    # Post-start events to interleave with swaps
    mints_post = mints[(mints['evt_block_time'] > t0) & (mints['evt_block_time'] <= t1)].copy()
    burns_post = burns[(burns['evt_block_time'] > t0) & (burns['evt_block_time'] <= t1)].copy()
    mints_post = mints_post.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
    burns_post = burns_post.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)

    # Build tick map to t0 (best we can without a genesis snapshot)
    tick_map = build_liquidity_net_from_events(mints_before, burns_before)
    if args.use_swap_liquidity:
        if len(swaps_w) == 0:
            raise SystemExit('No swaps in selected window to infer liquidity; remove --use-swap-liquidity or widen the window')
        try:
            init.liquidity_active = int(swaps_w.iloc[0]['liquidity'])
        except Exception:
            raise SystemExit('Swaps dataset missing liquidity column; cannot use --use-swap-liquidity')
    else:
        init.liquidity_active = init_active_liquidity_from_ticks(init.tick, tick_map)

    # Capture start snapshot
    global0_start = init.fee_growth_global0_x128
    global1_start = init.fee_growth_global1_x128
    tick_start = init.tick

    # Snapshot ticks at start for accurate inside-growth delta
    tick_map_start = {t: TickInfo(info.liquidity_net,
                                  info.fee_growth_outside0_x128,
                                  info.fee_growth_outside1_x128,
                                  info.initialized)
                      for t, info in tick_map.items()}

    # Optional validator
    validator = None
    # Prepare range
    lower_px = Decimal(str(args.price_lower))
    upper_px = Decimal(str(args.price_upper))
    lower_tick = round_tick_to_spacing(price_to_tick(lower_px, cfg.decimals0, cfg.decimals1), cfg.tick_spacing)
    upper_tick = round_up_to_spacing(price_to_tick(upper_px, cfg.decimals0, cfg.decimals1), cfg.tick_spacing)
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
    if args.validate or args.accounting_mode == 'direct':
        validator = FeeAccrualValidator(lower_tick, upper_tick, L)

    # Initialize feeGrowthOutside for user's chosen ticks at start (mimic mint at t0)
    ls = tick_map_start.get(lower_tick)
    if ls is None:
        ls = TickInfo(liquidity_net=0, initialized=True)
        tick_map_start[lower_tick] = ls
    if not ls.initialized:
        ls.initialized = True
    # lower: if tick_start >= lower_tick -> outside = global_start else 0
    if tick_start >= lower_tick:
        ls.fee_growth_outside0_x128 = global0_start
        ls.fee_growth_outside1_x128 = global1_start
    else:
        ls.fee_growth_outside0_x128 = 0
        ls.fee_growth_outside1_x128 = 0

    us = tick_map_start.get(upper_tick)
    if us is None:
        us = TickInfo(liquidity_net=0, initialized=True)
        tick_map_start[upper_tick] = us
    if not us.initialized:
        us.initialized = True
    # upper: if tick_start < upper_tick -> outside = 0 else global_start
    if tick_start < upper_tick:
        us.fee_growth_outside0_x128 = 0
        us.fee_growth_outside1_x128 = 0
    else:
        us.fee_growth_outside0_x128 = global0_start
        us.fee_growth_outside1_x128 = global1_start

    # Run simulation
    final_state, tick_map_out = simulate_swaps(swaps_w, cfg.fee, init, tick_map, validator, mints_post, burns_post, use_swap_liquidity=args.use_swap_liquidity, protocol_fee_encoding=args.protocol_fee_encoding)

    # Fee growth inside delta over [t0, t1]
    # Ensure end snapshot contains user's ticks (if never crossed during simulation)
    if lower_tick not in tick_map_out:
        tick_map_out[lower_tick] = TickInfo(
            liquidity_net=tick_map_start[lower_tick].liquidity_net,
            fee_growth_outside0_x128=tick_map_start[lower_tick].fee_growth_outside0_x128,
            fee_growth_outside1_x128=tick_map_start[lower_tick].fee_growth_outside1_x128,
            initialized=True,
        )
    if upper_tick not in tick_map_out:
        tick_map_out[upper_tick] = TickInfo(
            liquidity_net=tick_map_start[upper_tick].liquidity_net,
            fee_growth_outside0_x128=tick_map_start[upper_tick].fee_growth_outside0_x128,
            fee_growth_outside1_x128=tick_map_start[upper_tick].fee_growth_outside1_x128,
            initialized=True,
        )

    d0, d1 = fee_growth_inside_delta(
        tick_map_start,
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
    if args.accounting_mode == 'direct':
        if validator is None:
            raise SystemExit('Direct accounting mode requires validator')
        fees0 = int(validator.accr_token0)
        fees1 = int(validator.accr_token1)
    else:
        fees0 = (L * d0) // Q128
        fees1 = (L * d1) // Q128

    # Compute principal holdings at start and end prices and valuations
    # Derive sqrt prices for bounds and current start/end
    raw_lower = lower_px * (Decimal(10) ** Decimal(cfg.decimals1 - cfg.decimals0))
    raw_upper = upper_px * (Decimal(10) ** Decimal(cfg.decimals1 - cfg.decimals0))
    sqrt_a = int((raw_lower.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    sqrt_b = int((raw_upper.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    sqrt_p_start = init.sqrt_price_x96
    sqrt_p_end = final_state.sqrt_price_x96

    start_principal0, start_principal1 = principal_amounts_at_price(L, sqrt_a, sqrt_b, sqrt_p_start)
    end_principal0, end_principal1 = principal_amounts_at_price(L, sqrt_a, sqrt_b, sqrt_p_end)

    # Prices and valuations (token1 is USD-like; convert token0 via price)
    end_price = sqrt_price_x96_to_price(final_state.sqrt_price_x96, cfg.decimals0, cfg.decimals1)
    start_raw = start_price * (Decimal(10) ** Decimal(cfg.decimals1 - cfg.decimals0))
    end_raw = end_price * (Decimal(10) ** Decimal(cfg.decimals1 - cfg.decimals0))

    def to_token1_units(amt0_wei: int, amt1_wei: int, raw_price: Decimal) -> Decimal:
        return Decimal(amt1_wei) + (Decimal(amt0_wei) * raw_price)

    start_value_token1_units = to_token1_units(start_principal0, start_principal1, start_raw)
    end_value_principal_token1_units = to_token1_units(end_principal0, end_principal1, end_raw)
    end_value_total_token1_units = to_token1_units(end_principal0 + fees0, end_principal1 + fees1, end_raw)
    # HODL baseline: hold start principal tokens to end price
    hodl_end_value_token1_units = to_token1_units(start_principal0, start_principal1, end_raw)
    il_token1_units = end_value_principal_token1_units - hodl_end_value_token1_units
    il_pct = (Decimal(0) if hodl_end_value_token1_units == 0 else (Decimal(il_token1_units) / Decimal(hodl_end_value_token1_units)))
    total_pnl_token1_units = end_value_total_token1_units - start_value_token1_units
    total_pnl_pct = (Decimal(0) if start_value_token1_units == 0 else (Decimal(total_pnl_token1_units) / Decimal(start_value_token1_units)))

    # Impermanent loss in USD (token1 units) and percent
    il_usd = float(Decimal(il_token1_units) / Decimal(10 ** cfg.decimals1))
    hodl_end_value_usd = float(Decimal(hodl_end_value_token1_units) / Decimal(10 ** cfg.decimals1))
    end_principal_value_usd = float(Decimal(end_value_principal_token1_units) / Decimal(10 ** cfg.decimals1))

    # If ETHUSDT prices are available, print ETH and USD conversions using hourly close
    def get_eth_usd_at(ts):
        if eth_usdt_df is None:
            return None
        sub = eth_usdt_df[eth_usdt_df['open_time'] <= ts]
        if len(sub) == 0:
            return None
        return Decimal(str(sub.iloc[-1]['close']))

    price_start_usd = get_eth_usd_at(t0)
    price_end_usd = get_eth_usd_at(t1)

    # Minimal USD-centric summary
    dep0_eth = Decimal(amount0) / Decimal(10 ** cfg.decimals0)
    dep1_usd = Decimal(amount1) / Decimal(10 ** cfg.decimals1)
    fees0_eth = Decimal(fees0) / Decimal(10 ** cfg.decimals0)
    fees1_usd = Decimal(fees1) / Decimal(10 ** cfg.decimals1)
    end_tot0_eth = Decimal(end_principal0 + fees0) / Decimal(10 ** cfg.decimals0)
    end_tot1_usd = Decimal(end_principal1 + fees1) / Decimal(10 ** cfg.decimals1)

    # Principal-only token units at start and end (exclude fees)
    start_principal0_eth = Decimal(start_principal0) / Decimal(10 ** cfg.decimals0)
    start_principal1_token1 = Decimal(start_principal1) / Decimal(10 ** cfg.decimals1)
    end_principal0_eth_units = Decimal(end_principal0) / Decimal(10 ** cfg.decimals0)
    end_principal1_token1 = Decimal(end_principal1) / Decimal(10 ** cfg.decimals1)

    deposit_token0_usd = float(dep0_eth * price_start_usd) if price_start_usd is not None else None
    # token1 is USD-like (USDT); compute regardless of ETH price availability
    deposit_token1_usd = float(dep1_usd)
    fees_token0_usd = float(fees0_eth * price_end_usd) if price_end_usd is not None else None
    fees_token1_usd = float(fees1_usd)
    end_token0_usd = float(end_tot0_eth * price_end_usd) if price_end_usd is not None else None
    end_token1_usd = float(end_tot1_usd)

    # Principal-only USD valuations
    start_principal_token0_usd = float(start_principal0_eth * price_start_usd) if price_start_usd is not None else None
    start_principal_token1_usd = float(start_principal1_token1)
    end_principal_token0_usd = float(end_principal0_eth_units * price_end_usd) if price_end_usd is not None else None
    end_principal_token1_usd = float(end_principal1_token1)

    summary = {
        'pool': cfg.pool,
        'fee_bps': cfg.fee,
        'tokens': {
            'token0': {'address': cfg.token0, 'symbol': cfg.symbol0, 'decimals': cfg.decimals0},
            'token1': {'address': cfg.token1, 'symbol': cfg.symbol1, 'decimals': cfg.decimals1},
        },
        'window_utc': {'start': t0.isoformat(), 'end': t1.isoformat()},
        'position_ticks': {'lower': int(lower_tick), 'upper': int(upper_tick)},
        'price': {
            'start': float(start_price),
            'end': float(end_price),
        },
        'start_tokens': {
            'token0_amount': float(dep0_eth),
            'token1_amount': float(dep1_usd),
            'token0_usd': deposit_token0_usd,
            'token1_usd': deposit_token1_usd,
            'total_usd': (deposit_token0_usd if deposit_token0_usd is not None else 0.0) + deposit_token1_usd,
        },
        'start_principal_tokens': {
            'token0_amount': float(start_principal0_eth),
            'token1_amount': float(start_principal1_token1),
            'token0_usd': start_principal_token0_usd,
            'token1_usd': start_principal_token1_usd,
            'total_usd': (start_principal_token0_usd if start_principal_token0_usd is not None else 0.0) + start_principal_token1_usd,
        },
        'end_tokens': {
            'token0_amount': float(end_tot0_eth),
            'token1_amount': float(end_tot1_usd),
            'token0_usd': end_token0_usd,
            'token1_usd': end_token1_usd,
            'total_usd': (end_token0_usd if end_token0_usd is not None else 0.0) + end_token1_usd,
        },
        'end_principal_tokens': {
            'token0_amount': float(end_principal0_eth_units),
            'token1_amount': float(end_principal1_token1),
            'token0_usd': end_principal_token0_usd,
            'token1_usd': end_principal_token1_usd,
            'total_usd': (end_principal_token0_usd if end_principal_token0_usd is not None else 0.0) + end_principal_token1_usd,
        },
        'fees_usd': {'token0': fees_token0_usd, 'token1': fees_token1_usd},
        'impermanent_loss': {
            'usd': il_usd,
            'pct': float(il_pct),
        },
        'valuation_baselines': {
            'hodl_end_usd': hodl_end_value_usd,
            'end_principal_usd': end_principal_value_usd,
        },
    }
    _json = json.dumps(summary, indent=4)
    # Replace 4 spaces with a tab for tab-indented output
    _json = _json.replace('    ', '\t')
    print(_json)


if __name__ == '__main__':
    main()


