from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, getcontext
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from bisect import bisect_left, bisect_right, insort
from datetime import timedelta # <-- NEW REQUIRED IMPORT

# High precision for tick/price math
getcontext().prec = 80

Q96 = 2 ** 96
Q128 = 2 ** 128
FEE_DENOM = 1_000_000  # Uniswap v3 fee in hundredths of a bip


# -----------------------------
# Public structured return types
# -----------------------------

@dataclass
class TokenMetaOut:
    address: str
    symbol: str
    decimals: int


@dataclass
class TokensOut:
    token0: TokenMetaOut
    token1: TokenMetaOut


@dataclass
class WindowOut:
    start: str  # ISO UTC
    end: str    # ISO UTC


@dataclass
class PositionTicksOut:
    lower: int
    upper: int


@dataclass
class PriceOut:
    start: float
    end: float


@dataclass
class TokenAmountsOut:
    token0_amount: float
    token1_amount: float
    token0_usd: float | None
    token1_usd: float
    total_usd: float


@dataclass
class FeesUsdOut:
    token0: float | None
    token1: float


@dataclass
class ImpermanentLossOut:
    usd: float
    pct: float


@dataclass
class BaselinesOut:
    hodl_end_usd: float
    end_principal_usd: float


@dataclass
class SimulationSummaryOut:
    pool: str
    fee_bps: int
    tokens: TokensOut
    window_utc: WindowOut
    position_ticks: PositionTicksOut
    price: PriceOut
    start_tokens: TokenAmountsOut
    start_principal_tokens: TokenAmountsOut
    end_tokens: TokenAmountsOut
    end_principal_tokens: TokenAmountsOut
    fees_usd: FeesUsdOut
    impermanent_loss: ImpermanentLossOut
    valuation_baselines: BaselinesOut

    def to_dict(self) -> dict:
        return asdict(self)


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
    p = Decimal(sqrt_price_x96) / Decimal(Q96)
    return (p * p) * (Decimal(10) ** Decimal(decimals0 - decimals1))


def price_to_tick(price_token1_per_token0: Decimal, decimals0: int, decimals1: int) -> int:
    if price_token1_per_token0 <= 0:
        raise ValueError('price must be positive')
    price_raw = price_token1_per_token0 * (Decimal(10) ** Decimal(decimals1 - decimals0))
    return int((price_raw.ln() / Decimal('1.0001').ln()).to_integral_value(rounding='ROUND_FLOOR'))


def round_tick_to_spacing(tick: int, spacing: int) -> int:
    if tick >= 0:
        return tick - (tick % spacing)
    mod = (-tick) % spacing
    return tick - (spacing - mod) if mod != 0 else tick


def round_up_to_spacing(tick: int, spacing: int) -> int:
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
        protocol_cut = fee_amount // int(proto)
        lp_fee = fee_amount - protocol_cut
        return lp_fee, protocol_cut
    else:
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

    initialized_ticks_sorted = sorted([t for t, info in ticks.items() if info.initialized])

    def ensure_tick_initialized(tick_key: int):
        info = ticks.get(tick_key)
        if info is None:
            info = TickInfo(liquidity_net=0, initialized=True)
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
        if mints_post is not None or burns_post is not None:
            swap_time = s['evt_block_time']
            swap_bn = int(s['evt_block_number'])

            while True:
                did_one = False
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
                nxt_tick = next_initialized_left(state.tick)
                next_sqrt = tick_to_sqrt_price_x96(nxt_tick) if nxt_tick is not None else target_sqrt
                step_target = max(next_sqrt, target_sqrt)
                in_net = get_amount0_delta(step_target, state.sqrt_price_x96, active_L, round_up=True)
            else:
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
                    info.fee_growth_outside0_x128 = state.fee_growth_global0_x128 - info.fee_growth_outside0_x128
                    info.fee_growth_outside1_x128 = state.fee_growth_global1_x128 - info.fee_growth_outside1_x128
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
    raw_lower = price_lower * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_upper = price_upper * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_current = price_current * (Decimal(10) ** Decimal(decimals1 - decimals0))
    a = int((raw_lower.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    b = int((raw_upper.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    p = int((raw_current.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    if amount0 is None and amount1 is None:
        L = 10 ** 18
    elif amount0 is not None and amount1 is None:
        L = (int(amount0) * (a * b // Q96)) // (b - a)
    elif amount1 is not None and amount0 is None:
        L = (int(amount1) * Q96) // (b - a)
    else:
        L = 10 ** 18
    amt0 = get_amount0_delta(p if p < b else a, b, L, round_up=True) if p < b else 0
    amt1 = get_amount1_delta(a, p if p > a else b, L, round_up=True) if p > a else 0
    return L, int(amt0), int(amt1)


def compute_L_for_budget_usd(price_lower: Decimal, price_upper: Decimal, price_current: Decimal,
                             total_usd: Decimal, decimals0: int, decimals1: int) -> Tuple[int, int, int]:
    raw_lower = price_lower * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_upper = price_upper * (Decimal(10) ** Decimal(decimals1 - decimals0))
    raw_current = price_current * (Decimal(10) ** Decimal(decimals1 - decimals0))
    a = int((raw_lower.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    b = int((raw_upper.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
    p = int((raw_current.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))

    if p <= a:
        num = (b - a) << 96
        denom = b * a
        amt0_per_L = Decimal(num) / Decimal(denom)
        k = raw_current * amt0_per_L
    elif p >= b:
        amt1_per_L = Decimal(b - a) / Decimal(Q96)
        k = amt1_per_L
    else:
        amt0_per_L = Decimal(((b - p) << 96)) / Decimal(b * p)
        amt1_per_L = Decimal(p - a) / Decimal(Q96)
        k = amt1_per_L + raw_current * amt0_per_L

    total_usd_units = total_usd * (Decimal(10) ** Decimal(decimals1))
    if k == 0:
        L = 0
    else:
        L = int((total_usd_units / k).to_integral_value(rounding='ROUND_FLOOR'))

    amt0 = get_amount0_delta(p if p < b else a, b, L, round_up=True) if p < b else 0
    amt1 = get_amount1_delta(a, p if p > a else b, L, round_up=True) if p > a else 0
    return L, int(amt0), int(amt1)


def principal_amounts_at_price(L: int, sqrt_a: int, sqrt_b: int, sqrt_p: int) -> Tuple[int, int]:
    if sqrt_p <= sqrt_a:
        amt0 = get_amount0_delta(sqrt_a, sqrt_b, L, round_up=False)
        return int(amt0), 0
    elif sqrt_p >= sqrt_b:
        amt1 = get_amount1_delta(sqrt_a, sqrt_b, L, round_up=False)
        return 0, int(amt1)
    else:
        amt0 = get_amount0_delta(sqrt_p, sqrt_b, L, round_up=False)
        amt1 = get_amount1_delta(sqrt_a, sqrt_p, L, round_up=False)
        return int(amt0), int(amt1)


class UniswapV3FeeSimulator:
    """
    Uniswap v3 fee simulator with structured outputs.

    Usage:
        sim = UniswapV3FeeSimulator(pool_cfg, tokens, slot0, swaps, mints, burns, eth_usdt_prices)
        result = sim.simulate(
            price_lower=4200,
            price_upper=4400,
            start='2025-09-01T00:00:00Z',
            end='2025-09-13T06:00:00Z',
            total_usd=1000,
            accounting_mode='direct',
            use_swap_liquidity=True,
        )

        # Access fields:
        result.fees_usd.token1
        result.end_principal_tokens.total_usd

        # Convert to dict/json:
        as_dict = result.to_dict()
    """
    def __init__(self,
                 pool_cfg: pd.DataFrame,
                 tokens: pd.DataFrame,
                 slot0: pd.DataFrame,
                 swaps: pd.DataFrame,
                 mints: pd.DataFrame,
                 burns: pd.DataFrame,
                 eth_usdt_prices: Optional[pd.DataFrame] = None):
        tokens = tokens.copy()
        tokens['contract_address'] = tokens['contract_address'].str.lower()
        t0 = tokens.set_index('contract_address').loc[pool_cfg.loc[0, 'token0'].lower()]
        t1 = tokens.set_index('contract_address').loc[pool_cfg.loc[0, 'token1'].lower()]
        self.cfg = PoolConfig(
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
        self.slot0 = slot0.copy()
        if 'call_block_time' in self.slot0.columns:
            self.slot0['call_block_time'] = pd.to_datetime(self.slot0['call_block_time'], utc=True)
        self.swaps = swaps.copy()
        self.swaps['evt_block_time'] = pd.to_datetime(self.swaps['evt_block_time'], utc=True)
        self.swaps = self.swaps.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
        self.mints = mints.copy()
        self.mints['evt_block_time'] = pd.to_datetime(self.mints['evt_block_time'], utc=True)
        self.mints = self.mints.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
        self.burns = burns.copy()
        self.burns['evt_block_time'] = pd.to_datetime(self.burns['evt_block_time'], utc=True)
        self.burns = self.burns.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
        self.eth_usdt = None
        if eth_usdt_prices is not None:
            eth_usdt_prices = eth_usdt_prices.copy()
            eth_usdt_prices['open_time'] = pd.to_datetime(eth_usdt_prices['open_time'], utc=True)
            self.eth_usdt = eth_usdt_prices.sort_values('open_time').reset_index(drop=True)

    def _init_state_at_t0(self, t0_time: pd.Timestamp) -> PoolState:
        # Start from earliest slot0 but will realign
        first = self.slot0.sort_values('call_block_time').iloc[0]
        fee_proto0, fee_proto1 = decode_fee_protocol(int(first['output_feeProtocol']))
        init = PoolState(
            sqrt_price_x96=int(first['output_sqrtPriceX96']),
            tick=int(first['output_tick']),
            liquidity_active=0,
            fee_protocol_token0=fee_proto0,
            fee_protocol_token1=fee_proto1,
        )
        # align
        swaps_before = self.swaps[self.swaps['evt_block_time'] < t0_time]
        if not swaps_before.empty:
            prev = swaps_before.iloc[-1]
            init.sqrt_price_x96 = int(prev['sqrtPriceX96'])
            init.tick = int(prev['tick'])
        slot0_at_or_before = self.slot0[self.slot0['call_block_time'] <= t0_time]
        if not slot0_at_or_before.empty:
            row = slot0_at_or_before.sort_values('call_block_time').iloc[-1]
        else:
            row = self.slot0.sort_values('call_block_time').iloc[0]
        fp0, fp1 = decode_fee_protocol(int(row['output_feeProtocol']))
        init.fee_protocol_token0 = fp0
        init.fee_protocol_token1 = fp1
        if swaps_before.empty:
            init.sqrt_price_x96 = int(row['output_sqrtPriceX96'])
            init.tick = int(row['output_tick'])
        return init

    def _get_eth_usd_at(self, ts: pd.Timestamp) -> Optional[Decimal]:
        if self.eth_usdt is None:
            return None
        sub = self.eth_usdt[self.eth_usdt['open_time'] <= ts]
        if len(sub) == 0:
            return None
        return Decimal(str(sub.iloc[-1]['close']))

    def simulate(self,
                 price_lower: float,
                 price_upper: float,
                 start: str,
                 end: str,
                 liquidity: Optional[int] = None,
                 total_usd: Optional[float] = None,
                 validate: bool = False,
                 use_swap_liquidity: bool = False,
                 accounting_mode: str = 'growth',
                 protocol_fee_encoding: str = 'base256') -> SimulationSummaryOut:
        """
        Run the fee simulation over [start, end] UTC for the given price range.

        Returns a structured SimulationSummaryOut with nested fields.
        """
        t0 = pd.to_datetime(start, utc=True)
        t1 = pd.to_datetime(end, utc=True)

        init = self._init_state_at_t0(t0)
        swaps_w = self.swaps[(self.swaps['evt_block_time'] >= t0) & (self.swaps['evt_block_time'] <= t1)].copy()
        swaps_w = swaps_w.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)

        mints_before = self.mints[(self.mints['evt_block_time'] <= t0)].copy()
        burns_before = self.burns[(self.burns['evt_block_time'] <= t0)].copy()
        mints_post = self.mints[(self.mints['evt_block_time'] > t0) & (self.mints['evt_block_time'] <= t1)].copy()
        burns_post = self.burns[(self.burns['evt_block_time'] > t0) & (self.burns['evt_block_time'] <= t1)].copy()
        mints_post = mints_post.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)
        burns_post = burns_post.sort_values(['evt_block_time', 'evt_block_number']).reset_index(drop=True)

        tick_map = build_liquidity_net_from_events(mints_before, burns_before)
        if use_swap_liquidity:
            if len(swaps_w) == 0:
                raise SystemExit('No swaps in selected window to infer liquidity; remove use_swap_liquidity or widen the window')
            try:
                init.liquidity_active = int(swaps_w.iloc[0]['liquidity'])
            except Exception:
                raise SystemExit('Swaps dataset missing liquidity column; cannot use use_swap_liquidity')
        else:
            init.liquidity_active = init_active_liquidity_from_ticks(init.tick, tick_map)

        global0_start = init.fee_growth_global0_x128
        global1_start = init.fee_growth_global1_x128
        tick_start = init.tick

        tick_map_start = {t: TickInfo(info.liquidity_net,
                                      info.fee_growth_outside0_x128,
                                      info.fee_growth_outside1_x128,
                                      info.initialized)
                          for t, info in tick_map.items()}

        lower_px = Decimal(str(price_lower))
        upper_px = Decimal(str(price_upper))
        lower_tick = round_tick_to_spacing(price_to_tick(lower_px, self.cfg.decimals0, self.cfg.decimals1), self.cfg.tick_spacing)
        upper_tick = round_up_to_spacing(price_to_tick(upper_px, self.cfg.decimals0, self.cfg.decimals1), self.cfg.tick_spacing)
        if lower_tick >= upper_tick:
            raise ValueError('lower_tick must be < upper_tick')
        start_price = sqrt_price_x96_to_price(init.sqrt_price_x96, self.cfg.decimals0, self.cfg.decimals1)
        if liquidity is not None:
            L, amount0, amount1 = compute_L_from_deposit(lower_px, upper_px, start_price,
                                                         amount0=None, amount1=None,
                                                         decimals0=self.cfg.decimals0, decimals1=self.cfg.decimals1)
            L = int(liquidity)
            a = Decimal(price_lower) * (Decimal(10) ** Decimal(self.cfg.decimals1 - self.cfg.decimals0))
            b = Decimal(price_upper) * (Decimal(10) ** Decimal(self.cfg.decimals1 - self.cfg.decimals0))
            p = start_price * (Decimal(10) ** Decimal(self.cfg.decimals1 - self.cfg.decimals0))
            sqrt_a = int((a.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
            sqrt_b = int((b.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
            sqrt_p = int((p.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
            amount0 = get_amount0_delta(sqrt_p if sqrt_p < sqrt_b else sqrt_a, sqrt_b, L, round_up=True) if sqrt_p < sqrt_b else 0
            amount1 = get_amount1_delta(sqrt_a, sqrt_p if sqrt_p > sqrt_a else sqrt_b, L, round_up=True) if sqrt_p > sqrt_a else 0
        else:
            if total_usd is None:
                raise SystemExit('Either liquidity or total_usd must be provided')
            L, amount0, amount1 = compute_L_for_budget_usd(lower_px, upper_px, start_price,
                                                           Decimal(str(total_usd)),
                                                           self.cfg.decimals0, self.cfg.decimals1)

        validator = FeeAccrualValidator(lower_tick, upper_tick, L) if (validate or accounting_mode == 'direct') else None

        ls = tick_map_start.get(lower_tick)
        if ls is None:
            ls = TickInfo(liquidity_net=0, initialized=True)
            tick_map_start[lower_tick] = ls
        if not ls.initialized:
            ls.initialized = True
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
        if tick_start < upper_tick:
            us.fee_growth_outside0_x128 = 0
            us.fee_growth_outside1_x128 = 0
        else:
            us.fee_growth_outside0_x128 = global0_start
            us.fee_growth_outside1_x128 = global1_start

        final_state, tick_map_out = simulate_swaps(swaps_w, self.cfg.fee, init, tick_map, validator, mints_post, burns_post, use_swap_liquidity=use_swap_liquidity, protocol_fee_encoding=protocol_fee_encoding)

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

        if accounting_mode == 'direct':
            if validator is None:
                raise SystemExit('Direct accounting mode requires validator')
            fees0 = int(validator.accr_token0)
            fees1 = int(validator.accr_token1)
        else:
            fees0 = (L * d0) // Q128
            fees1 = (L * d1) // Q128

        raw_lower = lower_px * (Decimal(10) ** Decimal(self.cfg.decimals1 - self.cfg.decimals0))
        raw_upper = upper_px * (Decimal(10) ** Decimal(self.cfg.decimals1 - self.cfg.decimals0))
        sqrt_a = int((raw_lower.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
        sqrt_b = int((raw_upper.sqrt() * Q96).to_integral_value(rounding='ROUND_FLOOR'))
        sqrt_p_start = init.sqrt_price_x96
        sqrt_p_end = final_state.sqrt_price_x96

        start_principal0, start_principal1 = principal_amounts_at_price(L, sqrt_a, sqrt_b, sqrt_p_start)
        end_principal0, end_principal1 = principal_amounts_at_price(L, sqrt_a, sqrt_b, sqrt_p_end)

        end_price = sqrt_price_x96_to_price(final_state.sqrt_price_x96, self.cfg.decimals0, self.cfg.decimals1)
        start_raw = start_price * (Decimal(10) ** Decimal(self.cfg.decimals1 - self.cfg.decimals0))
        end_raw = end_price * (Decimal(10) ** Decimal(self.cfg.decimals1 - self.cfg.decimals0))

        def to_token1_units(amt0_wei: int, amt1_wei: int, raw_price: Decimal) -> Decimal:
            return Decimal(amt1_wei) + (Decimal(amt0_wei) * raw_price)

        start_value_token1_units = to_token1_units(start_principal0, start_principal1, start_raw)
        end_value_principal_token1_units = to_token1_units(end_principal0, end_principal1, end_raw)
        end_value_total_token1_units = to_token1_units(end_principal0 + fees0, end_principal1 + fees1, end_raw)
        hodl_end_value_token1_units = to_token1_units(start_principal0, start_principal1, end_raw)
        il_token1_units = end_value_principal_token1_units - hodl_end_value_token1_units
        il_pct = (Decimal(0) if hodl_end_value_token1_units == 0 else (Decimal(il_token1_units) / Decimal(hodl_end_value_token1_units)))

        price_start_usd = self._get_eth_usd_at(t0)
        price_end_usd = self._get_eth_usd_at(t1)

        dep0_eth = Decimal(amount0) / Decimal(10 ** self.cfg.decimals0)
        dep1_usd = Decimal(amount1) / Decimal(10 ** self.cfg.decimals1)
        fees0_eth = Decimal(fees0) / Decimal(10 ** self.cfg.decimals0)
        fees1_usd = Decimal(fees1) / Decimal(10 ** self.cfg.decimals1)
        end_tot0_eth = Decimal(end_principal0 + fees0) / Decimal(10 ** self.cfg.decimals0)
        end_tot1_usd = Decimal(end_principal1 + fees1) / Decimal(10 ** self.cfg.decimals1)

        start_principal0_eth = Decimal(start_principal0) / Decimal(10 ** self.cfg.decimals0)
        start_principal1_token1 = Decimal(start_principal1) / Decimal(10 ** self.cfg.decimals1)
        end_principal0_eth_units = Decimal(end_principal0) / Decimal(10 ** self.cfg.decimals0)
        end_principal1_token1 = Decimal(end_principal1) / Decimal(10 ** self.cfg.decimals1)

        deposit_token0_usd = float(dep0_eth * price_start_usd) if price_start_usd is not None else None
        deposit_token1_usd = float(dep1_usd)
        fees_token0_usd = float(fees0_eth * price_end_usd) if price_end_usd is not None else None
        fees_token1_usd = float(fees1_usd)
        end_token0_usd = float(end_tot0_eth * price_end_usd) if price_end_usd is not None else None
        end_token1_usd = float(end_tot1_usd)

        start_principal_token0_usd = float(start_principal0_eth * price_start_usd) if price_start_usd is not None else None
        start_principal_token1_usd = float(start_principal1_token1)
        end_principal_token0_usd = float(end_principal0_eth_units * price_end_usd) if price_end_usd is not None else None
        end_principal_token1_usd = float(end_principal1_token1)

        il_usd = float(Decimal(il_token1_units) / Decimal(10 ** self.cfg.decimals1))
        hodl_end_value_usd = float(Decimal(hodl_end_value_token1_units) / Decimal(10 ** self.cfg.decimals1))
        end_principal_value_usd = float(Decimal(end_value_principal_token1_units) / Decimal(10 ** self.cfg.decimals1))

        return SimulationSummaryOut(
            pool=self.cfg.pool,
            fee_bps=self.cfg.fee,
            tokens=TokensOut(
                token0=TokenMetaOut(address=self.cfg.token0, symbol=self.cfg.symbol0, decimals=self.cfg.decimals0),
                token1=TokenMetaOut(address=self.cfg.token1, symbol=self.cfg.symbol1, decimals=self.cfg.decimals1),
            ),
            window_utc=WindowOut(start=t0.isoformat(), end=t1.isoformat()),
            position_ticks=PositionTicksOut(lower=int(lower_tick), upper=int(upper_tick)),
            price=PriceOut(start=float(start_price), end=float(end_price)),
            start_tokens=TokenAmountsOut(
                token0_amount=float(dep0_eth),
                token1_amount=float(dep1_usd),
                token0_usd=deposit_token0_usd,
                token1_usd=float(deposit_token1_usd),
                total_usd=(deposit_token0_usd if deposit_token0_usd is not None else 0.0) + float(deposit_token1_usd),
            ),
            start_principal_tokens=TokenAmountsOut(
                token0_amount=float(start_principal0_eth),
                token1_amount=float(start_principal1_token1),
                token0_usd=start_principal_token0_usd,
                token1_usd=float(start_principal1_token1),
                total_usd=(start_principal_token0_usd if start_principal_token0_usd is not None else 0.0) + float(start_principal1_token1),
            ),
            end_tokens=TokenAmountsOut(
                token0_amount=float(end_tot0_eth),
                token1_amount=float(end_tot1_usd),
                token0_usd=end_token0_usd,
                token1_usd=float(end_tot1_usd),
                total_usd=(end_token0_usd if end_token0_usd is not None else 0.0) + float(end_tot1_usd),
            ),
            end_principal_tokens=TokenAmountsOut(
                token0_amount=float(end_principal0_eth_units),
                token1_amount=float(end_principal1_token1),
                token0_usd=end_principal_token0_usd,
                token1_usd=float(end_principal1_token1),
                total_usd=(end_principal_token0_usd if end_principal_token0_usd is not None else 0.0) + float(end_principal1_token1),
            ),
            fees_usd=FeesUsdOut(token0=fees_token0_usd, token1=float(fees_token1_usd)),
            impermanent_loss=ImpermanentLossOut(usd=float(il_usd), pct=float(il_pct)),
            valuation_baselines=BaselinesOut(hodl_end_usd=float(hodl_end_value_usd), end_principal_usd=float(end_principal_value_usd)),
        )

# -----------------------------------------------------------
    # FINAL METHOD: Adaptive Volatility Rebalancing Strategy (Time Step in Hours)
    # -----------------------------------------------------------
    def run_rebalancing_strategy_by_time_step(
        self,
        start_time_iso: str,
        end_time_iso: str,
        total_usd_budget: float,
        rebalance_fee_usd: float,
        time_step_hours: int = 24,           # Time step is now in HOURS
        volatility_lookback_hours: int = 24, # How far back to calculate volatility
        base_tick_width: int = 100,          # The *minimum* width for your range
        volatility_multiplier: float = 1.5   # How much to scale the range by volatility
    ) -> tuple[dict, list]:
        
        # Setup
        t_start_total = pd.to_datetime(start_time_iso, utc=True)
        t_end_total = pd.to_datetime(end_time_iso, utc=True)
        
        # Ensure base width is a multiple of tick spacing
        TICK_SPACING = self.cfg.tick_spacing
        base_tick_width = max(TICK_SPACING, (base_tick_width // TICK_SPACING) * TICK_SPACING)

        # Accumulators
        total_fees_earned_usd = 0.0
        total_il_usd = 0.0  # accum realized IL
        total_rebalances = 0
        total_rebalance_cost_usd = 0.0
        position_history = [] # Initialize list to store range data for plotting

        current_t_start = t_start_total
        
        # Get the initial price state to set the first range
        init_state = self._init_state_at_t0(t_start_total)
        current_tick = init_state.tick
        
        # Main Loop
        while current_t_start < t_end_total:
            
            # 1. Calculate Adaptive Range based on Volatility
            t_lookback_start = current_t_start - timedelta(hours=volatility_lookback_hours)
            swaps_in_lookback = self.swaps[
                (self.swaps['evt_block_time'] >= t_lookback_start) & 
                (self.swaps['evt_block_time'] < current_t_start)
            ]
            
            vol_std_px = Decimal(0)
            if len(swaps_in_lookback) > 2:
                # FIX: Convert price to float for standard deviation calculation (avoids TypeError)
                prices_float = swaps_in_lookback['sqrtPriceX96'].apply(
                    lambda x: float(sqrt_price_x96_to_price(int(x), self.cfg.decimals0, self.cfg.decimals1))
                )
                
                vol_std_px_float = prices_float.std()
                
                if pd.isna(vol_std_px_float):
                    vol_std_px = Decimal(0)
                else:
                    # Convert the float std back to Decimal for subsequent precise math
                    vol_std_px = Decimal(str(vol_std_px_float))

            # Get current price
            price_current_px = sqrt_price_x96_to_price(tick_to_sqrt_price_x96(current_tick), self.cfg.decimals0, self.cfg.decimals1)

            # Define price range based on volatility (e.g., +/- 1.5 std dev)
            price_low_vol = price_current_px - (vol_std_px * Decimal(str(volatility_multiplier)))
            price_high_vol = price_current_px + (vol_std_px * Decimal(str(volatility_multiplier)))
            
            # Handle negative/zero price
            price_low_vol = max(Decimal('1e-18'), price_low_vol)

            # Convert the volatility-based price range into a tick width
            try:
                tick_low_vol = price_to_tick(price_low_vol, self.cfg.decimals0, self.cfg.decimals1)
                tick_high_vol = price_to_tick(price_high_vol, self.cfg.decimals0, self.cfg.decimals1)
                tick_volatility_width = tick_high_vol - tick_low_vol
            except ValueError:
                tick_volatility_width = 0

            # Combine base width with volatility width
            dynamic_tick_width = base_tick_width + tick_volatility_width
            
            # Ensure final width is valid and a multiple of tick spacing
            final_dynamic_width = max(TICK_SPACING, (dynamic_tick_width // TICK_SPACING) * TICK_SPACING)
            HALF_RANGE_TICKS = final_dynamic_width // 2

            # 2. Define the Range and Segment Time
            
            # The new range is centered around the current price/tick
            tick_center = current_tick
            tick_lower = tick_center - HALF_RANGE_TICKS
            tick_upper = tick_center + HALF_RANGE_TICKS
            
            # Calculate the price boundaries for the simulation call
            price_lower_px = sqrt_price_x96_to_price(tick_to_sqrt_price_x96(tick_lower), self.cfg.decimals0, self.cfg.decimals1)
            price_upper_px = sqrt_price_x96_to_price(tick_to_sqrt_price_x96(tick_upper), self.cfg.decimals0, self.cfg.decimals1)
            
            # Determine the segment's end time (e.g., 24 hours, or total end)
            current_t_end = current_t_start + timedelta(hours=time_step_hours) # <--- Using hours
            if current_t_end > t_end_total:
                current_t_end = t_end_total
            
            # 3. Run the Simulation for the segment
            segment_result = None
            try:
                segment_result = self.simulate(
                    price_lower=float(price_lower_px),
                    price_upper=float(price_upper_px),
                    start=current_t_start.isoformat(),
                    end=current_t_end.isoformat(),
                    total_usd=total_usd_budget,
                    validate=False,
                    use_swap_liquidity=False,
                    accounting_mode='growth',
                    protocol_fee_encoding='base256',
                )
            except SystemExit:
                # Record the ATTEMPTED position if simulation fails
                position_history.append({
                    'start': current_t_start,
                    'end': current_t_end,
                    'price_lower': float(price_lower_px),
                    'price_upper': float(price_upper_px),
                    'rebalance': False
                })
                current_t_start = current_t_end 
                continue

            # 4. Accumulate Fees and Check Rebalance Condition
            fees_usd = segment_result.fees_usd.token0 + segment_result.fees_usd.token1
            total_fees_earned_usd += fees_usd
            total_il_usd += segment_result.impermanent_loss.usd
            
            price_at_end = segment_result.price.end
            
            rebalance_needed = False
            # Check if the final price of the segment exited the range
            if price_at_end < float(price_lower_px) or price_at_end > float(price_upper_px):
                rebalance_needed = True

            # 5. Update State and Move to Next Segment

            rebalance_needed_final = rebalance_needed and current_t_start < t_end_total
            
            # Record the COMPLETED position for plotting
            position_history.append({
                'start': current_t_start,
                'end': current_t_end,
                'price_lower': float(price_lower_px),
                'price_upper': float(price_upper_px),
                'rebalance': rebalance_needed_final,
                'fees_usd': fees_usd,
                'il_usd': segment_result.impermanent_loss.usd,
                'price_start': segment_result.price.start,
                'price_end': segment_result.price.end,
            })
            
            current_t_start = current_t_end
            
            if rebalance_needed_final:
                # Rebalance action occurred
                total_rebalances += 1
                total_rebalance_cost_usd += rebalance_fee_usd
                
                # Recalculate the new center tick based on the final price
                price_end_px_decimal = Decimal(str(price_at_end))
                current_tick = price_to_tick(price_end_px_decimal, self.cfg.decimals0, self.cfg.decimals1)
                
        # Final Result
        # Net strategy gain = Fees + Realized IL (negative) - Gas Costs
        return {
            'total_fees_earned_usd': total_fees_earned_usd,
            'total_il_usd': total_il_usd,
            'total_rebalances': total_rebalances,
            'total_rebalance_cost_usd': total_rebalance_cost_usd,
            'net_strategy_gain_usd': total_fees_earned_usd + total_il_usd - total_rebalance_cost_usd
        }, position_history # Return both the summary and the history
    
    # -----------------------------------------------------------
    # FINAL METHOD: Simple Predictive ML (Regression) Strategy
    # -----------------------------------------------------------
    def run_predictive_ml_strategy(
        self,
        start_time_iso: str,
        end_time_iso: str,
        total_usd_budget: float,
        rebalance_fee_usd: float,
        time_step_hours: int = 24, # Fixed prediction horizon/rebalance interval
        model_file: str = 'predictive_models.pkl' # File containing the trained models
    ) -> tuple[dict, list]:
        
        import pickle
        from datetime import timedelta
        
        # Load the trained models
        try:
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
            model_min = models['model_min']
            model_max = models['model_max']
            D0 = models.get('D0', self.cfg.decimals0)
            D1 = models.get('D1', self.cfg.decimals1)
        except Exception as e:
            raise FileNotFoundError(f"Could not load trained models from {model_file}. Ensure training was successful. Error: {e}")
        
        # Setup
        t_start_total = pd.to_datetime(start_time_iso, utc=True)
        t_end_total = pd.to_datetime(end_time_iso, utc=True)

        # Accumulators
        total_fees_earned_usd = 0.0
        total_il_usd = 0.0
        total_rebalances = 0
        total_rebalance_cost_usd = 0.0
        position_history = [] 

        current_t_start = t_start_total
        
        init_state = self._init_state_at_t0(t_start_total)
        current_tick = init_state.tick
        
        # Prepare data for feature calculation
        swaps_df = self.swaps.copy()
        swaps_df['evt_block_time'] = pd.to_datetime(swaps_df['evt_block_time'], utc=True)

        Q96 = 2**96
        def get_price_float_internal(sqrtPriceX96):
            return (float(sqrtPriceX96) / Q96) ** 2 * (10 ** (D0 - D1))

        swaps_df['price'] = swaps_df['sqrtPriceX96'].apply(get_price_float_internal)
        swaps_df = swaps_df.sort_values('evt_block_time').set_index('evt_block_time')
        resampled_prices = swaps_df['price'].resample('1H').last().ffill().dropna()

        # Main Loop (rebalances on a fixed schedule)
        while current_t_start < t_end_total:
            
            # Define segment time (Fixed by prediction horizon)
            current_t_end = current_t_start + timedelta(hours=time_step_hours)
            current_t_end = min(current_t_end, t_end_total)
            
            # --- 1. Calculate Features at current_t_start ---
            feature_time_index = resampled_prices.index[resampled_prices.index <= current_t_start].max()
            
            if pd.isna(feature_time_index) or feature_time_index < (t_start_total + timedelta(hours=24)):
                current_t_start = current_t_end
                continue
            
            lookback_data = resampled_prices.loc[:feature_time_index].tail(24)
            
            if len(lookback_data) < 24: # Need 24 hours of data to calculate 24h vol
                current_t_start = current_t_end
                continue

            current_price = lookback_data.iloc[-1]
            log_ret_1h = np.log(current_price) - np.log(lookback_data.iloc[-2])
            log_returns_24h = np.log(lookback_data).diff().dropna()
            vol_24h = log_returns_24h.std()
            
            features = pd.DataFrame({
                'price': [current_price],
                'log_ret_1h': [log_ret_1h],
                'vol_24h': [vol_24h]
            })

            # --- 2. Make Prediction ---
            price_lower_px = model_min.predict(features)[0]
            price_upper_px = model_max.predict(features)[0]

            # Ensure lower < upper and both are positive
            price_lower_px = max(1e-18, price_lower_px)
            if price_lower_px >= price_upper_px:
                # Default to a narrow range if prediction fails (e.g., 2% range centered on current price)
                price_current_px_dec = sqrt_price_x96_to_price(tick_to_sqrt_price_x96(current_tick), self.cfg.decimals0, self.cfg.decimals1)
                price_lower_px = float(price_current_px_dec * Decimal('0.98'))
                price_upper_px = float(price_current_px_dec * Decimal('1.02'))
                
            # --- 3. Set Range and Simulate ---
            lower_tick = round_tick_to_spacing(price_to_tick(Decimal(str(price_lower_px)), self.cfg.decimals0, self.cfg.decimals1), self.cfg.tick_spacing)
            upper_tick = round_up_to_spacing(price_to_tick(Decimal(str(price_upper_px)), self.cfg.decimals0, self.cfg.decimals1), self.cfg.tick_spacing)
            
            price_lower_px = float(sqrt_price_x96_to_price(tick_to_sqrt_price_x96(lower_tick), self.cfg.decimals0, self.cfg.decimals1))
            price_upper_px = float(sqrt_price_x96_to_price(tick_to_sqrt_price_x96(upper_tick), self.cfg.decimals0, self.cfg.decimals1))
            
            try:
                segment_result = self.simulate(
                    price_lower=price_lower_px,
                    price_upper=price_upper_px,
                    start=current_t_start.isoformat(),
                    end=current_t_end.isoformat(),
                    total_usd=total_usd_budget,
                    validate=False,
                    use_swap_liquidity=False,
                    accounting_mode='growth',
                    protocol_fee_encoding='base256',
                )
            except SystemExit:
                current_t_start = current_t_end 
                continue

            # --- 4. Accumulate Fees and Update State ---
            fees_usd = segment_result.fees_usd.token0 + segment_result.fees_usd.token1
            total_fees_earned_usd += fees_usd
            total_il_usd += segment_result.impermanent_loss.usd
            
            # Rebalance is MANDATORY at the end of the predictive window
            rebalance_needed_final = True if current_t_start < t_end_total else False
            
            position_history.append({
                'start': current_t_start,
                'end': current_t_end,
                'price_lower': price_lower_px,
                'price_upper': price_upper_px,
                'rebalance': rebalance_needed_final
            })
            
            current_t_start = current_t_end
            
            if rebalance_needed_final:
                total_rebalances += 1
                total_rebalance_cost_usd += rebalance_fee_usd
                
                price_at_end = segment_result.price.end
                price_end_px_decimal = Decimal(str(price_at_end))
                current_tick = price_to_tick(price_end_px_decimal, self.cfg.decimals0, self.cfg.decimals1)
                
        # Final Result
        return {
            'total_fees_earned_usd': total_fees_earned_usd,
            'total_il_usd': total_il_usd,
            'total_rebalances': total_rebalances,
            'total_rebalance_cost_usd': total_rebalance_cost_usd,
            'net_strategy_gain_usd': total_fees_earned_usd + total_il_usd - total_rebalance_cost_usd
        }, position_history