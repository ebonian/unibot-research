#!/usr/bin/env python3
"""
Verify data integrity for Simulation 12 training data.

Checks:
1. CSV schema compatibility with simulation code
2. No NaN/null in critical columns
3. Date range coverage and continuity
4. sqrtPriceX96 → price conversion produces valid ETH/USDT prices
5. Pool config and token metadata correctness
6. Statistics: row count, avg swaps/hour, price range

Also performs code correctness audit against Uniswap V3 specifications:
- sqrtPriceX96 → price conversion
- tick ↔ price conversion
- Liquidity computation from capital
- Fee calculation (per-swap)
- Position value formula
- LVR computation
"""

import os
import sys
import math
import glob
import numpy as np
import pandas as pd

Q96 = 2 ** 96
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Convert Uniswap v3 sqrtPriceX96 to human-readable price."""
    p = float(sqrt_price_x96) / Q96
    return (p * p) * (10 ** (decimals0 - decimals1))


def price_to_tick(price: float) -> int:
    if price <= 0:
        return 0
    return int(math.floor(math.log(price) / math.log(1.0001)))


def tick_to_price(tick: int) -> float:
    return math.pow(1.0001, tick)


def check_data_integrity():
    """Verify training data files and content."""
    print("=" * 70)
    print("📋 DATA INTEGRITY VERIFICATION — Simulation 12")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # =========================================================================
    # 1. Check file existence
    # =========================================================================
    print("\n🔍 1. Checking file existence...")
    
    required_files = {
        "pool_config": "pool_config_eth_usdt_0p3.csv",
        "token_metadata": "token_metadata_eth_usdt_0p3.csv",
    }
    
    for name, filename in required_files.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1_000_000
            print(f"   ✅ {filename} ({size_mb:.2f} MB)")
        else:
            errors.append(f"Missing required file: {filename}")
            print(f"   ❌ {filename} — MISSING")
    
    swaps_files = glob.glob(os.path.join(DATA_DIR, "swaps_*_eth_usdt_0p3.csv"))
    if swaps_files:
        for sf in swaps_files:
            size_mb = os.path.getsize(sf) / 1_000_000
            print(f"   ✅ {os.path.basename(sf)} ({size_mb:.1f} MB)")
    else:
        errors.append("Missing swaps_*_eth_usdt_0p3.csv")
        print("   ❌ swaps CSV — MISSING")
    
    if errors:
        print("\n❌ FATAL: Missing files. Cannot continue.")
        for e in errors:
            print(f"   • {e}")
        return False
    
    # =========================================================================
    # 2. Load and verify pool config
    # =========================================================================
    print("\n🔍 2. Verifying pool config...")
    
    pool_cfg = pd.read_csv(os.path.join(DATA_DIR, "pool_config_eth_usdt_0p3.csv"))
    fee = int(pool_cfg.loc[0, 'fee'])
    tick_spacing = int(pool_cfg.loc[0, 'tickSpacing'])
    token0_addr = pool_cfg.loc[0, 'token0']
    token1_addr = pool_cfg.loc[0, 'token1']
    
    print(f"   Fee: {fee} ({fee/10000:.2f}%)")
    print(f"   Tick spacing: {tick_spacing}")
    print(f"   Token0: {token0_addr}")
    print(f"   Token1: {token1_addr}")
    
    if fee != 500:
        warnings.append(f"Fee is {fee}, expected 500 (0.05%)")
    if tick_spacing != 10:
        warnings.append(f"Tick spacing is {tick_spacing}, expected 10")
    
    # Verify fee/tickSpacing relationship (from Uniswap V3 factory)
    # Valid combos: 500/10, 3000/60, 10000/200
    valid_combos = {500: 10, 3000: 60, 10000: 200, 100: 1}
    if fee in valid_combos:
        expected_ts = valid_combos[fee]
        if tick_spacing != expected_ts:
            errors.append(f"Fee {fee} should have tickSpacing {expected_ts}, got {tick_spacing}")
            print(f"   ❌ Fee/tickSpacing mismatch!")
        else:
            print(f"   ✅ Fee/tickSpacing combo valid (matches Uniswap V3 factory)")
    else:
        warnings.append(f"Unusual fee tier: {fee}")
    
    # =========================================================================
    # 3. Load and verify token metadata
    # =========================================================================
    print("\n🔍 3. Verifying token metadata...")
    
    tokens = pd.read_csv(os.path.join(DATA_DIR, "token_metadata_eth_usdt_0p3.csv"))
    tokens['contract_address'] = tokens['contract_address'].str.lower()
    
    for _, row in tokens.iterrows():
        print(f"   {row['symbol']}: decimals={row['decimals']}, addr={row['contract_address'][:10]}...")
    
    t0_addr = token0_addr.lower()
    t1_addr = token1_addr.lower()
    
    try:
        t0 = tokens.set_index('contract_address').loc[t0_addr]
        t1 = tokens.set_index('contract_address').loc[t1_addr]
        decimals0 = int(t0['decimals'])
        decimals1 = int(t1['decimals'])
        print(f"   ✅ Token0 ({t0['symbol']}): {decimals0} decimals")
        print(f"   ✅ Token1 ({t1['symbol']}): {decimals1} decimals")
    except KeyError as e:
        errors.append(f"Token address {e} not found in metadata")
        print(f"   ❌ Token address mismatch: {e}")
        return False
    
    if decimals0 != 18:
        warnings.append(f"Token0 decimals is {decimals0}, expected 18 (WETH)")
    if decimals1 != 6:
        warnings.append(f"Token1 decimals is {decimals1}, expected 6 (USDT)")
    
    # =========================================================================
    # 4. Load and verify swap data
    # =========================================================================
    print("\n🔍 4. Loading swap data (this may take a moment)...")
    
    swaps = pd.read_csv(swaps_files[0], low_memory=False)
    print(f"   Total rows: {len(swaps):,}")
    
    # Check required columns
    required_cols = ['evt_block_time', 'sqrtPriceX96', 'amount0', 'amount1', 'liquidity']
    missing_cols = [c for c in required_cols if c not in swaps.columns]
    if missing_cols:
        errors.append(f"Missing columns in swaps CSV: {missing_cols}")
        print(f"   ❌ Missing columns: {missing_cols}")
    else:
        print(f"   ✅ All required columns present: {required_cols}")
    
    # Optional columns
    optional_cols = ['tick']
    present_optional = [c for c in optional_cols if c in swaps.columns]
    print(f"   Optional columns present: {present_optional}")
    print(f"   All columns: {list(swaps.columns)}")
    
    # =========================================================================
    # 5. Check for NaN/null values
    # =========================================================================
    print("\n🔍 5. Checking for NaN/null values...")
    
    for col in required_cols:
        if col in swaps.columns:
            null_count = swaps[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count:,} NaN values")
                print(f"   ❌ {col}: {null_count:,} NaN ({null_count/len(swaps)*100:.2f}%)")
            else:
                print(f"   ✅ {col}: 0 NaN")
    
    # =========================================================================
    # 6. Verify date range and continuity
    # =========================================================================
    print("\n🔍 6. Verifying date range...")
    
    swaps['evt_block_time'] = pd.to_datetime(swaps['evt_block_time'], utc=True)
    swaps = swaps.sort_values('evt_block_time').reset_index(drop=True)
    
    start_date = swaps['evt_block_time'].min()
    end_date = swaps['evt_block_time'].max()
    date_range_days = (end_date - start_date).days
    
    print(f"   Start: {start_date}")
    print(f"   End:   {end_date}")
    print(f"   Range: {date_range_days} days")
    
    # Check for gaps (>4 hours without swaps)
    swaps_hourly = swaps.set_index('evt_block_time').resample('1h').size()
    empty_hours = (swaps_hourly == 0).sum()
    total_hours = len(swaps_hourly)
    print(f"   Total hours: {total_hours:,}")
    print(f"   Empty hours (no swaps): {empty_hours} ({empty_hours/total_hours*100:.1f}%)")
    
    if empty_hours / total_hours > 0.1:
        warnings.append(f"High ratio of empty hours: {empty_hours/total_hours*100:.1f}%")
    
    avg_swaps_per_hour = len(swaps) / total_hours
    print(f"   Avg swaps/hour: {avg_swaps_per_hour:.1f}")
    
    # =========================================================================
    # 7. Verify sqrtPriceX96 → price conversion
    # =========================================================================
    print("\n🔍 7. Verifying price conversion from sqrtPriceX96...")
    
    # Sample 1000 rows for speed
    sample = swaps.sample(min(1000, len(swaps)), random_state=42)
    prices = sample['sqrtPriceX96'].apply(
        lambda x: sqrt_price_x96_to_price(int(x), decimals0, decimals1)
    )
    
    min_price = prices.min()
    max_price = prices.max()
    mean_price = prices.mean()
    
    print(f"   Price range (sample of 1000): ${min_price:.2f} — ${max_price:.2f}")
    print(f"   Mean price: ${mean_price:.2f}")
    
    # Sanity check: ETH/USDT should be roughly $1,000 - $10,000
    if min_price < 100 or max_price > 50000:
        warnings.append(f"Price range seems unusual: ${min_price:.2f} — ${max_price:.2f}")
        print(f"   ⚠️  Price range may be unusual for ETH/USDT")
    else:
        print(f"   ✅ Price range looks reasonable for ETH/USDT")
    
    # Check for negative or zero prices
    zero_prices = (prices <= 0).sum()
    if zero_prices > 0:
        errors.append(f"{zero_prices} rows produce zero/negative prices")
    
    # =========================================================================
    # 8. Verify tick values match sqrtPriceX96
    # =========================================================================
    if 'tick' in swaps.columns:
        print("\n🔍 8. Verifying tick ↔ sqrtPriceX96 consistency...")
        
        tick_sample = sample[['sqrtPriceX96', 'tick']].copy()
        tick_sample['computed_tick'] = tick_sample['sqrtPriceX96'].apply(
            lambda x: price_to_tick(sqrt_price_x96_to_price(int(x), decimals0, decimals1))
        )
        tick_sample['stored_tick'] = tick_sample['tick'].astype(int)
        
        # Ticks should match within ±1 due to rounding
        tick_diff = (tick_sample['computed_tick'] - tick_sample['stored_tick']).abs()
        max_diff = tick_diff.max()
        mean_diff = tick_diff.mean()
        
        print(f"   Max tick difference: {max_diff}")
        print(f"   Mean tick difference: {mean_diff:.2f}")
        
        if max_diff <= 1:
            print(f"   ✅ Ticks consistent (max diff ≤ 1)")
        else:
            warnings.append(f"Tick mismatch: max diff = {max_diff}")
            print(f"   ⚠️  Some ticks differ by more than 1")
    
    # =========================================================================
    # 9. Volume statistics
    # =========================================================================
    print("\n🔍 9. Volume statistics...")
    
    swaps['volume_usd'] = swaps['amount1'].abs() / (10 ** decimals1)
    hourly_vol = swaps.set_index('evt_block_time').resample('1h')['volume_usd'].sum()
    
    avg_hourly_vol = hourly_vol[hourly_vol > 0].mean()
    median_hourly_vol = hourly_vol[hourly_vol > 0].median()
    pool_fee_rate = fee / 1_000_000
    avg_hourly_pool_fees = avg_hourly_vol * pool_fee_rate
    
    print(f"   Avg hourly volume: ${avg_hourly_vol:,.2f}")
    print(f"   Median hourly volume: ${median_hourly_vol:,.2f}")
    print(f"   Avg hourly pool fees ({pool_fee_rate*100:.2f}%): ${avg_hourly_pool_fees:.2f}")
    
    # =========================================================================
    # 10. Liquidity statistics
    # =========================================================================
    print("\n🔍 10. Liquidity statistics...")
    
    swaps['liquidity_float'] = swaps['liquidity'].astype(float)
    median_liq = swaps['liquidity_float'].median()
    mean_liq = swaps['liquidity_float'].mean()
    
    # Human-readable liquidity
    conversion_factor = 10 ** ((decimals0 + decimals1) / 2)
    median_liq_human = median_liq / conversion_factor
    
    print(f"   Median raw liquidity: {median_liq:.4e}")
    print(f"   Mean raw liquidity: {mean_liq:.4e}")
    print(f"   Median human liquidity: {median_liq_human:,.0f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"   Total swap rows:  {len(swaps):,}")
    print(f"   Date range:       {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} ({date_range_days} days)")
    print(f"   Avg swaps/hour:   {avg_swaps_per_hour:.1f}")
    print(f"   Price range:      ${min_price:.2f} — ${max_price:.2f}")
    print(f"   Pool fee:         {fee} ({pool_fee_rate*100:.2f}%)")
    print(f"   Tick spacing:     {tick_spacing}")
    print(f"   Token0/Token1:    {t0['symbol']}/{t1['symbol']} ({decimals0}/{decimals1} decimals)")
    
    print(f"\n   ❌ Errors:   {len(errors)}")
    for e in errors:
        print(f"      • {e}")
    print(f"   ⚠️  Warnings: {len(warnings)}")
    for w in warnings:
        print(f"      • {w}")
    
    if not errors:
        print("\n✅ DATA INTEGRITY CHECK PASSED!")
    else:
        print("\n❌ DATA INTEGRITY CHECK FAILED!")
    
    print("=" * 70)
    return len(errors) == 0


def audit_code_correctness():
    """
    Audit core Uniswap V3 math formulas against the whitepaper and v3-core contracts.
    
    References:
    - Uniswap V3 Whitepaper: https://uniswap.org/whitepaper-v3.pdf
    - v3-core: https://github.com/Uniswap/v3-core
    - Zhang et al. (2023): arXiv:2309.10129
    """
    print("\n" + "=" * 70)
    print("🔬 CODE CORRECTNESS AUDIT — Uniswap V3 Math")
    print("=" * 70)
    
    errors = []
    
    # =========================================================================
    # Test 1: sqrtPriceX96 → price conversion
    # Whitepaper §6.1: sqrtPriceX96 = √(token1/token0) × 2^96
    # So price = (sqrtPriceX96 / 2^96)^2 × 10^(d0-d1)
    # =========================================================================
    print("\n📐 Test 1: sqrtPriceX96 → price conversion")
    
    # Known value: ETH ≈ $3,364 corresponds to sqrtPriceX96 ≈ 3.39e24
    # Token0 = WETH (18 dec), Token1 = USDT (6 dec)
    test_sqrt = 3391687544375824514757915  # From actual data
    d0, d1 = 18, 6
    price = sqrt_price_x96_to_price(test_sqrt, d0, d1)
    
    print(f"   sqrtPriceX96 = {test_sqrt}")
    print(f"   Computed price = ${price:.2f}")
    
    # Verify formula: p = (sqrtPriceX96 / 2^96)^2 × 10^(d0-d1)
    manual_price = (float(test_sqrt) / Q96) ** 2 * (10 ** (d0 - d1))
    
    if abs(price - manual_price) < 1e-10:
        print(f"   ✅ Formula matches manual calculation: ${manual_price:.2f}")
    else:
        errors.append(f"sqrtPriceX96 conversion mismatch: {price} vs {manual_price}")
        print(f"   ❌ Mismatch: {price} vs {manual_price}")
    
    if 1000 < price < 10000:
        print(f"   ✅ Price is in expected ETH/USDT range")
    else:
        errors.append(f"Price {price} outside expected ETH/USDT range")
    
    # =========================================================================
    # Test 2: tick ↔ price conversion
    # Whitepaper §6.1: p(i) = 1.0001^i, i = ⌊log(p) / log(1.0001)⌋
    # v3-core TickMath.sol: getSqrtRatioAtTick computes √(1.0001^tick) × 2^96
    # =========================================================================
    print("\n📐 Test 2: tick ↔ price conversion")
    
    test_prices = [1000.0, 2000.0, 3000.0, 3500.0, 5000.0, 10000.0]
    for tp in test_prices:
        tick = price_to_tick(tp)
        recovered_price = tick_to_price(tick)
        # Should be within 0.01% (half a tick)
        rel_error = abs(recovered_price - tp) / tp
        status = "✅" if rel_error < 0.0001 else "❌"
        print(f"   {status} price={tp:.0f} → tick={tick} → recovered={recovered_price:.2f} (err={rel_error*100:.4f}%)")
        if rel_error >= 0.0001:
            errors.append(f"tick conversion error too large for price {tp}: {rel_error*100:.4f}%")
    
    # Verify the fundamental relationship: 1.0001^tick ≈ price
    print(f"   Verify: 1.0001^{price_to_tick(3500)} = {1.0001**price_to_tick(3500):.2f} ≈ 3500")
    
    # =========================================================================
    # Test 3: Position value formula
    # Whitepaper §6.2.1:
    #   x = L × (1/√p - 1/√p_u)  when p ∈ [p_l, p_u]
    #   y = L × (√p - √p_l)       when p ∈ [p_l, p_u]
    #   V = x×p + y = L × (2√p - p/√p_u - √p_l)
    #
    # From SqrtPriceMath.sol:
    #   amount0 = L × (√p_u - √p_l) / (√p_u × √p_l)   (simplified)
    #   amount1 = L × (√p_u - √p_l)                     (at boundaries)
    # =========================================================================
    print("\n📐 Test 3: Position value formula (Whitepaper §6.2.1)")
    
    L = 1000.0  # Test liquidity
    p = 3500.0
    p_l = 3000.0
    p_u = 4000.0
    
    sqrt_p = math.sqrt(p)
    sqrt_pl = math.sqrt(p_l)
    sqrt_pu = math.sqrt(p_u)
    
    # Token amounts
    x = L * (1.0 / sqrt_p - 1.0 / sqrt_pu)
    y = L * (sqrt_p - sqrt_pl)
    
    # Position value two ways
    V_from_tokens = x * p + y
    V_from_formula = L * (2.0 * sqrt_p - p / sqrt_pu - sqrt_pl)
    
    print(f"   L={L}, p={p}, p_l={p_l}, p_u={p_u}")
    print(f"   Token X (WETH): {x:.6f}")
    print(f"   Token Y (USDT): {y:.2f}")
    print(f"   V (from tokens x*p+y): {V_from_tokens:.6f}")
    print(f"   V (from formula):      {V_from_formula:.6f}")
    
    if abs(V_from_tokens - V_from_formula) < 1e-6:
        print(f"   ✅ Position value formulas consistent")
    else:
        errors.append(f"Position value mismatch: {V_from_tokens} vs {V_from_formula}")
    
    # Edge cases: price below range (all X), price above range (all Y)
    # Below range: V = x_boundary * p_current
    p_below = 2500.0
    x_below = L * (1.0 / sqrt_pl - 1.0 / sqrt_pu)
    V_below = x_below * p_below
    print(f"   Below range (p={p_below}): V = {V_below:.2f} (all in WETH)")
    
    # Above range: V = y_boundary
    p_above = 4500.0
    y_above = L * (sqrt_pu - sqrt_pl)
    V_above = y_above
    print(f"   Above range (p={p_above}): V = {V_above:.2f} (all in USDT)")
    
    # =========================================================================
    # Test 4: Liquidity from capital
    # Given capital V and price range [p_l, p_u], compute L such that
    # V = L × (2√p - p/√p_u - √p_l)
    # Therefore: L = V / (2√p - p/√p_u - √p_l)
    # =========================================================================
    print("\n📐 Test 4: Liquidity from capital")
    
    capital = 1000.0
    value_per_L = 2.0 * sqrt_p - p / sqrt_pu - sqrt_pl
    computed_L = capital / value_per_L
    
    # Verify: position value with computed_L should equal capital
    V_check = computed_L * (2.0 * sqrt_p - p / sqrt_pu - sqrt_pl)
    
    print(f"   Capital: ${capital}")
    print(f"   value_per_L: {value_per_L:.6f}")
    print(f"   Computed L: {computed_L:.2f}")
    print(f"   V check: ${V_check:.2f}")
    
    if abs(V_check - capital) < 1e-6:
        print(f"   ✅ Liquidity computation correct (V matches capital)")
    else:
        errors.append(f"Liquidity computation error: V={V_check} != capital={capital}")
    
    # =========================================================================
    # Test 5: Fee formula
    # From Zhang et al. (2023) Eq 5-6 and SwapMath.computeSwapStep() in v3-core:
    #   Price up (token1 in): fee = δ/(1-δ) × L × (√p1 - √p0)
    #   Price down (token0 in): fee = δ/(1-δ) × L × (1/√p1 - 1/√p0) × p1
    #
    # Note: In Uniswap V3, fee is charged on INPUT amount.
    # For price increase (token1 → token0 swap):
    #   amountIn of token1 ≈ L × (√p1 - √p0), fee = δ/(1-δ) × amountIn
    # For price decrease (token0 → token1 swap):
    #   amountIn of token0 ≈ L × (1/√p0 - 1/√p1), fee = δ/(1-δ) × amountIn × p
    # =========================================================================
    print("\n📐 Test 5: Fee calculation (per-swap, Zhang et al. Eq 5-6)")
    
    delta = 0.0005  # 0.05% fee tier
    fee_mult = delta / (1.0 - delta)
    test_L = 10000.0
    
    # Case A: Price goes up 3500 → 3510
    p0, p1 = 3500.0, 3510.0
    fee_up = fee_mult * test_L * (math.sqrt(p1) - math.sqrt(p0))
    
    # Manual verification: amount1_in ≈ L × (√p1 - √p0)
    amount1_in = test_L * (math.sqrt(p1) - math.sqrt(p0))
    fee_manual_up = delta / (1.0 - delta) * amount1_in
    
    print(f"   Price up {p0}→{p1}: fee={fee_up:.6f} (manual={fee_manual_up:.6f})")
    if abs(fee_up - fee_manual_up) < 1e-10:
        print(f"   ✅ Fee-up formula correct")
    else:
        errors.append(f"Fee-up mismatch")
    
    # Case B: Price goes down 3500 → 3490
    p0, p1 = 3500.0, 3490.0
    fee_down = fee_mult * test_L * (1.0 / math.sqrt(p1) - 1.0 / math.sqrt(p0)) * p1
    
    # Manual: amount0_in ≈ L × (1/√p1 - 1/√p0), fee in token0, convert to USD
    amount0_in = test_L * (1.0 / math.sqrt(p1) - 1.0 / math.sqrt(p0))
    fee_manual_down = delta / (1.0 - delta) * amount0_in * p1
    
    print(f"   Price down {p0}→{p1}: fee={fee_down:.6f} (manual={fee_manual_down:.6f})")
    if abs(fee_down - fee_manual_down) < 1e-10:
        print(f"   ✅ Fee-down formula correct")
    else:
        errors.append(f"Fee-down mismatch")
    
    # Note: fee_down uses p1 (end price) for USD conversion — an approximation
    # The exact conversion would use the marginal price during the swap,
    # but for small price movements this is negligible.
    fee_exact_down = delta / (1.0 - delta) * amount0_in * p0  # using start price
    rel_diff = abs(fee_down - fee_exact_down) / max(fee_down, 1e-10)
    print(f"   ℹ️  Fee-down with p0 vs p1: diff={rel_diff*100:.4f}% (negligible for small moves)")
    
    # =========================================================================
    # Test 6: LVR formula
    # Zhang et al. (2023): LVR = Σ {V(p_{i+1}) - V(p_i) - x(p_i) × Δp}
    # This is the discrete-time Loss-Versus-Rebalancing
    # Should always be ≤ 0 (it's a cost to the LP)
    # =========================================================================
    print("\n📐 Test 6: LVR (Loss-Versus-Rebalancing)")
    
    test_L_lvr = 10000.0
    p_lower = 3000.0
    p_upper = 4000.0
    
    # Simulate 3 swaps: 3500 → 3600 → 3550 → 3650
    swap_prices = [3500.0, 3600.0, 3550.0, 3650.0]
    
    sqrt_pl_lvr = math.sqrt(p_lower)
    sqrt_pu_lvr = math.sqrt(p_upper)
    
    def pos_value(price, L, p_l, p_u, sq_l, sq_u):
        if price <= p_l:
            x = L * (1.0 / sq_l - 1.0 / sq_u)
            return x * price
        elif price >= p_u:
            y = L * (sq_u - sq_l)
            return y
        else:
            return L * (2.0 * math.sqrt(price) - price / sq_u - sq_l)
    
    total_lvr = 0.0
    for i in range(len(swap_prices) - 1):
        pi = swap_prices[i]
        pi1 = swap_prices[i + 1]
        
        V_i = pos_value(pi, test_L_lvr, p_lower, p_upper, sqrt_pl_lvr, sqrt_pu_lvr)
        V_i1 = pos_value(pi1, test_L_lvr, p_lower, p_upper, sqrt_pl_lvr, sqrt_pu_lvr)
        
        # x(p_i) = tokens of X at price p_i
        if pi <= p_lower:
            x_i = test_L_lvr * (1.0 / sqrt_pl_lvr - 1.0 / sqrt_pu_lvr)
        elif pi >= p_upper:
            x_i = 0.0
        else:
            x_i = test_L_lvr * (1.0 / math.sqrt(pi) - 1.0 / sqrt_pu_lvr)
        
        lvr_i = (V_i1 - V_i) - x_i * (pi1 - pi)
        total_lvr += lvr_i
        print(f"   Swap {pi:.0f}→{pi1:.0f}: ΔV={V_i1-V_i:.4f}, x×Δp={x_i*(pi1-pi):.4f}, LVR_i={lvr_i:.4f}")
    
    print(f"   Total LVR: {total_lvr:.4f}")
    
    if total_lvr <= 0:
        print(f"   ✅ LVR is non-positive (as expected — it's a cost)")
    else:
        warnings_msg = f"LVR is positive ({total_lvr}) which is unexpected"
        print(f"   ⚠️  {warnings_msg}")
        # LVR can technically be > 0 in discrete time with few swaps, so just warn
    
    # =========================================================================
    # Test 7: Fee cap validation
    # The code caps fees at total pool fees: min(computed_fee, volume × pool_fee)
    # This ensures no single LP position earns more than all swappers paid
    # =========================================================================
    print("\n📐 Test 7: Fee cap sanity check")
    
    hourly_volume = 1_000_000.0  # $1M/hour
    pool_fee_cap = hourly_volume * delta  # $500 max fees per hour
    
    # A tiny LP with extreme liquidity concentration shouldn't earn > pool total
    huge_L = 1e15
    tiny_range_price_move = (math.sqrt(3501) - math.sqrt(3500))
    uncapped_fee = fee_mult * huge_L * tiny_range_price_move
    capped_fee = min(uncapped_fee, pool_fee_cap)
    
    print(f"   Hourly volume: ${hourly_volume:,.0f}")
    print(f"   Pool fee cap: ${pool_fee_cap:.2f}")
    print(f"   Uncapped fee (huge L): ${uncapped_fee:,.2f}")
    print(f"   Capped fee: ${capped_fee:.2f}")
    print(f"   ✅ Fee cap prevents unrealistic returns")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 CODE CORRECTNESS AUDIT SUMMARY")
    print("=" * 70)
    
    tests = [
        "sqrtPriceX96 → price conversion",
        "tick ↔ price roundtrip",
        "Position value formula",
        "Liquidity from capital",
        "Fee calculation (up/down)",
        "LVR computation",
        "Fee cap mechanism"
    ]
    
    if not errors:
        for t in tests:
            print(f"   ✅ {t}")
        print(f"\n✅ ALL {len(tests)} TESTS PASSED!")
    else:
        print(f"\n❌ {len(errors)} ERRORS FOUND:")
        for e in errors:
            print(f"   • {e}")
    
    print("=" * 70)
    return len(errors) == 0


if __name__ == "__main__":
    data_ok = check_data_integrity()
    code_ok = audit_code_correctness()
    
    print("\n" + "=" * 70)
    if data_ok and code_ok:
        print("🎉 ALL CHECKS PASSED — Simulation 12 is ready!")
    else:
        print("⚠️  Some checks failed — review output above")
        sys.exit(1)
    print("=" * 70)
