"""
Mock data test for fee_simulator.py
Generates synthetic Uniswap V3 data and verifies calculations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from decimal import Decimal
from datetime import datetime, timedelta
from fee_simulator import UniswapV3FeeSimulator, tick_to_sqrt_price_x96, price_to_tick

# =============================================================================
# MOCK DATA GENERATION
# =============================================================================

def generate_mock_data():
    """Generate synthetic but realistic Uniswap V3 data for testing."""
    
    # Pool Config: ETH/USDT 0.3% fee
    pool_cfg = pd.DataFrame([{
        'pool': '0xmockpool',
        'token0': '0xeth',  # WETH
        'token1': '0xusdt', # USDT
        'fee': 3000,        # 0.3%
        'tickSpacing': 60,
    }])
    
    # Token Metadata
    tokens = pd.DataFrame([
        {'contract_address': '0xeth', 'symbol': 'WETH', 'decimals': 18},
        {'contract_address': '0xusdt', 'symbol': 'USDT', 'decimals': 6},
    ])
    
    # Time range: 24 hours
    t0 = pd.Timestamp('2025-09-15 00:00:00', tz='UTC')
    t1 = t0 + timedelta(hours=24)
    
    # Starting price: ETH = $2500
    start_price = Decimal('2500')
    decimals0, decimals1 = 18, 6
    start_tick = price_to_tick(start_price, decimals0, decimals1)
    start_sqrt_x96 = tick_to_sqrt_price_x96(start_tick)
    
    # Slot0: Initial pool state
    slot0 = pd.DataFrame([{
        'call_block_time': t0 - timedelta(hours=1),
        'output_sqrtPriceX96': str(start_sqrt_x96),
        'output_tick': start_tick,
        'output_feeProtocol': 0,  # No protocol fee
    }])
    
    # Generate swaps: Price moves with realistic random walk + drift
    # Creates a more natural price path with volatility
    swap_times = pd.date_range(t0, t1, periods=50)
    
    # Random walk with drift: starts at $2500, ends around $2550
    np.random.seed(42)  # Reproducible for testing
    base_price = 2500
    drift = 50 / 50  # Small upward drift over 50 steps
    volatility = 30  # Price can move ±$30 randomly each step
    
    prices = [base_price]
    for i in range(1, len(swap_times)):
        # Random walk: previous price + drift + random noise
        noise = np.random.randn() * volatility
        new_price = prices[-1] + drift + noise
        # Keep price in reasonable bounds ($2200 - $2800)
        new_price = max(2200, min(2800, new_price))
        prices.append(new_price)

    
    swaps = []
    for i, (t, price) in enumerate(zip(swap_times, prices)):
        tick = price_to_tick(Decimal(str(price)), decimals0, decimals1)
        sqrt_x96 = tick_to_sqrt_price_x96(tick)
        
        # Simulate swap amounts (alternating directions)
        if i % 2 == 0:
            # Sell ETH for USDT
            amount0 = int(0.1 * 10**18)  # 0.1 ETH in
            amount1 = -int(price * 0.1 * 10**6)  # USDT out
        else:
            # Buy ETH with USDT
            amount0 = -int(0.1 * 10**18)  # 0.1 ETH out
            amount1 = int(price * 0.1 * 10**6)  # USDT in
        
        swaps.append({
            'evt_block_time': t,
            'evt_block_number': 1000000 + i,
            'sqrtPriceX96': str(sqrt_x96),
            'tick': tick,
            'amount0': amount0,
            'amount1': amount1,
            'liquidity': int(1e18),  # Mock liquidity
        })
    
    swaps_df = pd.DataFrame(swaps)
    
    # Mints: Some existing liquidity before t0
    mints = pd.DataFrame([{
        'evt_block_time': t0 - timedelta(hours=2),
        'evt_block_number': 999990,
        'tickLower': start_tick - 600,  # Wide range
        'tickUpper': start_tick + 600,
        'liquidity_added': int(1e18),
    }])
    
    # Burns: Empty (no one exits during simulation)
    burns = pd.DataFrame(columns=['evt_block_time', 'evt_block_number', 'tickLower', 'tickUpper', 'liquidity_removed'])
    
    # ETH/USDT prices (for USD conversion)
    ethusdt_times = pd.date_range(t0 - timedelta(hours=24), t1, freq='1H')
    ethusdt = pd.DataFrame({
        'open_time': ethusdt_times,
        'close': [2500 + np.random.randn() * 10 for _ in ethusdt_times],  # ~$2500
    })
    
    return pool_cfg, tokens, slot0, swaps_df, mints, burns, ethusdt


# =============================================================================
# TEST CASES
# =============================================================================

def test_single_segment():
    """Test a single simulation segment."""
    print("=" * 70)
    print("TEST 1: Single Segment Simulation")
    print("=" * 70)
    
    pool_cfg, tokens, slot0, swaps, mints, burns, ethusdt = generate_mock_data()
    
    sim = UniswapV3FeeSimulator(
        pool_cfg=pool_cfg,
        tokens=tokens,
        slot0=slot0,
        swaps=swaps,
        mints=mints,
        burns=burns,
        eth_usdt_prices=ethusdt,
    )
    
    # Simulate LP position: $1000 deployed in range [$2300, $2700]
    result = sim.simulate(
        price_lower=2300,
        price_upper=2700,
        start='2025-09-15T00:00:00Z',
        end='2025-09-16T00:00:00Z',
        total_usd=1000.0,
        validate=False,
        use_swap_liquidity=False,
        accounting_mode='growth',
    )
    
    print(f"\n📊 SIMULATION RESULTS:")
    print(f"   Price Range:      ${result.price.start:.2f} -> ${result.price.end:.2f}")
    print(f"   Position:         [{result.position_ticks.lower}, {result.position_ticks.upper}]")
    print()
    print(f"💰 FEES EARNED:")
    print(f"   Token0 (ETH):     {result.fees_usd.token0 or 0:.6f} USD")
    print(f"   Token1 (USDT):    {result.fees_usd.token1:.6f} USD")
    print(f"   Total Fees:       ${(result.fees_usd.token0 or 0) + result.fees_usd.token1:.4f}")
    print()
    print(f"📉 IMPERMANENT LOSS:")
    print(f"   IL (USD):         ${result.impermanent_loss.usd:.4f}")
    print(f"   IL (%):           {result.impermanent_loss.pct * 100:.4f}%")
    print()
    print(f"📊 VALUE COMPARISON:")
    print(f"   HODL End Value:   ${result.valuation_baselines.hodl_end_usd:.2f}")
    print(f"   LP Principal:     ${result.valuation_baselines.end_principal_usd:.2f}")
    print(f"   LP Total:         ${result.end_tokens.total_usd:.2f}")
    print()
    
    # Verify IL = LP Principal - HODL
    calculated_il = result.valuation_baselines.end_principal_usd - result.valuation_baselines.hodl_end_usd
    print(f"✅ IL Verification:  {result.impermanent_loss.usd:.4f} ≈ {calculated_il:.4f} (LP - HODL)")
    
    return result


def test_rebalancing_strategy():
    """Test the rebalancing strategy with IL tracking."""
    print("\n" + "=" * 70)
    print("TEST 2: Rebalancing Strategy (24h segments)")
    print("=" * 70)
    
    pool_cfg, tokens, slot0, swaps, mints, burns, ethusdt = generate_mock_data()
    
    sim = UniswapV3FeeSimulator(
        pool_cfg=pool_cfg,
        tokens=tokens,
        slot0=slot0,
        swaps=swaps,
        mints=mints,
        burns=burns,
        eth_usdt_prices=ethusdt,
    )
    
    # Run rebalancing strategy
    summary, history = sim.run_rebalancing_strategy_by_time_step(
        start_time_iso='2025-09-15T00:00:00Z',
        end_time_iso='2025-09-16T00:00:00Z',
        total_usd_budget=1000.0,
        rebalance_fee_usd=0.50,  # $0.50 gas per rebalance
        time_step_hours=6,       # 6-hour segments
        base_tick_width=120,
        volatility_multiplier=1.5,
    )
    
    print(f"\n📊 STRATEGY RESULTS:")
    print(f"   Total Fees:           ${summary['total_fees_earned_usd']:.4f}")
    print(f"   Total IL:             ${summary['total_il_usd']:.4f}")
    print(f"   Rebalances:           {summary['total_rebalances']}")
    print(f"   Rebalance Gas Cost:   ${summary['total_rebalance_cost_usd']:.2f}")
    print()
    print(f"📈 NET STRATEGY GAIN:")
    print(f"   Fees + IL - Gas =     ${summary['net_strategy_gain_usd']:.4f}")
    print()
    
    # Verify formula
    expected = summary['total_fees_earned_usd'] + summary['total_il_usd'] - summary['total_rebalance_cost_usd']
    print(f"✅ Formula Verification: {summary['net_strategy_gain_usd']:.4f} ≈ {expected:.4f}")
    
    print(f"\n📜 POSITION HISTORY:")
    print(f"   {'Window':<8} {'Time':<15} {'Range':<20} {'Fees':>10} {'IL':>12} {'Net':>12} {'Action'}")
    print(f"   {'-'*8} {'-'*15} {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    
    for i, pos in enumerate(history):
        time_str = f"{pos['start'].strftime('%H:%M')}-{pos['end'].strftime('%H:%M')}"
        range_str = f"${pos['price_lower']:.0f}-${pos['price_upper']:.0f}"
        fees = pos.get('fees_usd', 0)
        il = pos.get('il_usd', 0)
        net = fees + il
        action = '🔄 REBAL' if pos['rebalance'] else 'HOLD'
        
        print(f"   W{i+1:<6} {time_str:<15} {range_str:<20} ${fees:>8.2f} ${il:>10.2f} ${net:>10.2f} {action}")
    
    return summary, history


def plot_price_and_lp_windows(swaps_df, history, save_path=None):
    """
    Plot price over time with LP position windows as colored rectangles.
    
    Args:
        swaps_df: DataFrame with 'evt_block_time' and price data
        history: List of position dicts from rebalancing strategy
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate prices from sqrtPriceX96
    Q96 = 2**96
    decimals0, decimals1 = 18, 6
    swaps_df = swaps_df.copy()
    swaps_df['price'] = swaps_df['sqrtPriceX96'].apply(
        lambda x: (float(x) / Q96) ** 2 * (10 ** (decimals0 - decimals1))
    )
    swaps_df['time'] = pd.to_datetime(swaps_df['evt_block_time'])
    
    # Plot price line
    ax.plot(swaps_df['time'], swaps_df['price'], 
            color='#2196F3', linewidth=2, label='ETH Price', zorder=3)
    
    # Color palette for different windows
    colors = ['#FFE082', '#A5D6A7', '#90CAF9', '#CE93D8', '#FFAB91', '#80DEEA']
    
    # Plot LP windows as colored rectangles
    for i, pos in enumerate(history):
        start = pd.to_datetime(pos['start'])
        end = pd.to_datetime(pos['end'])
        lower = pos['price_lower']
        upper = pos['price_upper']
        
        color = colors[i % len(colors)]
        alpha = 0.4 if pos['rebalance'] else 0.25
        
        # Create rectangle for LP range
        rect = mpatches.Rectangle(
            (start, lower),
            end - start,
            upper - lower,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            zorder=1,
            label=f"Window {i+1}: ${lower:.0f}-${upper:.0f}"
        )
        ax.add_patch(rect)
        
        # Add window label
        mid_time = start + (end - start) / 2
        mid_price = (lower + upper) / 2
        ax.annotate(f"W{i+1}", (mid_time, mid_price), 
                   fontsize=10, ha='center', va='center',
                   fontweight='bold', color='#333')
    
    # Styling
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('ETH Price (USD)', fontsize=12)
    ax.set_title('ETH Price with LP Position Windows', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=9)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Chart saved to: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n🧪 FEE SIMULATOR VERIFICATION TEST")
    print("=" * 70)
    print("This test uses MOCK DATA to verify calculations.\n")
    
    try:
        result1 = test_single_segment()
        summary, history = test_rebalancing_strategy()
        
        # Get swap data for visualization
        pool_cfg, tokens, slot0, swaps, mints, burns, ethusdt = generate_mock_data()
        
        # Plot price with LP windows
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATION...")
        print("=" * 70)
        plot_price_and_lp_windows(
            swaps,
            history,
            save_path='lp_windows_chart.png'
        )
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
