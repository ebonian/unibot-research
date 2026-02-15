
import math
import pandas as pd
import glob
import os

def check_fees():
    data_dir = 'training_data'
    
    # Load config and files
    swaps_files = glob.glob(os.path.join(data_dir, 'swaps_*_eth_usdt_0p3.csv'))
    if not swaps_files:
        print("No swap files found")
        return

    swaps = pd.read_csv(swaps_files[0], low_memory=False)
    pool_cfg = pd.read_csv(os.path.join(data_dir, 'pool_config_eth_usdt_0p3.csv'))
    tokens = pd.read_csv(os.path.join(data_dir, 'token_metadata_eth_usdt_0p3.csv'))

    tokens['contract_address'] = tokens['contract_address'].str.lower()
    t0 = pool_cfg.loc[0, 'token0'].lower()
    t1 = pool_cfg.loc[0, 'token1'].lower()
    d0 = int(tokens.set_index('contract_address').loc[t0, 'decimals']) # 18 (WETH)
    d1 = int(tokens.set_index('contract_address').loc[t1, 'decimals']) # 6 (USDT)

    # Calculate Hourly Volume and Fees
    swaps['evt_block_time'] = pd.to_datetime(swaps['evt_block_time'], utc=True)
    swaps['amount1_usd'] = swaps['amount1'].abs() / (10**d1)
    
    # Use last 1000 hours so we get recent data
    swaps_recent = swaps.tail(50000) # roughly 1000 hours if 50 swaps/hr
    hourly_vol = swaps_recent.set_index('evt_block_time').resample('1h')['amount1_usd'].sum()
    avg_hourly_vol = hourly_vol[hourly_vol > 0].mean()
    
    pool_fee_rate = 0.0005 # 0.05%
    avg_hourly_pool_fees = avg_hourly_vol * pool_fee_rate

    # Calculate Pool Liquidity (Human Units)
    # L_raw median over recent period
    pool_L_raw_median = swaps_recent['liquidity'].astype(float).median()
    
    # Conversion Factor: 10^((d0+d1)/2)
    conversion_factor = 10 ** ((d0 + d1) / 2)
    pool_L_human = pool_L_raw_median / conversion_factor

    # Calculate Our Liquidity for $1000 at Width=1 (±0.1%)
    price = 3500.0
    width_fraction = 0.001 # 0.1% range
    p_lower = price * (1 - width_fraction)
    p_upper = price * (1 + width_fraction)
    
    # Formula for Liquidity from Amount
    # x = L * (1/sqrt(P) - 1/sqrt(Pu))
    # y = L * (sqrt(P) - sqrt(Pl))
    # V = x*P + y
    # V = L * (2*sqrt(P) - P/sqrt(Pu) - sqrt(Pl))
    
    sqrt_p = math.sqrt(price)
    sqrt_pl = math.sqrt(p_lower)
    sqrt_pu = math.sqrt(p_upper)
    
    v_per_L = 2*sqrt_p - price/sqrt_pu - sqrt_pl
    our_L = 1000.0 / v_per_L if v_per_L > 0 else 0

    share = our_L / pool_L_human
    our_hourly_fee = avg_hourly_pool_fees * share

    print("-" * 40)
    print(f"Correctness Verification for $1000 Position")
    print("-" * 40)
    print(f"Token Decimals: {d0} / {d1}")
    print(f"L Conversion Factor: 10^{ (d0+d1)/2 } = {conversion_factor:,.0f}")
    print(f"Pool Raw Liquidity: {pool_L_raw_median:.4e}")
    print(f"Pool Human Liquidity: {pool_L_human:,.0f}")
    print("-" * 40)
    print(f"Our Amount: $1000")
    print(f"Width: ±{width_fraction:.1%} (Full range {p_lower:.1f}-{p_upper:.1f})")
    print(f"Our Calculated L: {our_L:,.0f}")
    print(f"Share of Active L: {share:.8%} (1 part in {1/share:,.0f})")
    print("-" * 40)
    print(f"Avg Pool Hourly Volume: ${avg_hourly_vol:,.2f}")
    print(f"Avg Pool Hourly Fees: ${avg_hourly_pool_fees:.2f}")
    print(f"Our ESTIMATED Hourly Fee: ${our_hourly_fee:.6f}")
    print("-" * 40)
    print(f"Gas Cost: $0.50")
    if our_hourly_fee > 0:
        print(f"Hours to breakeven on gas: {0.50/our_hourly_fee:.2f} hours")
    else:
        print("Hours to breakeven: infinite")
    print("-" * 40)

if __name__ == "__main__":
    check_fees()
