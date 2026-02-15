import pandas as pd
import glob
import os
import shutil

def convert_parquet_to_csv():
    # Paths
    base_dir = "/Users/ohm/Documents/GitHub/ice-senior-project"
    source_dir = os.path.join(base_dir, "training_data_for_sim_12", "daily")
    target_dir = os.path.join(base_dir, "training_data_for_sim_12_csv")
    
    # Create target directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    print(f"Created target directory: {target_dir}")

    # =========================================================================
    # 1. Convert Swaps (Chunked Processing)
    # =========================================================================
    print("Converting swaps...")
    swaps_dir = os.path.join(source_dir, "swaps")
    files = sorted(glob.glob(os.path.join(swaps_dir, "*.parquet")))
    
    if not files:
        print("No swap parquet files found!")
        return

    # Determine output filename upfront
    # Naming convention: swaps_START_to_END_eth_usdt_0p3.csv
    # We need start/end dates. Let's peek at first and last file dates.
    # Assuming file names are YYYY-MM-DD.parquet
    start_date = os.path.basename(files[0]).replace(".parquet", "").replace("-", "")
    end_date = os.path.basename(files[-1]).replace(".parquet", "").replace("-", "")
    csv_filename = f"swaps_{start_date}_to_{end_date}_eth_usdt_0p3.csv"
    output_path = os.path.join(target_dir, csv_filename)
    
    print(f"Target CSV: {output_path}")
    print(f"Total files: {len(files)}")

    chunk_size = 50
    total_rows = 0
    
    for i in range(0, len(files), chunk_size):
        chunk_files = files[i : i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1} ({len(chunk_files)} files)...")
        
        dfs = []
        for f in chunk_files:
            try:
                df = pd.read_parquet(f)
                
                # Standardize columns
                rename_map = {
                    'timestamp': 'evt_block_time',
                    'sqrt_price_x96': 'sqrtPriceX96',
                }
                df = df.rename(columns=rename_map)
                
                # Convert timestamp
                df['evt_block_time'] = pd.to_datetime(df['evt_block_time'], unit='s', utc=True)
                
                # Select columns
                cols = ['evt_block_time', 'sqrtPriceX96', 'amount0', 'amount1', 'liquidity', 'tick']
                existing_cols = [c for c in cols if c in df.columns]
                dfs.append(df[existing_cols])
                
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if dfs:
            chunk_df = pd.concat(dfs, ignore_index=True)
            chunk_df = chunk_df.sort_values('evt_block_time')
            
            # Append to CSV
            # Write header only for first chunk
            header = (i == 0)
            mode = 'w' if i == 0 else 'a'
            chunk_df.to_csv(output_path, index=False, header=header, mode=mode)
            
            total_rows += len(chunk_df)
            print(f"  Appended {len(chunk_df)} rows. Total so far: {total_rows}")

    print(f"Saved merged swaps to {output_path}")
    print(f"Total rows: {total_rows}")
        
    # =========================================================================
    # 2. Extract/Create Pool Config
    # =========================================================================
    print("Creating pool config...")
    states_dir = os.path.join(source_dir, "states") # or state
    state_files = sorted(glob.glob(os.path.join(states_dir, "*.parquet")))
    
    fee_tier = 3000 # default 0.3%
    tick_spacing = 60
    
    if state_files:
        try:
            sdf = pd.read_parquet(state_files[0])
            if 'fee' in sdf.columns:
                fee_tier = int(sdf['fee'].iloc[0])
            if 'tick_spacing' in sdf.columns:
                tick_spacing = int(sdf['tick_spacing'].iloc[0])
            print(f"Extracted from data: Fee={fee_tier}, TickSpacing={tick_spacing}")
        except Exception as e:
             print(f"Could not read state file: {e}")
    else:
         print("No state files found, using defaults.")

    pool_config = pd.DataFrame([{
        'fee': fee_tier, 
        'tickSpacing': tick_spacing,
        'token0': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', # WETH
        'token1': '0xdAC17F958D2ee523a2206206994597C13D831ec7'  # USDT
    }])
    
    pool_config.to_csv(os.path.join(target_dir, "pool_config_eth_usdt_0p3.csv"), index=False)
    print(f"Saved pool_config_eth_usdt_0p3.csv (Fee={fee_tier})")


    # =========================================================================
    # 3. Create Token Metadata
    # =========================================================================
    print("Creating token metadata...")
    tokens = pd.DataFrame([
        {'contract_address': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 'symbol': 'WETH', 'decimals': 18},
        {'contract_address': '0xdAC17F958D2ee523a2206206994597C13D831ec7', 'symbol': 'USDT', 'decimals': 6}
    ])
    tokens.to_csv(os.path.join(target_dir, "token_metadata_eth_usdt_0p3.csv"), index=False)
    print("Saved token_metadata_eth_usdt_0p3.csv")

    print("\nConversion complete!")

if __name__ == "__main__":
    convert_parquet_to_csv()
