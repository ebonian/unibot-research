#!/usr/bin/env python3
"""
prepare_training_data.py

Concatenates daily parquet files from B2 into unified CSVs 
and maps columns to match the expected Dune pipeline format.

This enables using B2 data with the existing fee_simulator and training code.
"""

import os
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "downloaded_data" / "daily"
OUTPUT_DIR = BASE_DIR / "training_data"

# Column mappings: B2 column name -> Dune column name
SWAP_COLUMN_MAP = {
    "block_number": "evt_block_number",
    "timestamp": "evt_block_time",
    "tx_hash": "evt_tx_hash",
    "sqrt_price_x96": "sqrtPriceX96",
    "sender": "sender",
    "recipient": "recipient",
    "amount0": "amount0",
    "amount1": "amount1",
    "liquidity": "liquidity",
    "tick": "tick",
    "log_index": "log_index",
}

MINT_COLUMN_MAP = {
    "block_number": "evt_block_number",
    "timestamp": "evt_block_time",
    "tx_hash": "evt_tx_hash",
    "liquidity": "liquidity_added",
    "owner": "owner",
    "tick_lower": "tickLower",
    "tick_upper": "tickUpper",
    "amount0": "token0_in",
    "amount1": "token1_in",
    "log_index": "log_index",
}

BURN_COLUMN_MAP = {
    "block_number": "evt_block_number",
    "timestamp": "evt_block_time",
    "tx_hash": "evt_tx_hash",
    "liquidity": "liquidity_removed",
    "owner": "owner",
    "tick_lower": "tickLower",
    "tick_upper": "tickUpper",
    "amount0": "token0_out",
    "amount1": "token1_out",
    "log_index": "log_index",
}

SLOT0_COLUMN_MAP = {
    "block_number": "call_block_number",
    "timestamp": "call_block_time",
    "sqrt_price_x96": "output_sqrtPriceX96",
    "tick": "output_tick",
    "fee": "output_feeProtocol",
}

PRICE_COLUMN_MAP = {
    "open_time": "open_time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "close_time": "close_time",
}


def timestamp_to_datetime_str(ts: int) -> str:
    """Convert Unix timestamp to Dune-style datetime string."""
    dt = datetime.utcfromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.000 UTC")


def concat_parquet_files(folder_path: Path, column_map: dict, output_file: Path):
    """
    Concatenate all parquet files in a folder into a single CSV.
    
    Args:
        folder_path: Path to folder containing daily parquet files
        column_map: Dict mapping B2 columns to Dune columns
        output_file: Path to output CSV file
    """
    parquet_files = sorted(glob.glob(str(folder_path / "*.parquet")))
    
    if not parquet_files:
        print(f"⚠️  No parquet files found in {folder_path}")
        return None
    
    print(f"📂 Processing {folder_path.name}: {len(parquet_files)} files")
    
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Error reading {pf}: {e}")
    
    if not dfs:
        print(f"  ❌ No valid data found")
        return None
    
    # Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  📊 Combined: {len(combined):,} rows")
    
    # Rename columns to match Dune format
    rename_map = {k: v for k, v in column_map.items() if k in combined.columns}
    combined = combined.rename(columns=rename_map)
    
    # Convert timestamp to datetime string for time columns
    time_cols = ["evt_block_time", "call_block_time"]
    for col in time_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(timestamp_to_datetime_str)
    
    # Sort by time and block number
    sort_cols = []
    if "evt_block_number" in combined.columns:
        sort_cols.append("evt_block_number")
    if "call_block_number" in combined.columns:
        sort_cols.append("call_block_number")
    if "evt_block_time" in combined.columns:
        sort_cols.append("evt_block_time")
    if "call_block_time" in combined.columns:
        sort_cols.append("call_block_time")
    
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    
    # Save to CSV
    combined.to_csv(output_file, index=False)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✅ Saved: {output_file.name} ({file_size_mb:.1f} MB)")
    
    return combined


def create_ethusdt_hourly(prices_folder: Path, output_file: Path):
    """
    Create ETHUSDT hourly data file from B2 prices data.
    
    Maps to the format expected by the training code:
    - timestamp (ms)
    - open, high, low, close
    """
    parquet_files = sorted(glob.glob(str(prices_folder / "*.parquet")))
    
    if not parquet_files:
        print(f"⚠️  No price files found")
        return None
    
    print(f"📂 Processing prices: {len(parquet_files)} files")
    
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Error reading {pf}: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Expected columns for ETHUSDT hourly:
    # timestamp, open, high, low, close (at minimum)
    output_cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    existing_cols = [c for c in output_cols if c in combined.columns]
    
    if not existing_cols:
        print(f"  ❌ Required price columns not found")
        return None
    
    combined = combined[existing_cols].copy()
    combined = combined.sort_values("open_time").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["open_time"], keep="first")
    
    print(f"  📊 Combined: {len(combined):,} hourly rows")
    
    combined.to_csv(output_file, index=False)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✅ Saved: {output_file.name} ({file_size_mb:.1f} MB)")
    
    return combined


def create_pool_config(output_file: Path):
    """
    Create pool_config CSV (static data).
    Format expected by fee_simulator:
        pool, token0, token1, fee, tickSpacing
    """
    # ETH/USDT 0.05% pool on Arbitrum (the one we're tracking)
    config = {
        "pool": ["0x8c9d230d45d6cfee39a6680fb7cb7e8de7ea8e71"],
        "token0": ["0x82af49447d8a07e3bd95bd0d56f35241523fbab1"],  # WETH on Arbitrum
        "token1": ["0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9"],  # USDT on Arbitrum
        "fee": [500],       # 0.05% = 500 (from the data - fee column shows 500)
        "tickSpacing": [10],  # tick_spacing for 0.05% pools
    }
    df = pd.DataFrame(config)
    df.to_csv(output_file, index=False)
    print(f"  ✅ Created: {output_file.name}")
    return df


def create_token_metadata(output_file: Path):
    """
    Create token_metadata CSV (static data).
    Format expected by fee_simulator:
        contract_address, symbol, decimals
    """
    # Arbitrum token addresses
    metadata = {
        "contract_address": [
            "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",  # USDT on Arbitrum
            "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",  # WETH on Arbitrum
        ],
        "symbol": ["USDT", "WETH"],
        "decimals": [6, 18],
    }
    df = pd.DataFrame(metadata)
    df.to_csv(output_file, index=False)
    print(f"  ✅ Created: {output_file.name}")
    return df


def get_date_range(folder_path: Path) -> tuple:
    """Get start and end dates from parquet filenames."""
    parquet_files = sorted(glob.glob(str(folder_path / "*.parquet")))
    if not parquet_files:
        return None, None
    
    start_date = Path(parquet_files[0]).stem
    end_date = Path(parquet_files[-1]).stem
    return start_date, end_date


def main():
    print("=" * 60)
    print("🚀 B2 Data Preparation for Model Training")
    print("=" * 60)
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Get date range from swaps folder
    start_date, end_date = get_date_range(DATA_DIR / "swaps")
    if start_date and end_date:
        print(f"📅 Date range: {start_date} to {end_date}")
        date_suffix = f"_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}"
    else:
        date_suffix = ""
    
    print()
    
    # 1. Swaps
    concat_parquet_files(
        DATA_DIR / "swaps",
        SWAP_COLUMN_MAP,
        OUTPUT_DIR / f"swaps{date_suffix}_eth_usdt_0p3.csv"
    )
    
    print()
    
    # 2. Mints
    concat_parquet_files(
        DATA_DIR / "mints",
        MINT_COLUMN_MAP,
        OUTPUT_DIR / f"mints{date_suffix}_eth_usdt_0p3.csv"
    )
    
    print()
    
    # 3. Burns
    concat_parquet_files(
        DATA_DIR / "burns",
        BURN_COLUMN_MAP,
        OUTPUT_DIR / f"burns{date_suffix}_eth_usdt_0p3.csv"
    )
    
    print()
    
    # 4. Slot0 (states)
    concat_parquet_files(
        DATA_DIR / "states",
        SLOT0_COLUMN_MAP,
        OUTPUT_DIR / f"slot0{date_suffix}_eth_usdt_0p3.csv"
    )
    
    print()
    
    # 5. ETH/USDT prices
    create_ethusdt_hourly(
        DATA_DIR / "prices",
        OUTPUT_DIR / f"ETHUSDT_hourly_data{date_suffix}.csv"
    )
    
    print()
    
    # 6. Static config files
    print("📂 Creating static configuration files:")
    create_pool_config(OUTPUT_DIR / "pool_config_eth_usdt_0p3.csv")
    create_token_metadata(OUTPUT_DIR / "token_metadata_eth_usdt_0p3.csv")
    
    print()
    print("=" * 60)
    print("✅ Data preparation complete!")
    print(f"📁 Output files are in: {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print("  1. Update uniswap_v3_ppo_continuous.py DATA_DIR to point to training_data/")
    print("  2. Update file paths in the environment __init__ method")
    print("  3. Run training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
