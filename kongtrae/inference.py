"""
Kongtrae – Single-Step LP Decision Engine
==========================================
Feed current market data, get the model's LP decision for this hour.

Usage:
    python inference.py --model dqn --price 3000 --volume 5000000
    python inference.py --model ppo --ohlcv-csv market_data.csv
"""

import os
import sys
import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# Add simulation_12 to path for model/env definitions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(SCRIPT_DIR, "..", "research", "simulation_12")
sys.path.insert(0, SIM_DIR)

from uniswap_v3_dqn_paper import (
    DuelingDDQNAgent,
    LSTMDDQNAgent,
    SequenceStateWrapper,
    tick_to_price,
    price_to_tick,
    compute_technical_indicators,
)


# ─── Constants ────────────────────────────────────────────────────────────────
TICK_SPACING = 10
POOL_FEE = 0.0005  # 0.05%

ACTIONS = {
    0: {"label": "HOLD",      "width": None, "range_pct": 0},
    1: {"label": "WIDTH-1",   "width": 1,    "range_pct": 0.1},
    2: {"label": "WIDTH-3",   "width": 3,    "range_pct": 0.3},
    3: {"label": "WIDTH-5",   "width": 5,    "range_pct": 0.5},
    4: {"label": "WIDTH-10",  "width": 10,   "range_pct": 1.0},
    5: {"label": "WIDTH-20",  "width": 20,   "range_pct": 2.0},
    6: {"label": "WIDTH-40",  "width": 40,   "range_pct": 4.0},
}


def process_swap_csv(csv_path: str) -> pd.DataFrame:
    """
    Load raw Uniswap V3 swap CSV and conversion to hourly OHLCV with features.
    Expected cols: evt_block_time, sqrtPriceX96, amount0, amount1, liquidity, tick
    (or at least evt_block_time and sqrtPriceX96)
    """
    print(f"📊 Loading swap data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Check for required columns
    required = ['evt_block_time', 'sqrtPriceX96']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Swap CSV missing '{col}' column. Found: {df.columns.tolist()}")

    # Parse timestamps
    df['evt_block_time'] = pd.to_datetime(df['evt_block_time'], utc=True)
    df = df.sort_values('evt_block_time')

    # Convert sqrtPriceX96 to USD price
    # Assuming ETH/USDT 0.05% (decimals 18/6)
    DEC0, DEC1 = 18, 6
    Q96 = 2**96
    
    def price_fn(x):
        try:
            p = float(x) / Q96
            return (p*p) * (10**(DEC0 - DEC1))
        except:
            return 0.0
    
    df['price'] = df['sqrtPriceX96'].apply(price_fn)
    
    # Compute volume in USD
    if 'amount1' in df.columns:
        df['amount1'] = pd.to_numeric(df['amount1'], errors='coerce').abs()
        df['volume_usd'] = df['amount1'] / (10**DEC1)
    elif 'amount0' in df.columns:
        df['amount0'] = pd.to_numeric(df['amount0'], errors='coerce').abs()
        df['volume_usd'] = (df['amount0'] / (10**DEC0)) * df['price']
    else:
        df['volume_usd'] = 0.0

    # Resample to hourly OHLCV
    print(f"   Date range: {df['evt_block_time'].min()} → {df['evt_block_time'].max()}")
    df.set_index('evt_block_time', inplace=True)
    
    hourly = df['price'].resample('h').ohlc()
    hourly['volume'] = df['volume_usd'].resample('h').sum()
    
    # Drop empty hours (no swaps) - forward fill close price
    hourly['close'] = hourly['close'].ffill()
    hourly['open'] = hourly['open'].fillna(hourly['close'])
    hourly['high'] = hourly['high'].fillna(hourly['close'])
    hourly['low'] = hourly['low'].fillna(hourly['close'])
    hourly['volume'] = hourly['volume'].fillna(0)
    
    # Rename for feature computation
    hourly.columns = ['open', 'high', 'low', 'close', 'volume']
    
    print(f"   Converted to {len(hourly)} hourly candles.")
    return hourly


def compute_features_from_ohlcv(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the 31 technical features from OHLCV DataFrame.
    Expects columns: open, high, low, close, volume (indexed by hourly timestamp).
    Returns the feature vector for the LAST row.
    """
    df = compute_technical_indicators(df.copy())

    feature_cols = [
        'high_open_ratio', 'low_open_ratio', 'close_open_ratio',
        'dema_ratio', 'momentum_12', 'roc_12', 'atr_14', 'natr_14',
        'adx_14', 'plus_di', 'minus_di', 'cci_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width',
        'stoch_k', 'stoch_d', 'volume_sma_ratio',
        'return_1h', 'return_24h', 'return_7d',
        'price_vs_ma50', 'price_vs_ma200', 'ma50_vs_ma200',
        'market_regime', 'trend_strength_24h', 'trend_strength_7d',
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    last_row = df.iloc[-1]
    features = last_row[feature_cols].values.astype(np.float32)
    # Replace NaN with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features, last_row.name, float(last_row['close'])


def build_observation(tech_features: np.ndarray,
                      has_position: bool = False,
                      position_width: int = 0,
                      position_value_ratio: float = 0.0,
                      in_range: bool = False,
                      hours_since_rebalance: int = 0,
                      unrealized_pnl: float = 0.0,
                      current_fee_rate: float = 0.0) -> np.ndarray:
    """
    Build the full observation vector (38 dims) matching the env.
    31 tech features + 7 position features.
    """
    max_width = 40
    cash_ratio = 1.0 if not has_position else 0.0
    width_normalized = position_width / max_width if max_width > 0 else 0.0
    in_range_val = 1.0 if in_range else 0.0
    hours_norm = min(hours_since_rebalance / 168.0, 1.0)  # normalize by 1 week

    position_features = np.array([
        cash_ratio,
        width_normalized,
        position_value_ratio,
        in_range_val,
        hours_norm,
        unrealized_pnl,
        current_fee_rate,
    ], dtype=np.float32)

    return np.concatenate([tech_features, position_features])


def get_lp_range(price: float, width: int) -> dict:
    """Compute LP range around current price for given width."""
    tick = price_to_tick(price)
    center_tick = (tick // TICK_SPACING) * TICK_SPACING
    lower_tick = center_tick - width * TICK_SPACING
    upper_tick = center_tick + width * TICK_SPACING
    p_lower = tick_to_price(lower_tick)
    p_upper = tick_to_price(upper_tick)
    return {
        "center_tick": center_tick,
        "lower_tick": lower_tick,
        "upper_tick": upper_tick,
        "price_lower": round(p_lower, 2),
        "price_upper": round(p_upper, 2),
        "range_pct": round((p_upper / p_lower - 1) * 100, 2),
    }


def load_model(model_name: str, model_dir: str, device: str = "cpu",
               seq_len: int = 24):
    """Load a trained model."""
    state_dim = 38  # 31 tech + 7 position
    action_dim = 7  # HOLD + 6 widths

    if model_name == "ppo":
        from stable_baselines3 import PPO
        path = os.path.join(model_dir, "comparison_ppo.zip")
        model = PPO.load(path, device=device)
        vec_norm_path = os.path.join(model_dir, "comparison_ppo_vec_normalize.pkl")
        return {"type": "ppo", "model": model,
                "vec_norm_path": vec_norm_path if os.path.exists(vec_norm_path) else None}

    elif model_name == "dqn":
        path = os.path.join(model_dir, "comparison_dqn_best.pth")
        agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim,
                                 device=device)
        agent.load(path)
        return {"type": "dqn", "model": agent}

    elif model_name == "lstm":
        path = os.path.join(model_dir, "comparison_lstm_dqn_best.pth")
        agent = LSTMDDQNAgent(state_dim=state_dim, action_dim=action_dim,
                               seq_len=seq_len, device=device)
        agent.load(path)
        history = SequenceStateWrapper(seq_len=seq_len, state_dim=state_dim)
        return {"type": "lstm", "model": agent, "history": history}

    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_action(loaded_model: dict, obs: np.ndarray) -> int:
    """Get action from loaded model given observation."""
    model_type = loaded_model["type"]

    if model_type == "ppo":
        model = loaded_model["model"]
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        return int(action[0])

    elif model_type == "dqn":
        return loaded_model["model"].select_action(obs, deterministic=True)

    elif model_type == "lstm":
        history = loaded_model["history"]
        history.push(obs)
        seq = history.get_sequence()
        return loaded_model["model"].select_action(seq, deterministic=True)


def decide(args):
    """Main decision function."""
    model_dir = os.path.join(SCRIPT_DIR, "models")

    # ── Load data ──
    if args.swap_csv:
        df = process_swap_csv(args.swap_csv)
    elif args.ohlcv_csv:
        df = pd.read_csv(args.ohlcv_csv, parse_dates=[0], index_col=0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns:
            raise ValueError("CSV must have 'close' column")
    else:
        # Minimal mode
        current_price = args.price
        if current_price is None:
            raise ValueError("Provide --swap-csv, --ohlcv-csv, or --price")
        # dummy df
        dates = pd.date_range(end=datetime.now(tz=timezone.utc), periods=200, freq='h')
        df = pd.DataFrame({
            'open': current_price, 'high': current_price,
            'low': current_price, 'close': current_price,
            'volume': args.volume or 5_000_000,
        }, index=dates)
        print("⚠️  Using price-only mode. Technical indicators will be zeros.")

    # ── Compute features ──
    tech_features, last_ts, current_price = compute_features_from_ohlcv(df)

    # ── Build observation ──
    obs = build_observation(
        tech_features=tech_features,
        has_position=args.has_position,
        position_width=args.current_width or 0,
        position_value_ratio=1.0 if args.has_position else 0.0,
        in_range=args.in_range,
        hours_since_rebalance=args.hours_since_rebalance or 0,
    )

    # ── Load model & predict ──
    loaded_model = load_model(args.model, model_dir, args.device, args.seq_len)
    action = predict_action(loaded_model, obs)

    # ── Format output ──
    action_info = ACTIONS[action]
    result = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "current_price": current_price,
        "model": args.model,
        "action": action,
        "action_label": action_info["label"],
        "recommendation": "",
    }

    if action == 0:
        result["recommendation"] = "HOLD current position. No rebalance needed."
    else:
        width = action_info["width"]
        lp_range = get_lp_range(current_price, width)
        result["recommendation"] = (
            f"Set LP range: ${lp_range['price_lower']:,.2f} – "
            f"${lp_range['price_upper']:,.2f} "
            f"(±{action_info['range_pct']}%)"
        )
        result["lp_range"] = lp_range

    # ── Print ──
    print(f"\n{'=' * 50}")
    print(f"  🤖 Kongtrae Decision")
    print(f"{'=' * 50}")
    print(f"  Model:   {args.model.upper()}")
    print(f"  Price:   ${current_price:,.2f}")
    print(f"  Action:  {result['action_label']}")
    print(f"  →  {result['recommendation']}")
    if "lp_range" in result:
        r = result["lp_range"]
        print(f"  Range:   [{r['lower_tick']}, {r['upper_tick']}]")
        print(f"  Width:   ±{action_info['range_pct']}%")
    print(f"{'=' * 50}\n")

    # ── Save JSON ──
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  📄 Saved to {args.output}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Kongtrae – Get LP decision for the current hour"
    )

    # Model selection
    parser.add_argument("--model", required=True, choices=["ppo", "dqn", "lstm"],
                        help="Which model to use")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seq-len", type=int, default=24, help="LSTM seq len")

    # Market data (option A: CSV with raw swaps)
    parser.add_argument("--swap-csv", default=None,
                        help="CSV with raw swaps (evt_block_time, sqrtPriceX96, amount0/1). "
                             "Will be resampled to hourly candles.")

    # Market data (option B: CSV with OHLCV history)
    parser.add_argument("--ohlcv-csv", default=None,
                        help="CSV with columns: timestamp, open, high, low, close, volume "
                             "(at least 200 hourly rows for accurate indicators)")

    # Market data (option B: just current price)
    parser.add_argument("--price", type=float, default=None,
                        help="Current ETH price in USD (simpler but less accurate)")
    parser.add_argument("--volume", type=float, default=5_000_000,
                        help="Hourly volume in USD (default: 5M)")

    # Current position state
    parser.add_argument("--has-position", action="store_true",
                        help="Whether you currently have an LP position")
    parser.add_argument("--current-width", type=int, default=None,
                        help="Current LP width (1,3,5,10,20,40)")
    parser.add_argument("--in-range", action="store_true",
                        help="Whether current position is in range")
    parser.add_argument("--hours-since-rebalance", type=int, default=0,
                        help="Hours since last rebalance")

    # Output
    parser.add_argument("--output", "-o", default=None,
                        help="Save decision to JSON file")

    args = parser.parse_args()
    decide(args)


if __name__ == "__main__":
    main()
