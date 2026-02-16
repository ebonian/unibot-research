"""
Compute realistic fees using the official Uniswap V3 fee formula.

From UniswapV3Pool.sol:
  feeGrowthGlobal += feeAmount * Q128 / liquidity

Each LP's fee per swap = swap_fee × (LP_liquidity / pool_total_liquidity)

This script:
1. Replays the model's decisions during the test period
2. For each real swap, computes the agent's on-chain L for its current position
3. Computes: agent_fee = swap_fee_amount × (agent_L / pool_L)
"""

import os
import sys
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(SCRIPT_DIR, "..", "research", "simulation_12")
sys.path.insert(0, SIM_DIR)

from uniswap_v3_dqn_paper import (
    prepare_hourly_data_extended,
    UniswapV3DQNEnv,
    DuelingDDQNAgent,
    LSTMDDQNAgent,
    SequenceStateWrapper,
    price_to_tick,
)

# ─── Pool Config ──────────────────────────────────────────────────────────────
POOL_FEE = 500  # fee in 1/1_000_000
DEC0 = 18  # WETH
DEC1 = 6   # USDT
TICK_SPACING = 10
Q96 = 2**96


def sqrt_price_x96_to_price(sqrt_price_x96: int) -> float:
    """Convert sqrtPriceX96 to USD price."""
    p = float(sqrt_price_x96) / Q96
    return (p * p) * (10 ** (DEC0 - DEC1))


def compute_agent_L_onchain(capital_usd: float, price_usd: float,
                            center_tick: int, width: int) -> float:
    """
    Compute agent's liquidity L in ON-CHAIN units (same as pool's liquidity).
    
    On-chain:
      amount0 = L × (1/√p - 1/√p_upper)   [in wei, 10^18]
      amount1 = L × (√p - √p_lower)        [in micro-USDT, 10^6]
      
    Total USD = amount0/10^18 × price_usd + amount1/10^6
    
    So: L_onchain = capital_usd / value_per_L_usd
    """
    lower_tick = center_tick - width * TICK_SPACING
    upper_tick = center_tick + width * TICK_SPACING

    # On-chain sqrt prices (raw units: sqrt(token1_smallest / token0_smallest))
    # √p = √(1.0001^tick) but we need to use the log trick for large ticks
    sqrt_p_raw = math.exp(0.5 * center_tick * math.log(1.0001))
    # But this is problematic for very negative ticks... 
    # Instead, use price_usd to derive sqrt_p_raw:
    # price_raw = price_usd / 10^12
    # sqrt_p_raw = sqrt(price_raw)
    price_raw = price_usd / (10 ** (DEC0 - DEC1))
    sqrt_p_raw = math.sqrt(price_raw)

    p_lower_raw = price_raw * math.exp((lower_tick - center_tick) * math.log(1.0001))
    p_upper_raw = price_raw * math.exp((upper_tick - center_tick) * math.log(1.0001))
    sqrt_pl_raw = math.sqrt(p_lower_raw)
    sqrt_pu_raw = math.sqrt(p_upper_raw)

    # Value per unit of on-chain L in USD
    if price_raw <= p_lower_raw:
        # All in token0 (ETH)
        value_per_L = (1.0/sqrt_pl_raw - 1.0/sqrt_pu_raw) / (10**DEC0) * price_usd
    elif price_raw >= p_upper_raw:
        # All in token1 (USDT)
        value_per_L = (sqrt_pu_raw - sqrt_pl_raw) / (10**DEC1)
    else:
        # In range: mix of both tokens
        value_per_L = (
            (1.0/sqrt_p_raw - 1.0/sqrt_pu_raw) / (10**DEC0) * price_usd +
            (sqrt_p_raw - sqrt_pl_raw) / (10**DEC1)
        )

    if value_per_L <= 0:
        return 0.0

    return capital_usd / value_per_L


def run_fee_calculation(data_dir: str, swap_file: str, model_name: str,
                        capital: float, device: str = "cpu"):
    """
    Replay model decisions over test period and compute real fees per swap.
    """
    # ── Load env + model ──
    print("📊 Loading hourly data...")
    hourly_data = prepare_hourly_data_extended(data_dir)

    env = UniswapV3DQNEnv(hourly_data, initial_capital_usd=capital, mode="test")
    state_dim = env.state_dim
    action_dim = env.action_space.n

    model_dir = os.path.join(SCRIPT_DIR, "models")

    if model_name == "ppo":
        from stable_baselines3 import PPO
        from uniswap_v3_ppo_paper import make_env_fn
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        ppo = PPO.load(os.path.join(model_dir, "comparison_ppo.zip"), device=device)
        eval_fn = make_env_fn(hourly_data, initial_capital_usd=capital, mode="test")
        vec_env = DummyVecEnv([eval_fn])
        vec_norm_path = os.path.join(model_dir, "comparison_ppo_vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            vec_env = VecNormalize.load(vec_norm_path, vec_env)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        vec_env.training = False
        vec_env.norm_reward = False
    elif model_name == "dqn":
        agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
        agent.load(os.path.join(model_dir, "comparison_dqn_best.pth"))
    elif model_name == "lstm":
        seq_len = 24
        agent = LSTMDDQNAgent(state_dim=state_dim, action_dim=action_dim,
                               seq_len=seq_len, device=device)
        agent.load(os.path.join(model_dir, "comparison_lstm_dqn_best.pth"))
        history = SequenceStateWrapper(seq_len=seq_len, state_dim=state_dim)

    # ── Load raw swap data ──
    print("📊 Loading swap data...")
    swaps = pd.read_csv(swap_file, low_memory=False)
    swaps['evt_block_time'] = pd.to_datetime(swaps['evt_block_time'], utc=True)
    swaps = swaps.sort_values('evt_block_time').reset_index(drop=True)
    swaps['liquidity'] = pd.to_numeric(swaps['liquidity'], errors='coerce')
    swaps['amount0'] = pd.to_numeric(swaps['amount0'], errors='coerce')
    swaps['amount1'] = pd.to_numeric(swaps['amount1'], errors='coerce')
    swaps['tick'] = pd.to_numeric(swaps['tick'], errors='coerce')
    swaps['sqrtPriceX96'] = pd.to_numeric(swaps['sqrtPriceX96'], errors='coerce')

    # ── Replay model decisions ──
    print(f"🤖 Replaying {model_name.upper()} model decisions over test period...")
    obs, _ = env.reset()
    if model_name == "ppo":
        vec_obs = vec_env.reset()

    # Map of test timestamps
    test_timestamps = env.timestamps
    print(f"   Test period: {test_timestamps[0]} → {test_timestamps[-1]}")
    print(f"   Test hours: {len(test_timestamps)}")

    # For each hour: store the model's decision (position width + center tick)
    hourly_decisions = []
    done = False
    step = 0

    while not done:
        ts = env.timestamps[env.idx]
        price = env.hourly_data.prices[ts]

        if model_name == "ppo":
            action, _ = ppo.predict(vec_obs, deterministic=True)
            action_int = int(action[0])
        elif model_name == "dqn":
            action_int = agent.select_action(obs, deterministic=True)
        elif model_name == "lstm":
            history.push(obs)
            seq = history.get_sequence()
            action_int = agent.select_action(seq, deterministic=True)

        # Record position BEFORE stepping (this is what the agent holds this hour)
        has_pos = env.has_position
        pos_width = env.position_width
        pos_center = env.position_center_tick

        obs, reward, done, trunc, info = env.step(action_int)
        if model_name == "ppo":
            vec_obs, _, _, _ = vec_env.step(action)

        # After step, the agent may have a new position
        # Convert the agent's position to on-chain tick space
        agent_center_tick_sim = env.position_center_tick
        agent_width = env.position_width

        # The simulation uses tick_to_price(tick) = 1.0001^tick where tick = log(price_usd)/log(1.0001)
        # On-chain uses: tick = log(price_raw)/log(1.0001) where price_raw = price_usd / 10^12
        # Conversion: tick_onchain = tick_sim + log(10^(-12))/log(1.0001)
        # = tick_sim - 12*log(10)/log(1.0001)
        # = tick_sim - 276324 (approximately)
        TICK_OFFSET = int(round((DEC0 - DEC1) * math.log(10) / math.log(1.0001)))

        if env.has_position and agent_width > 0:
            agent_center_onchain = agent_center_tick_sim - TICK_OFFSET
            agent_lower_onchain = agent_center_onchain - agent_width * TICK_SPACING
            agent_upper_onchain = agent_center_onchain + agent_width * TICK_SPACING
        else:
            agent_center_onchain = 0
            agent_lower_onchain = 0
            agent_upper_onchain = 0

        hourly_decisions.append({
            "timestamp": ts,
            "price": price,
            "action": action_int,
            "has_position": env.has_position,
            "position_width": agent_width,
            "center_tick_sim": agent_center_tick_sim,
            "center_tick_onchain": agent_center_onchain,
            "lower_tick_onchain": agent_lower_onchain,
            "upper_tick_onchain": agent_upper_onchain,
            "sim_reward": reward,
        })
        step += 1
        done = done or trunc

    print(f"   Total steps: {step}")
    sim_total_reward = sum(d["sim_reward"] for d in hourly_decisions)
    print(f"   Sim total reward: ${sim_total_reward:.2f}")

    # ── Compute real fees per swap ──
    print("\n💰 Computing real fees from Uniswap V3 formula...")
    print("   Fee formula: agent_fee = swap_fee × (agent_L / pool_L)")
    print("   Source: UniswapV3Pool.sol line 'feeGrowthGlobal += feeAmount * Q128 / liquidity'")

    # Build hourly index from decisions
    hourly_index = {}
    for d in hourly_decisions:
        hourly_index[d["timestamp"]] = d

    # Filter swaps to test period
    test_start = test_timestamps[0]
    test_end = test_timestamps[-1] + pd.Timedelta(hours=1)
    test_swaps = swaps[(swaps['evt_block_time'] >= test_start) &
                       (swaps['evt_block_time'] < test_end)].copy()
    print(f"   Swaps in test period: {len(test_swaps):,}")

    total_agent_fee_usd = 0.0
    total_pool_fee_usd = 0.0
    swaps_in_range = 0
    swaps_out_range = 0

    for _, swap in test_swaps.iterrows():
        swap_time = swap['evt_block_time']
        swap_hour = swap_time.floor('h')

        decision = hourly_index.get(swap_hour)
        if decision is None or not decision["has_position"]:
            continue

        width = decision["position_width"]
        center_tick_onchain = decision["center_tick_onchain"]
        lower_tick_onchain = decision["lower_tick_onchain"]
        upper_tick_onchain = decision["upper_tick_onchain"]
        swap_tick = int(swap['tick'])
        pool_L = float(swap['liquidity'])

        if pool_L <= 0:
            continue

        # Check if swap is within agent's LP range (on-chain tick space)
        if swap_tick < lower_tick_onchain or swap_tick > upper_tick_onchain:
            swaps_out_range += 1
            continue

        swaps_in_range += 1

        # Compute swap fee (from Uniswap V3: fee is on the input token)
        # For exactInput: fee = amountIn × pool_fee / (1_000_000 - pool_fee)
        # Or simply: fee ≈ |amount_in| × pool_fee / 1_000_000
        # amount0 < 0 means token0 goes out (trader buys ETH), amount1 > 0 is input
        # amount0 > 0 means token0 goes in (trader sells ETH), amount0 is input
        amount0 = float(swap['amount0'])
        amount1 = float(swap['amount1'])

        # Determine which token was input
        if amount0 > 0:
            # Token0 (ETH) was input, fee is in token0
            fee_amount_raw = abs(amount0) * POOL_FEE / (1_000_000 - POOL_FEE)
            price_usd = sqrt_price_x96_to_price(int(swap['sqrtPriceX96']))
            fee_usd = fee_amount_raw / (10**DEC0) * price_usd
        else:
            # Token1 (USDT) was input, fee is in token1
            fee_amount_raw = abs(amount1) * POOL_FEE / (1_000_000 - POOL_FEE)
            fee_usd = fee_amount_raw / (10**DEC1)

        total_pool_fee_usd += fee_usd

        # Compute agent's on-chain L
        swap_price_usd = sqrt_price_x96_to_price(int(swap['sqrtPriceX96']))
        agent_L = compute_agent_L_onchain(capital, swap_price_usd, center_tick_onchain, width)

        if agent_L <= 0:
            continue

        # Agent's fee share (from UniswapV3Pool.sol)
        agent_fee = fee_usd * (agent_L / pool_L)
        total_agent_fee_usd += agent_fee

    # ── Results ──
    test_hours = len(hourly_decisions)
    test_days = test_hours / 24

    print(f"\n{'=' * 60}")
    print(f"  RESULTS – {model_name.upper()} model with ${capital} capital")
    print(f"{'=' * 60}")
    print(f"  Pool:             ETH/USDT 0.05% (Ethereum Mainnet)")
    print(f"  Test period:      {test_timestamps[0].date()} → {test_timestamps[-1].date()}")
    print(f"  Test duration:    {test_hours} hours ({test_days:.0f} days)")
    print(f"")
    print(f"  Total swaps:      {len(test_swaps):,}")
    print(f"  Swaps in range:   {swaps_in_range:,} ({swaps_in_range/max(len(test_swaps),1)*100:.1f}%)")
    print(f"  Swaps out range:  {swaps_out_range:,}")
    print(f"")
    print(f"  Total pool fees:  ${total_pool_fee_usd:,.2f}")
    print(f"  Agent fees:       ${total_agent_fee_usd:.4f}")
    print(f"  Agent fee share:  {total_agent_fee_usd/max(total_pool_fee_usd,1e-10)*100:.6f}%")
    print(f"")
    print(f"  Agent fee/hour:   ${total_agent_fee_usd/test_hours:.6f}")
    print(f"  Agent fee/day:    ${total_agent_fee_usd/test_days:.4f}")
    print(f"  Agent fee/month:  ${total_agent_fee_usd/test_days*30:.4f}")
    print(f"  Monthly return:   {total_agent_fee_usd/test_days*30/capital*100:.4f}%")
    print(f"  APR:              {total_agent_fee_usd/test_days*365/capital*100:.2f}%")
    print(f"")
    print(f"  Sim total reward: ${sim_total_reward:.2f}")
    print(f"  Real/Sim ratio:   {total_agent_fee_usd/max(sim_total_reward,1e-10):.6f}")
    print(f"  → fee_share:      {total_agent_fee_usd/max(sim_total_reward,1e-10):.6f}")
    print(f"{'=' * 60}")

    return {
        "model": model_name,
        "capital": capital,
        "test_hours": test_hours,
        "total_pool_fees": total_pool_fee_usd,
        "agent_fees": total_agent_fee_usd,
        "sim_reward": sim_total_reward,
        "fee_share": total_agent_fee_usd / max(sim_total_reward, 1e-10),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute real Uniswap V3 fees using official fee formula"
    )
    parser.add_argument("--data-dir",
                        default=os.path.join(SIM_DIR, "training_data"),
                        help="Path to hourly data directory")
    parser.add_argument("--swap-file", default=None,
                        help="Path to raw swap CSV (auto-detected if not set)")
    parser.add_argument("--model", default="dqn", choices=["ppo", "dqn", "lstm"],
                        help="Model to replay")
    parser.add_argument("--capital", type=float, default=100.0,
                        help="Agent capital in USD")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.swap_file is None:
        import glob
        files = glob.glob(os.path.join(args.data_dir, "swaps_*_eth_usdt_0p3.csv"))
        if not files:
            raise FileNotFoundError("No swap file found")
        args.swap_file = files[0]

    run_fee_calculation(args.data_dir, args.swap_file, args.model,
                       args.capital, args.device)


if __name__ == "__main__":
    main()
