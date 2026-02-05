#!/usr/bin/env python3
"""
verify_optimized_env.py

Sanity-check the optimized PPO env before long training:
- Precomputation runs without error
- Obs/reward are finite, no NaNs
- Action space is respected
- Episode runs to completion
- Capital stays non-negative
"""

import os
import sys
import numpy as np

# Add parent so we can run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uniswap_v3_ppo_optimized import (
    precompute_training_data,
    UniswapV3FastEnv,
    make_fast_env_fn,
)


def main():
    # Must use training_data inside simulation_6 (prepare_training_data.py outputs here)
    data_dir = os.environ.get(
        "DATA_DIR",
        os.path.join(os.path.dirname(__file__), "training_data"),
    )
    if not os.path.isdir(data_dir):
        print(f"❌ Data dir not found: {data_dir}")
        print("   Run scripts/prepare_training_data.py first to create simulation_6/training_data/")
        return 1

    print("=" * 60)
    print("🔍 Verifying optimized PPO environment")
    print("=" * 60)
    print(f"  Data dir: {data_dir}\n")

    errors = []

    # 1. Precomputation
    print("1️⃣ Precomputing training data...")
    try:
        precomputed = precompute_training_data(data_dir, window_hours=1)
    except Exception as e:
        errors.append(f"Precomputation failed: {e}")
        print(f"   ❌ {e}\n")
    else:
        n_win = len(precomputed.windows)
        n_price = len(precomputed.prices)
        n_vol = len(precomputed.volatilities)
        n_fee = len(precomputed.fee_rates)
        if n_price != n_win or n_vol != n_win or n_fee != n_win:
            errors.append(f"Precompute length mismatch: windows={n_win} prices={n_price} vol={n_vol} fee={n_fee}")
        else:
            print(f"   ✅ {n_win} windows, prices/vol/fee OK\n")

    if errors:
        for e in errors:
            print(f"❌ {e}")
        return 1

    # 2. Env creation and one full episode
    print("2️⃣ Running one full episode (train mode)...")
    env = UniswapV3FastEnv(
        precomputed=precomputed,
        total_usd=1000.0,
        gas_per_action_usd=0.1,
        mode="train",
    )

    obs, info = env.reset(seed=42)
    step = 0
    total_reward = 0.0
    last_capital = 1000.0

    while True:
        # Check obs
        if not np.all(np.isfinite(obs)):
            errors.append(f"Step {step}: non-finite obs {obs}")
        if obs.shape != (7,):
            errors.append(f"Step {step}: wrong obs shape {obs.shape}")
        if not env.observation_space.contains(obs):
            # Allow slight float tolerance
            if np.any(obs < env.observation_space.low - 1e-5) or np.any(obs > env.observation_space.high + 1e-5):
                errors.append(f"Step {step}: obs out of bounds (low={env.observation_space.low}, high={env.observation_space.high})")

        # Random action in [-1, 1]
        action = np.array(
            [np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
            dtype=np.float32,
        )
        if not env.action_space.contains(action):
            errors.append(f"Step {step}: action {action} not in action_space")

        obs, reward, terminated, truncated, info = env.step(action)

        if not np.isfinite(reward):
            errors.append(f"Step {step}: non-finite reward {reward}")
        total_reward += reward

        cap = info.get("current_capital", last_capital)
        if cap < -1e-6:
            errors.append(f"Step {step}: negative capital {cap}")
        last_capital = cap

        step += 1
        if terminated or truncated:
            break

    print(f"   Episode finished: {step} steps, total_reward={total_reward:.4f}, final_capital={last_capital:.2f}")
    if step != env.n_windows:
        errors.append(f"Episode length {step} != n_windows {env.n_windows}")
    if last_capital < 0:
        errors.append("Final capital is negative")
    if not errors:
        print("   ✅ Episode and obs/reward/capital checks OK\n")
    else:
        for e in errors:
            print(f"   ❌ {e}\n")

    # 3. Eval env
    print("3️⃣ Eval env (different windows)...")
    eval_env = UniswapV3FastEnv(
        precomputed=precomputed,
        total_usd=1000.0,
        mode="eval",
    )
    obs_e, _ = eval_env.reset(seed=123)
    if obs_e.shape != (7,) or not np.all(np.isfinite(obs_e)):
        errors.append("Eval env reset obs invalid")
    else:
        print(f"   Eval n_windows={eval_env.n_windows}")
        print("   ✅ Eval env OK\n")

    # Summary
    print("=" * 60)
    if errors:
        print("❌ VERIFICATION FAILED")
        for e in errors:
            print(f"   - {e}")
        return 1
    print("✅ All checks passed. Safe to run 100k timesteps.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
