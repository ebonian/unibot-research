#!/usr/bin/env python3
"""
Compare PPO vs Dueling DDQN for Uniswap v3 LP.

This script trains and evaluates both algorithms, producing comparison results
for thesis evaluation.

Usage:
    python compare_algorithms.py --data-dir ../simulation_6/training_data
"""

import os
import sys
import argparse
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run_ppo_training(data_dir: str, timesteps: int = 50000, num_envs: int = 4, save_prefix: str = "models"):
    """Train PPO model."""
    print("\n" + "=" * 60, flush=True)
    print("  Training PPO (Xu & Brini 2025 methodology)", flush=True)
    print("=" * 60, flush=True)
    
    from uniswap_v3_ppo_paper import train_paper_method
    
    model, env = train_paper_method(
        data_dir=data_dir,
        num_envs=num_envs,
        total_timesteps=timesteps,
        save_path=f"{save_prefix}/comparison_ppo",
    )
    return model


def run_dqn_training(data_dir: str, episodes: int = 200, device: str = "cpu", save_prefix: str = "models"):
    """Train Dueling DDQN model."""
    print("\n" + "=" * 60, flush=True)
    print("  Training Dueling DDQN (Zhang et al. 2023 methodology)", flush=True)
    print("=" * 60, flush=True)
    
    from uniswap_v3_dqn_paper import train_dqn
    
    rewards = train_dqn(
        data_dir=data_dir,
        n_episodes=episodes,
        save_path=f"{save_prefix}/comparison_dqn",
        device=device,
        n_steps=3,  # Added n_steps
    )
    return rewards


def run_lstm_dqn_training(data_dir: str, episodes: int = 1000, seq_len: int = 24, device: str = "cpu", save_prefix: str = "models"):
    """Train LSTM-based Dueling DDQN model."""
    print("\n" + "=" * 60, flush=True)
    print("  Training LSTM Dueling DDQN (Sequence-aware)", flush=True)
    print("=" * 60, flush=True)
    
    from uniswap_v3_dqn_paper import train_lstm_dqn
    
    agent = train_lstm_dqn(
        data_dir=data_dir,
        n_episodes=episodes,
        seq_len=seq_len,
        save_path=f"{save_prefix}/comparison_lstm_dqn",
        device=device,
    )
    return agent


def evaluate_ppo(data_dir: str, n_episodes: int = 10, save_prefix: str = "models"):
    """Evaluate PPO on test set."""
    print("\n" + "=" * 60, flush=True)
    print("  Evaluating PPO on test set", flush=True)
    print("=" * 60, flush=True)
    
    from uniswap_v3_ppo_paper import evaluate_paper_method
    
    results = evaluate_paper_method(
        data_dir=data_dir,
        model_path=f"{save_prefix}/comparison_ppo.zip",
        n_episodes=n_episodes,
    )
    return results


def evaluate_lstm_dqn(data_dir: str, n_episodes: int = 10, seq_len: int = 24, device: str = "cpu", save_prefix: str = "models"):
    """Evaluate LSTM Dueling DDQN on test set."""
    print("\n" + "=" * 60, flush=True)
    print("  Evaluating LSTM Dueling DDQN on test set", flush=True)
    print("=" * 60, flush=True)
    
    from uniswap_v3_dqn_paper import (
        prepare_hourly_data_extended, 
        UniswapV3DQNEnv, 
        LSTMDDQNAgent, 
        evaluate_lstm_agent
    )
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    test_env = UniswapV3DQNEnv(hourly_data, mode="test")
    
    agent = LSTMDDQNAgent(
        state_dim=test_env.state_dim,
        action_dim=test_env.action_space.n,
        seq_len=seq_len,
        device=device,
    )
    
    model_path = f"{save_prefix}/comparison_lstm_dqn_best.pth"
    if not os.path.exists(model_path):
        model_path = f"{save_prefix}/comparison_lstm_dqn_final.pth"
    
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration
    
    # Evaluate
    rewards = []
    action_counts = {i: 0 for i in range(test_env.action_space.n)}
    
    for ep in range(n_episodes):
        state, _ = test_env.reset()
        agent.reset_sequence()
        agent.update_sequence(state)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            state_seq = agent.get_current_sequence()
            action = agent.select_action(state_seq, deterministic=True)
            action_counts[action] += 1
            
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            agent.update_sequence(next_state)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    # Print action distribution
    print("\nAction distribution:")
    total_actions = sum(action_counts.values())
    for action_id, count in sorted(action_counts.items()):
        if count > 0:
            width = test_env.action_map.get(action_id, None)
            if width is None:
                action_name = "HOLD"
            else:
                action_name = f"w={width}"
            print(f"  {action_name:20s}: {count:5d} ({100*count/total_actions:.1f}%)")
    
    print(f"\nResults:")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "action_distribution": {str(k): v for k, v in action_counts.items()},
    }


def evaluate_dqn(data_dir: str, n_episodes: int = 10, device: str = "cpu", save_prefix: str = "models"):
    """Evaluate Dueling DDQN on test set."""
    print("\n" + "=" * 60, flush=True)
    print("  Evaluating Dueling DDQN on test set", flush=True)
    print("=" * 60, flush=True)
    
    from uniswap_v3_dqn_paper import evaluate_on_test
    
    model_path = f"{save_prefix}/comparison_dqn_best.pth"
    if not os.path.exists(model_path):
        model_path = f"{save_prefix}/comparison_dqn_final.pth"
        
    results = evaluate_on_test(
        data_dir=data_dir,
        model_path=model_path,
        n_episodes=n_episodes,
        device=device,
    )
    return results


def run_baseline_hold(data_dir: str, n_episodes: int = 10):
    """Run baseline: always hold (no LP position)."""
    print("\n" + "=" * 60)
    print("📊 Baseline: HOLD (no LP)")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import prepare_hourly_data_extended, UniswapV3DQNEnv
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    env = UniswapV3DQNEnv(hourly_data, mode="test")
    
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = 0  # Always hold
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return {"mean_reward": mean_reward, "std_reward": std_reward}


def run_baseline_fixed_width(data_dir: str, width: int = 5, n_episodes: int = 10):
    """Run baseline: fixed width LP, centered, reallocate every step."""
    print("\n" + "=" * 60)
    print(f"📊 Baseline: Fixed Width={width} (centered, always reallocate)")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import prepare_hourly_data_extended, UniswapV3DQNEnv
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    env = UniswapV3DQNEnv(hourly_data, mode="test")
    
    # Find action ID for this width
    # Action 0 = HOLD, Action 1-5 = widths [1,2,5,8,10]
    widths = [1, 2, 5, 8, 10]
    
    if width not in widths:
        width = min(widths, key=lambda w: abs(w - width))
        print(f"  (Using closest available width: {width})")
    
    action_id = 1 + widths.index(width)
    
    print(f"  Action mapping: width={width} → action_id={action_id}")
    
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = action_id  # Fixed width, centered
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return {"mean_reward": mean_reward, "std_reward": std_reward}


def visualize_ppo_decisions(data_dir: str, model_path: str = "models/comparison_ppo.zip"):
    """Visualize PPO agent decisions on test set."""
    print("\n" + "=" * 60)
    print("📊 Visualizing PPO Decisions on Test Set")
    print("=" * 60)
    
    from uniswap_v3_ppo_paper import UniswapV3PaperEnv, tick_to_price, make_env_fn
    from uniswap_v3_dqn_paper import prepare_hourly_data_extended
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    
    # Must use VecNormalize wrapper - PPO was trained with normalized observations
    eval_fn = make_env_fn(hourly_data, mode="test")
    vec_env = DummyVecEnv([eval_fn])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Load VecNormalize stats if available
    vec_normalize_path = model_path.replace(".zip", "_vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    # Also need the raw env to access internal state
    raw_env = UniswapV3PaperEnv(hourly_data, mode="test")
    
    model = PPO.load(model_path, env=vec_env)
    
    # Collect trajectory
    timestamps = []
    prices = []
    actions = []
    rewards = []
    cumulative_rewards = []
    lp_windows = []  # List of (start_time, end_time, lower_price, upper_price)
    
    # Reset both environments
    obs = vec_env.reset()
    raw_env.reset()
    
    done = False
    cum_reward = 0.0
    
    current_window_start = None
    current_lower = None
    current_upper = None
    
    while not done:
        t = raw_env.timestamps[raw_env.idx]
        price = raw_env._get_price(t)
        
        # Check LP state before action
        old_lower = raw_env.lp_lower_tick
        old_upper = raw_env.lp_upper_tick
        
        # Use normalized observations for prediction
        action, _ = model.predict(obs, deterministic=True)
        
        # Step both environments
        obs, reward, terminated, truncated = vec_env.step(action)[:4]
        raw_env.step(int(action[0]))  # Keep raw env in sync
        
        done = bool(terminated[0]) if hasattr(terminated, '__len__') else bool(terminated)
        
        timestamps.append(t)
        prices.append(price)
        actions.append(int(action[0]))
        rewards.append(float(reward[0]) if hasattr(reward, '__len__') else float(reward))
        cum_reward += rewards[-1]
        cumulative_rewards.append(cum_reward)
        
        # Track LP windows using raw env state
        if raw_env.has_lp:
            lower_price = tick_to_price(raw_env.lp_lower_tick)
            upper_price = tick_to_price(raw_env.lp_upper_tick)
            
            # New position or position changed
            if current_window_start is None or (raw_env.lp_lower_tick != old_lower or raw_env.lp_upper_tick != old_upper):
                # Close previous window if exists
                if current_window_start is not None:
                    lp_windows.append((current_window_start, t, current_lower, current_upper))
                current_window_start = t
                current_lower = lower_price
                current_upper = upper_price
        else:
            # Position closed
            if current_window_start is not None:
                lp_windows.append((current_window_start, t, current_lower, current_upper))
                current_window_start = None
    
    # Close final window if still open
    if current_window_start is not None:
        lp_windows.append((current_window_start, timestamps[-1], current_lower, current_upper))
    
    # Diagnostic
    cum = cumulative_rewards
    if cum:
        print(f"  PPO trajectory: min cumulative = {min(cum):.0f}, final = {cum[-1]:.0f}", flush=True)
    
    return {
        "timestamps": timestamps,
        "prices": prices,
        "actions": actions,
        "rewards": rewards,
        "cumulative_rewards": cumulative_rewards,
        "action_ticks": raw_env.action_ticks,
        "lp_windows": lp_windows,
    }


def visualize_dqn_decisions(data_dir: str, model_path: str = "models/comparison_dqn_best.pth", device: str = "cpu"):
    """Visualize DQN agent decisions on test set."""
    print("\n" + "=" * 60)
    print("📊 Visualizing DQN Decisions on Test Set")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import prepare_hourly_data_extended, UniswapV3DQNEnv, DuelingDDQNAgent, tick_to_price
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    env = UniswapV3DQNEnv(hourly_data, mode="test")
    
    agent = DuelingDDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        device=device,
    )
    agent.load(model_path)
    agent.epsilon = 0.0
    
    # Collect trajectory
    timestamps = []
    prices = []
    actions = []
    rewards = []
    cumulative_rewards = []
    lp_windows = []  # List of (start_time, end_time, lower_price, upper_price)
    
    state, _ = env.reset()
    done = False
    cum_reward = 0.0
    
    current_window_start = None
    current_lower = None
    current_upper = None
    prev_center = None
    prev_width = None
    
    while not done:
        t = env.timestamps[env.idx]
        price = env._get_price(t)
        
        # Check LP state before action
        had_position = env.has_position
        old_center = env.position_center_tick
        old_width = env.position_width
        
        action = agent.select_action(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        timestamps.append(t)
        prices.append(price)
        actions.append(action)
        rewards.append(reward)
        cum_reward += reward
        cumulative_rewards.append(cum_reward)
        
        # Track LP windows
        if env.has_position:
            lower_tick = env.position_center_tick - env.position_width * env.tick_spacing
            upper_tick = env.position_center_tick + env.position_width * env.tick_spacing
            lower_price = tick_to_price(lower_tick)
            upper_price = tick_to_price(upper_tick)
            
            # New position or position changed
            if current_window_start is None or (env.position_center_tick != old_center or env.position_width != old_width):
                # Close previous window if exists
                if current_window_start is not None:
                    lp_windows.append((current_window_start, t, current_lower, current_upper))
                current_window_start = t
                current_lower = lower_price
                current_upper = upper_price
        else:
            # Position closed
            if current_window_start is not None:
                lp_windows.append((current_window_start, t, current_lower, current_upper))
                current_window_start = None
    
    # Close final window if still open
    if current_window_start is not None:
        lp_windows.append((current_window_start, timestamps[-1], current_lower, current_upper))
    
    return {
        "timestamps": timestamps,
        "prices": prices,
        "actions": actions,
        "rewards": rewards,
        "cumulative_rewards": cumulative_rewards,
        "max_width": env.max_width,
        "lp_windows": lp_windows,
        "action_map": env.action_map,  # For interpreting actions
    }


def visualize_lstm_decisions(data_dir: str, model_path: str = "models/comparison_lstm_dqn_best.pth", 
                              seq_len: int = 24, device: str = "cpu"):
    """Visualize LSTM DQN agent decisions on test set."""
    print("\n" + "=" * 60)
    print("📊 Visualizing LSTM DQN Decisions on Test Set")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import prepare_hourly_data_extended, UniswapV3DQNEnv, LSTMDDQNAgent, tick_to_price
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    env = UniswapV3DQNEnv(hourly_data, mode="test")
    
    agent = LSTMDDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        seq_len=seq_len,
        device=device,
    )
    agent.load(model_path)
    agent.epsilon = 0.0
    
    # Collect trajectory
    timestamps = []
    prices = []
    actions = []
    rewards = []
    cumulative_rewards = []
    lp_windows = []
    
    state, _ = env.reset()
    agent.reset_sequence()
    agent.update_sequence(state)
    
    done = False
    cum_reward = 0.0
    
    current_window_start = None
    current_lower = None
    current_upper = None
    
    while not done:
        t = env.timestamps[env.idx]
        price = env._get_price(t)
        
        # Check LP state before action
        old_center = env.position_center_tick
        old_width = env.position_width
        
        # LSTM uses sequence
        state_seq = agent.get_current_sequence()
        action = agent.select_action(state_seq, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update sequence
        agent.update_sequence(state)
        
        timestamps.append(t)
        prices.append(price)
        actions.append(action)
        rewards.append(reward)
        cum_reward += reward
        cumulative_rewards.append(cum_reward)
        
        # Track LP windows
        if env.has_position:
            lower_tick = env.position_center_tick - env.position_width * env.tick_spacing
            upper_tick = env.position_center_tick + env.position_width * env.tick_spacing
            lower_price = tick_to_price(lower_tick)
            upper_price = tick_to_price(upper_tick)
            
            if current_window_start is None or (env.position_center_tick != old_center or env.position_width != old_width):
                if current_window_start is not None:
                    lp_windows.append((current_window_start, t, current_lower, current_upper))
                current_window_start = t
                current_lower = lower_price
                current_upper = upper_price
        else:
            if current_window_start is not None:
                lp_windows.append((current_window_start, t, current_lower, current_upper))
                current_window_start = None
    
    if current_window_start is not None:
        lp_windows.append((current_window_start, timestamps[-1], current_lower, current_upper))
    
    return {
        "timestamps": timestamps,
        "prices": prices,
        "actions": actions,
        "rewards": rewards,
        "cumulative_rewards": cumulative_rewards,
        "max_width": env.max_width,
        "lp_windows": lp_windows,
        "action_map": env.action_map,
    }


def plot_lstm_decisions(lstm_data: Dict, save_path: str = "visualizations/test_lstm_decisions.png"):
    """Plot LSTM DQN decisions on test set."""
    print("\n" + "=" * 60)
    print("📈 Generating LSTM Decision Visualization")
    print("=" * 60)
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    
    timestamps = lstm_data["timestamps"]
    prices = lstm_data["prices"]
    
    n_rebalances = sum(1 for a in lstm_data["actions"] if a > 0)
    
    # --- Plot 1: LSTM with LP Windows ---
    ax1 = axes[0]
    ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='ETH/USDT Price', zorder=10)
    
    # Draw LP windows
    for start, end, lower, upper in lstm_data["lp_windows"]:
        ax1.fill_between([start, end], [lower, lower], [upper, upper], 
                         alpha=0.3, color='purple', edgecolor='darkviolet', linewidth=1, zorder=5)
    
    ax1.set_ylabel('Price (USD)', fontsize=10)
    n_holds = sum(1 for a in lstm_data["actions"] if a == 0)
    ax1.set_title(f'LSTM DQN (this trajectory: {n_holds} HOLDs, {n_rebalances} Rebalances, {len(lstm_data["lp_windows"])} LP windows)', 
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=1.5, label='ETH/USDT Price'),
        Patch(facecolor='purple', alpha=0.4, edgecolor='darkviolet', label='LP Range'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # --- Plot 2: Cumulative Reward ---
    ax2 = axes[1]
    ax2.plot(timestamps, lstm_data["cumulative_rewards"], 'purple', linewidth=1.5)
    ax2.set_ylabel('Cumulative Reward ($)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_title(f'LSTM DQN Cumulative Reward: ${lstm_data["cumulative_rewards"][-1]:.2f}', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Saved to: {save_path}")
    return save_path


def plot_decisions(ppo_data: Dict, dqn_data: Dict, save_path: str = "visualizations/test_decisions.png"):
    """Plot comparison of PPO and DQN decisions on test set with LP windows."""
    print("\n" + "=" * 60)
    print("📈 Generating Decision Visualization")
    print("=" * 60)
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 2, 1])
    
    timestamps = ppo_data["timestamps"]
    prices = ppo_data["prices"]
    
    n_ppo_rebalances = sum(1 for a in ppo_data["actions"] if a > 0)
    n_dqn_rebalances = sum(1 for a in dqn_data["actions"] if a > 0)
    
    # --- Plot 1: PPO with LP Windows ---
    ax1 = axes[0]
    ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='ETH/USDT Price', zorder=10)
    
    # Draw PPO LP windows as colored rectangles
    for i, (start, end, lower, upper) in enumerate(ppo_data["lp_windows"]):
        ax1.fill_between([start, end], [lower, lower], [upper, upper], 
                         alpha=0.3, color='green', edgecolor='darkgreen', linewidth=1, zorder=5)
    
    ax1.set_ylabel('Price (USD)', fontsize=10)
    n_holds = sum(1 for a in ppo_data["actions"] if a == 0)
    ax1.set_title(f'PPO (this trajectory: {n_holds} HOLDs, {n_ppo_rebalances} Rebalances, {len(ppo_data["lp_windows"])} LP windows)', 
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=1.5, label='ETH/USDT Price'),
        Patch(facecolor='green', alpha=0.4, edgecolor='darkgreen', label='LP Range (in position)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # --- Plot 2: DQN with LP Windows ---
    ax2 = axes[1]
    ax2.plot(timestamps, prices, 'b-', linewidth=1.5, label='ETH/USDT Price', zorder=10)
    
    # Draw DQN LP windows as colored rectangles
    for i, (start, end, lower, upper) in enumerate(dqn_data["lp_windows"]):
        ax2.fill_between([start, end], [lower, lower], [upper, upper], 
                         alpha=0.3, color='orange', edgecolor='darkorange', linewidth=1, zorder=5)
    
    ax2.set_ylabel('Price (USD)', fontsize=10)
    n_dqn_holds = sum(1 for a in dqn_data["actions"] if a == 0)
    n_dqn_windows = len(dqn_data["lp_windows"])
    if n_dqn_rebalances == 0:
        title = f'DQN (this trajectory: {n_dqn_holds} HOLDs, no LP)'
    else:
        title = f'DQN (this trajectory: {n_dqn_holds} HOLDs, {n_dqn_rebalances} Rebalances, {n_dqn_windows} LP windows)'
    ax2.set_title(title, fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=1.5, label='ETH/USDT Price'),
    ]
    if dqn_data["lp_windows"]:
        legend_elements.append(Patch(facecolor='orange', alpha=0.4, edgecolor='darkorange', label='LP Range (in position)'))
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # --- Plot 3: Cumulative Rewards ---
    ax3 = axes[2]
    ax3.plot(timestamps, ppo_data["cumulative_rewards"], 'g-', linewidth=2, 
             label=f'PPO (final: {ppo_data["cumulative_rewards"][-1]:.1f})')
    ax3.plot(timestamps, dqn_data["cumulative_rewards"], 'orange', linewidth=2, 
             label=f'DQN (final: {dqn_data["cumulative_rewards"][-1]:.1f})')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1, label='HOLD baseline')
    ax3.set_ylabel('Cumulative Reward ($)', fontsize=10)
    ax3.set_xlabel('Time', fontsize=10)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Cumulative Rewards Comparison', fontsize=11, fontweight='bold')
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved to: {save_path}")
    plt.close()
    
    return save_path


def plot_all_models(ppo_data: Dict, dqn_data: Dict, lstm_data: Dict, save_path: str, run_name: str = ""):
    """Plot all three models (PPO, DQN, LSTM) in one figure with cumulative rewards comparison."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[2, 2, 2, 1])
    timestamps = ppo_data["timestamps"]
    prices = ppo_data["prices"]

    def _add_panel(ax, data: Dict, color: str, title: str):
        ax.plot(timestamps, prices, 'b-', linewidth=1.5, label='Price', zorder=10)
        for start, end, lower, upper in data["lp_windows"]:
            ax.fill_between([start, end], [lower, lower], [upper, upper],
                            alpha=0.3, color=color, edgecolor=color, linewidth=1, zorder=5)
        ax.set_ylabel('Price (USD)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    n_ppo_h = sum(1 for a in ppo_data["actions"] if a == 0)
    n_dqn_h = sum(1 for a in dqn_data["actions"] if a == 0)
    n_lstm_h = sum(1 for a in lstm_data["actions"] if a == 0)
    _add_panel(axes[0], ppo_data, 'green', f'PPO {run_name} ({n_ppo_h} HOLDs)')
    _add_panel(axes[1], dqn_data, 'orange', f'DQN {run_name} ({n_dqn_h} HOLDs)')
    _add_panel(axes[2], lstm_data, 'purple', f'LSTM DQN {run_name} ({n_lstm_h} HOLDs)')

    axes[3].plot(timestamps, ppo_data["cumulative_rewards"], 'g-', linewidth=2, label=f'PPO ({ppo_data["cumulative_rewards"][-1]:.0f})')
    axes[3].plot(timestamps, dqn_data["cumulative_rewards"], color='orange', linewidth=2, label=f'DQN ({dqn_data["cumulative_rewards"][-1]:.0f})')
    axes[3].plot(timestamps, lstm_data["cumulative_rewards"], 'purple', linewidth=2, label=f'LSTM ({lstm_data["cumulative_rewards"][-1]:.0f})')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Cumulative Reward ($)', fontsize=10)
    axes[3].set_xlabel('Date', fontsize=10)
    axes[3].legend(loc='lower left')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Cumulative Rewards', fontsize=11, fontweight='bold')
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved to: {save_path}", flush=True)
    return save_path

def plot_learning_curves(dqn_rewards: List[float], ppo_log_dir: str, save_path: str):
    """Plot training reward curves."""
    print("\n" + "=" * 60)
    print("📈 Generating Learning Curves")
    print("=" * 60)
    
    plt.figure(figsize=(10, 6))
    
    # Plot DQN
    if dqn_rewards:
        # Smooth DQN rewards
        window = max(len(dqn_rewards) // 20, 10)
        smoothed = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        
        # Convert episodes to approx steps (assuming 2200 steps/ep)
        steps_per_ep = 2200
        dqn_steps = np.arange(len(dqn_rewards)) * steps_per_ep
        smoothed_steps = (np.arange(len(smoothed)) + window - 1) * steps_per_ep
        
        plt.plot(dqn_steps, dqn_rewards, alpha=0.2, color='orange', label='DQN (Raw)')
        plt.plot(smoothed_steps, smoothed, color='orange', linewidth=2, label='DQN (Smoothed)')
        
    # Plot PPO (from eval logs)
    eval_file = os.path.join(ppo_log_dir, "evaluations.npz")
    if os.path.exists(eval_file):
        try:
            data = np.load(eval_file)
            timesteps = data['timesteps']
            results = data['results']
            mean_rewards = np.mean(results, axis=1)
            plt.plot(timesteps, mean_rewards, 'g-o', linewidth=2, label='PPO (Eval)')
        except Exception as e:
            print(f"  Could not load PPO logs: {e}")
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Training Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Saved to: {save_path}")
    return save_path
# DQN/LSTM action ID -> width. Widths [1,2,5,8,10]. Action 0 = HOLD.
DQN_WIDTHS = [1, 3, 5, 10, 20, 40]


def action_id_to_label_dqn(action_id: int) -> str:
    """Human-readable label for DQN/LSTM action (0 = HOLD, 1-5 = width)."""
    if action_id == 0:
        return "HOLD"
    k = int(action_id) - 1
    if k < 0 or k >= len(DQN_WIDTHS):
        return f"action_{action_id}"
    w = DQN_WIDTHS[k]
    return f"width={w}"


def format_action_distribution_human(ad: dict, total_steps: int) -> str:
    """Format action distribution for DQN/LSTM as human-readable lines (only actions with count > 0)."""
    if not ad or total_steps <= 0:
        return ""
    lines = []
    for k, v in sorted(ad.items(), key=lambda x: int(x[0])):
        count = int(v)
        if count == 0:
            continue
        label = action_id_to_label_dqn(int(k))
        pct = 100.0 * count / total_steps
        lines.append(f"  - **{label}**: {count} steps ({pct:.1f}%)")
    return "\n".join(lines) if lines else "  (no actions)"


def compute_pnl_1000(trajectory_data: dict) -> dict:
    """
    From trajectory (one episode): cumulative reward is USD PnL for 2 ETH position.
    Scale to "if we invested $1000 at test start, what would we have at end?"
    initial_cap_usd = 2 * price[0] (2 ETH in USD at first step).
    return = cumulative_reward / initial_cap_usd.
    end_value_1000 = 1000 * (1 + return) = 1000 + PnL on $1000.
    """
    out = {}
    for name, data in trajectory_data.items():
        if not data or "prices" not in data or "cumulative_rewards" not in data:
            continue
        prices = data["prices"]
        cum = data["cumulative_rewards"]
        if not prices or not cum:
            continue
        initial_cap_usd = 2.0 * float(prices[0])  # 2 ETH at start
        cumulative_reward = float(cum[-1])
        return_pct = cumulative_reward / initial_cap_usd if initial_cap_usd > 0 else 0.0
        pnl_1000 = 1000.0 * return_pct
        end_value_1000 = 1000.0 + pnl_1000
        out[name] = {
            "initial_cap_usd": initial_cap_usd,
            "cumulative_reward": cumulative_reward,
            "return_pct": return_pct,
            "pnl_1000": pnl_1000,
            "end_value_1000": end_value_1000,
        }
    return out


def generate_run_config(run_dir: str, args, results: dict, trajectory_data: dict = None):
    """Generate a RUN_CONFIG.md documenting this run's settings and results."""
    
    # DQN/LSTM widths (no offsets, always centered)
    dqn_widths = [1, 3, 5, 10, 20, 40]
    ppo_action_ticks = [0, 1, 3, 5, 10, 20, 40]
    ppo_labels = ["HOLD", "W=1", "W=3", "W=5", "W=10", "W=20", "W=40"]
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config = f"""# Run Configuration: {os.path.basename(run_dir)}

**Last updated**: {now}
**Data**: {args.data_dir}

## Action Space

### DQN / LSTM DQN
- **Widths**: {dqn_widths} (tick_spacing units, ~0.1% to 1.0% ranges)
- **Always centered** at current price (no directional offset)
- **Total actions**: 1 (HOLD) + {len(dqn_widths)} = {1 + len(dqn_widths)}
- Action 0 = HOLD, Actions 1-{len(dqn_widths)} = width choices

### PPO
- **Action ticks**: {ppo_action_ticks}
  - 0 = HOLD
  - {ppo_action_ticks[1:]} map to widths {dqn_widths} (in tick_spacing units)
- **Total actions**: {len(ppo_action_ticks)}

## Training Configuration

| Parameter | PPO | DQN | LSTM DQN |
|-----------|-----|-----|----------|
| Episodes/Timesteps | {args.ppo_timesteps:,} ts | {args.dqn_episodes} ep | {args.lstm_episodes} ep |
| LSTM Seq Length | - | - | {args.lstm_seq_len} |
| Device | cpu | {args.device} | {args.device} |
| Eval Episodes | {args.eval_episodes} | {args.eval_episodes} | {args.eval_episodes} |

## Environment Settings (Unchanged)
- **Fee model**: Exact per-swap (paper method)
- **Gas cost**: $0.50 per rebalance (Arbitrum L2)
- **Initial position**: 2.0 ETH
- **Pool**: ETH/USDT 0.3% (tick_spacing=10)

## Results

### Baselines
"""
    for name, res in results.get("baselines", {}).items():
        if "mean_reward" in res:
            config += f"- **{name}**: {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}\n"
    
    config += "\n### Algorithms\n"
    for name, res in results.get("algorithms", {}).items():
        if "mean_reward" in res:
            config += f"- **{name}**: {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}\n"
            if "action_distribution" in res and name == "ppo":
                ad = res["action_distribution"]
                parts = [f"{ppo_labels[int(k)]}:{v}" for k, v in sorted(ad.items(), key=lambda x: int(x[0]))]
                config += f"  - PPO actions: {', '.join(parts)}\n"
            elif "action_distribution" in res:
                ad = res["action_distribution"]
                total = sum(int(v) for v in ad.values())
                config += f"  - See \"Actions in plain English\" below for breakdown.\n"
        elif "error" in res:
            config += f"- **{name}**: ERROR - {res['error']}\n"
    
    config += "\n## Actions in plain English\n\n"
    config += "**PPO** (6 actions):\n"
    config += "- **HOLD** = do nothing, keep current LP range (no gas).\n"
    config += "- **W=1** = set range width to ±1 tick_spacing (~0.1%), centered at current price.\n"
    config += "- **W=3** = ±3 tick_spacings (~0.3%), centered.\n"
    config += "- **W=5** = ±5 (~0.5%), centered.\n"
    config += "- **W=10** = ±10 (~1%), centered.\n"
    config += "- **W=20** = ±20 (~2%), centered.\n"
    config += "- **W=40** = ±40 (~4%), centered.\n"
    config += "Each non-HOLD action recenters the range at current price and costs gas ($0.50).\n\n"
    config += "**DQN / LSTM** (7 actions):\n"
    config += "- Same as PPO: HOLD + width=1,3,5,10,20,40, always centered.\n\n"
    config += "**Important:** Both \"Plotted trajectory\" and \"Eval (10 ep)\" use the **same test period** (same dates). Plotted = 1 run through that period (what you see in the viz). Eval (10 ep) = 10 runs through the same period, aggregated.\n\n"
    config += "**Viz panels:** In the plots, each panel is one model. PPO panel shows PPO's HOLD/rebalance counts (e.g. 785 HOLDs). DQN panel shows DQN's (often 0 HOLDs). LSTM panel shows LSTM's. The numbers in \"Plotted trajectory\" below match the panel for that model.\n\n"
    # Plotted trajectory (what you see in the viz) - from trajectory_data
    if trajectory_data:
        config += "**Plotted trajectory (what you see in the viz)**\n\n"
        for name, key in [("ppo", "ppo"), ("dqn", "dqn"), ("lstm_dqn", "lstm_dqn")]:
            data = trajectory_data.get(key)
            if not data or "actions" not in data:
                continue
            actions_list = data["actions"]
            total = len(actions_list)
            if total <= 0:
                continue
            counts = Counter(int(a) for a in actions_list)
            ad = {str(k): v for k, v in counts.items()}
            config += f"**{name}**\n"
            if name == "ppo":
                for k, v in sorted(ad.items(), key=lambda x: int(x[0])):
                    count = int(v)
                    if count == 0:
                        continue
                    label = ppo_labels[int(k)]
                    pct = 100.0 * count / total
                    config += f"- **{label}**: {count} steps ({pct:.1f}%)\n"
            else:
                config += format_action_distribution_human(ad, total) + "\n"
            config += "\n"
        config += "**Eval (10 episodes, aggregate)**\n\n"
    else:
        config += "**How often each model chose what** (10 eval episodes)\n\n"
    for name, res in results.get("algorithms", {}).items():
        if "action_distribution" not in res or "mean_reward" not in res:
            continue
        ad = res["action_distribution"]
        total = sum(int(v) for v in ad.values())
        if total <= 0:
            continue
        config += f"**{name}**\n"
        if name == "ppo":
            for k, v in sorted(ad.items(), key=lambda x: int(x[0])):
                count = int(v)
                if count == 0:
                    continue
                label = ppo_labels[int(k)]
                pct = 100.0 * count / total
                config += f"- **{label}**: {count} steps ({pct:.1f}%)\n"
        else:
            config += format_action_distribution_human(ad, total) + "\n"
        config += "\n"
    
    if getattr(args, "run_note", None):
        config += f"\n## Notes\n{args.run_note}\n"
    
    # PnL for $1000 invested: plotted trajectory + mean over 10 episodes
    if trajectory_data:
        pnl = compute_pnl_1000(trajectory_data)
        if pnl:
            initial_cap = list(pnl.values())[0]["initial_cap_usd"]
            config += "\n## PnL: $1000 invested at test start\n\n"
            config += "| Model | Plotted cum (USD) | Plotted return% | Plotted end $1000 | Mean cum (10 ep) | Mean return% | Mean end $1000 |\n"
            config += "|-------|-------------------|-----------------|-------------------|------------------|---------------|------------------|\n"
            for name, v in pnl.items():
                mean_reward = results.get("algorithms", {}).get(name, {}).get("mean_reward")
                if mean_reward is not None and initial_cap > 0:
                    mean_ret = mean_reward / initial_cap
                    mean_pnl = 1000.0 * mean_ret
                    mean_end = 1000.0 + mean_pnl
                    config += f"| {name} | {v['cumulative_reward']:.2f} | {v['return_pct']*100:.2f}% | ${v['end_value_1000']:.2f} | {mean_reward:.2f} | {mean_ret*100:.2f}% | ${mean_end:.2f} |\n"
                else:
                    config += f"| {name} | {v['cumulative_reward']:.2f} | {v['return_pct']*100:.2f}% | ${v['end_value_1000']:.2f} | - | - | - |\n"
            config += "\n(Plotted = single trajectory in the viz; Mean = average over 10 eval episodes. Env uses 2 ETH ≈ ${:.0f} at start.)\n\n".format(initial_cap)
    
    config += """
## Why PPO cumulative reward can dip (e.g. -15301)
- The **plot shows one test trajectory** (one episode). Cumulative reward is step-by-step.
- Each step: reward = fee - LVR - gas (if rebalance) - opportunity_cost (if out of range).
- **LVR** uses the instantaneous formula ℓ = L×σ²/4×√p (Equation 16). In high-volatility hours this can be large and exceed fees, so the step reward is negative.
- Over many such hours the cumulative can drop (e.g. to -15301), then recover when fees dominate again. The **reported mean reward** (e.g. 1170) is over 10 episodes; the plotted trajectory can be one where the curve dips then recovers.
"""
    
    config += f"""
## Key Differences from Previous Runs
- **Width options**: [1,2,5,8,10] (DQN) / [0,1,2,4,10,16,20] (PPO)
  - Added narrowest width=1 (~0.1% range) to both models
  - **Removed offsets** — all positions always centered at current price
  - Simplified action space: DQN 6 actions, PPO 7 actions
- **Bug fixes**: opportunity cost corrected (was 52.6% APY, now 5% APY), swap fee halved (50% of position swapped)
"""
    
    config_path = os.path.join(run_dir, "RUN_CONFIG.md")
    with open(config_path, "w") as f:
        f.write(config)
    
    print(f"\n  Saved run config to: {config_path}", flush=True)
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Compare PPO vs DQN vs LSTM for Uniswap v3 LP")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--run-dir", type=str, default="run_001", help="Output directory for this run")
    parser.add_argument("--ppo-timesteps", type=int, default=50000, help="PPO training timesteps")
    parser.add_argument("--dqn-episodes", type=int, default=200, help="DQN training episodes")
    parser.add_argument("--lstm-episodes", type=int, default=1000, help="LSTM DQN training episodes")
    parser.add_argument("--lstm-seq-len", type=int, default=24, help="LSTM sequence length (hours of history)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device for DQN/LSTM")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating visualization plot")
    parser.add_argument("--run-note", type=str, default="", help="Optional note for RUN_CONFIG.md (e.g. 'LSTM stopped early')")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel PPO environments")
    
    args = parser.parse_args()
    
    run_dir = args.run_dir
    models_dir = os.path.join(run_dir, "models")
    viz_dir = os.path.join(run_dir, "visualizations")
    
    # Ensure output directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}", flush=True)
    print(f"  RUN: {run_dir}", flush=True)
    print(f"  Models  -> {models_dir}/", flush=True)
    print(f"  Visuals -> {viz_dir}/", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": run_dir,
        "data_dir": args.data_dir,
        "algorithms": {},
        "baselines": {},
    }
    
    # ── Baselines ──
    print("\n" + "=" * 70, flush=True)
    print("  RUNNING BASELINES", flush=True)
    print("=" * 70, flush=True)
    
    results["baselines"]["hold"] = run_baseline_hold(args.data_dir, args.eval_episodes)
    results["baselines"]["fixed_width_1"] = run_baseline_fixed_width(args.data_dir, 1, args.eval_episodes)
    results["baselines"]["fixed_width_5"] = run_baseline_fixed_width(args.data_dir, 5, args.eval_episodes)
    results["baselines"]["fixed_width_10"] = run_baseline_fixed_width(args.data_dir, 10, args.eval_episodes)
    
    # ── Training (all 3 models) ──
    if not args.skip_training:
        print("\n" + "=" * 70, flush=True)
        print("  TRAINING ALL MODELS", flush=True)
        print("=" * 70, flush=True)
        
        # PPO
        try:
            run_ppo_training(args.data_dir, args.ppo_timesteps, num_envs=args.num_envs, save_prefix=models_dir)
        except Exception as e:
            print(f"PPO training failed: {e}", flush=True)
            import traceback; traceback.print_exc()
        
        # DQN
        dqn_rewards = []
        try:
            dqn_rewards = run_dqn_training(args.data_dir, args.dqn_episodes, args.device, save_prefix=models_dir)
        except Exception as e:
            print(f"DQN training failed: {e}", flush=True)
            import traceback; traceback.print_exc()
        
        # LSTM DQN
        try:
            run_lstm_dqn_training(args.data_dir, args.lstm_episodes, args.lstm_seq_len, args.device, save_prefix=models_dir)
        except Exception as e:
            print(f"LSTM DQN training failed: {e}", flush=True)
            import traceback; traceback.print_exc()
    
    # ── Evaluation ──
    print("\n" + "=" * 70, flush=True)
    print("  EVALUATING ALL MODELS", flush=True)
    print("=" * 70, flush=True)
    
    try:
        results["algorithms"]["ppo"] = evaluate_ppo(args.data_dir, args.eval_episodes, save_prefix=models_dir)
    except Exception as e:
        print(f"PPO evaluation failed: {e}", flush=True)
        results["algorithms"]["ppo"] = {"error": str(e)}
    
    try:
        results["algorithms"]["dqn"] = evaluate_dqn(args.data_dir, args.eval_episodes, args.device, save_prefix=models_dir)
    except Exception as e:
        print(f"DQN evaluation failed: {e}", flush=True)
        results["algorithms"]["dqn"] = {"error": str(e)}
    
    try:
        results["algorithms"]["lstm_dqn"] = evaluate_lstm_dqn(
            args.data_dir, args.eval_episodes, args.lstm_seq_len, args.device, save_prefix=models_dir
        )
    except Exception as e:
        print(f"LSTM DQN evaluation failed: {e}", flush=True)
        results["algorithms"]["lstm_dqn"] = {"error": str(e)}
    
    # ── Summary ──
    print("\n" + "=" * 70, flush=True)
    print("  FINAL RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print("\nBaselines:", flush=True)
    for name, res in results["baselines"].items():
        if "mean_reward" in res:
            print(f"  {name:20s}: {res['mean_reward']:8.2f} +/- {res['std_reward']:.2f}", flush=True)
    
    print("\nAlgorithms:", flush=True)
    for name, res in results["algorithms"].items():
        if "mean_reward" in res:
            print(f"  {name:20s}: {res['mean_reward']:8.2f} +/- {res['std_reward']:.2f}", flush=True)
        elif "error" in res:
            print(f"  {name:20s}: ERROR - {res['error']}", flush=True)
    
    # Save results JSON
    results_path = os.path.join(viz_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}", flush=True)
    
    # ── Visualization ──
    trajectory_data = None
    if not args.no_plot:
        print("\n" + "=" * 70, flush=True)
        print("  GENERATING VISUALIZATIONS", flush=True)
        print("=" * 70, flush=True)
        
        ppo_data = dqn_data = lstm_data = None
        try:
            model_path = f"{models_dir}/comparison_dqn_best.pth"
            if not os.path.exists(model_path):
                model_path = f"{models_dir}/comparison_dqn_final.pth"
            
            ppo_data = visualize_ppo_decisions(args.data_dir, f"{models_dir}/comparison_ppo.zip")
            dqn_data = visualize_dqn_decisions(args.data_dir, model_path, args.device)
            plot_path = plot_decisions(ppo_data, dqn_data, f"{viz_dir}/test_decisions.png")
            print(f"\n  PPO+DQN saved to: {plot_path}", flush=True)
        except Exception as e:
            print(f"  PPO+DQN visualization failed: {e}", flush=True)
        
        try:
            model_path = f"{models_dir}/comparison_lstm_dqn_best.pth"
            if not os.path.exists(model_path):
                model_path = f"{models_dir}/comparison_lstm_dqn_final.pth"
                
            lstm_data = visualize_lstm_decisions(
                args.data_dir, model_path,
                args.lstm_seq_len, args.device
            )
            lstm_plot_path = plot_lstm_decisions(lstm_data, f"{viz_dir}/test_lstm_decisions.png")
            print(f"  LSTM saved to: {lstm_plot_path}", flush=True)
        except Exception as e:
            print(f"  LSTM visualization failed: {e}", flush=True)
        
        if ppo_data and dqn_data and lstm_data:
            try:
                plot_all_models(ppo_data, dqn_data, lstm_data, f"{viz_dir}/test_all_models.png", run_name=run_dir)
            except Exception as e:
                print(f"  All-models plot failed: {e}", flush=True)
        
        # Plot Learning Curves
        if not args.skip_training:
            try:
                plot_learning_curves(dqn_rewards if 'dqn_rewards' in locals() else [], 
                                     "./eval_logs_paper/", 
                                     f"{viz_dir}/learning_curves.png")
            except Exception as e:
                print(f"  Learning curve plot failed: {e}")

        # PnL for $1000 invested (plotted trajectory + mean over 10 ep)
        trajectory_data = {"ppo": ppo_data, "dqn": dqn_data, "lstm_dqn": lstm_data}
        pnl = compute_pnl_1000(trajectory_data)
        if pnl:
            initial_cap = list(pnl.values())[0]["initial_cap_usd"]
            print("\n  PnL if you invested $1000 at test start:", flush=True)
            print("    (Plotted = single trajectory in viz; Mean = average over 10 eval episodes)", flush=True)
            for name, v in pnl.items():
                mean_reward = results.get("algorithms", {}).get(name, {}).get("mean_reward")
                if mean_reward is not None and initial_cap > 0:
                    mean_end = 1000.0 + 1000.0 * mean_reward / initial_cap
                    print(f"    {name:10s}  plotted: cum ${v['cumulative_reward']:+.2f}  →  end ${v['end_value_1000']:.2f}   |   mean (10ep): cum ${mean_reward:+.2f}  →  end ${mean_end:.2f}", flush=True)
                else:
                    print(f"    {name:10s}  plotted: cum ${v['cumulative_reward']:+.2f}  →  end ${v['end_value_1000']:.2f}", flush=True)
    
    # ── Generate RUN_CONFIG.md (always, so it stays in sync with results) ──
    generate_run_config(run_dir, args, results, trajectory_data=trajectory_data)
    
    print(f"\n{'=' * 70}", flush=True)
    print(f"  RUN COMPLETE: {run_dir}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
