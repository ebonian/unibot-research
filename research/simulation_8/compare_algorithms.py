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
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run_ppo_training(data_dir: str, timesteps: int = 50000, num_envs: int = 4):
    """Train PPO model."""
    print("\n" + "=" * 60)
    print("🎯 Training PPO (Xu & Brini 2025 methodology)")
    print("=" * 60)
    
    from uniswap_v3_ppo_paper import train_paper_method
    
    model, env = train_paper_method(
        data_dir=data_dir,
        num_envs=num_envs,
        total_timesteps=timesteps,
        save_path="models/comparison_ppo",
    )
    return model


def run_dqn_training(data_dir: str, episodes: int = 200, device: str = "cpu"):
    """Train Dueling DDQN model."""
    print("\n" + "=" * 60)
    print("🎯 Training Dueling DDQN (Zhang et al. 2023 methodology)")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import train_dqn
    
    agent = train_dqn(
        data_dir=data_dir,
        n_episodes=episodes,
        save_path="models/comparison_dqn",
        device=device,
    )
    return agent


def run_lstm_dqn_training(data_dir: str, episodes: int = 1000, seq_len: int = 24, device: str = "cpu"):
    """Train LSTM-based Dueling DDQN model."""
    print("\n" + "=" * 60)
    print("🧠 Training LSTM Dueling DDQN (Sequence-aware)")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import train_lstm_dqn
    
    agent = train_lstm_dqn(
        data_dir=data_dir,
        n_episodes=episodes,
        seq_len=seq_len,
        save_path="models/comparison_lstm_dqn",
        device=device,
    )
    return agent


def evaluate_ppo(data_dir: str, n_episodes: int = 10):
    """Evaluate PPO on test set."""
    print("\n" + "=" * 60)
    print("📊 Evaluating PPO on test set")
    print("=" * 60)
    
    from uniswap_v3_ppo_paper import evaluate_paper_method
    
    results = evaluate_paper_method(
        data_dir=data_dir,
        model_path="models/comparison_ppo.zip",
        n_episodes=n_episodes,
    )
    return results


def evaluate_lstm_dqn(data_dir: str, n_episodes: int = 10, seq_len: int = 24, device: str = "cpu"):
    """Evaluate LSTM Dueling DDQN on test set."""
    print("\n" + "=" * 60)
    print("📊 Evaluating LSTM Dueling DDQN on test set")
    print("=" * 60)
    
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
    
    model_path = "models/comparison_lstm_dqn_best.pth"
    if not os.path.exists(model_path):
        model_path = "models/comparison_lstm_dqn_final.pth"
    
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
            width, offset = test_env.action_map.get(action_id, (None, None))
            if width is None:
                action_name = "HOLD"
            else:
                action_name = f"w={width}, off={offset:+d}"
            print(f"  {action_name:20s}: {count:5d} ({100*count/total_actions:.1f}%)")
    
    print(f"\nResults:")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "action_distribution": {str(k): v for k, v in action_counts.items()},
    }


def evaluate_dqn(data_dir: str, n_episodes: int = 10, device: str = "cpu"):
    """Evaluate Dueling DDQN on test set."""
    print("\n" + "=" * 60)
    print("📊 Evaluating Dueling DDQN on test set")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import evaluate_on_test
    
    results = evaluate_on_test(
        data_dir=data_dir,
        model_path="models/comparison_dqn_best.pth",
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
    """Run baseline: fixed width LP, centered (offset=0), reallocate every step."""
    print("\n" + "=" * 60)
    print(f"📊 Baseline: Fixed Width={width} (centered, always reallocate)")
    print("=" * 60)
    
    from uniswap_v3_dqn_paper import prepare_hourly_data_extended, UniswapV3DQNEnv
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    env = UniswapV3DQNEnv(hourly_data, mode="test")
    
    # Find action ID for (width, offset=0) in the new action mapping
    # Action 0 = HOLD, then widths=[5,10,25,50], offsets=[-2,-1,0,1,2]
    # For each width, offset=0 is at index 2 (third offset)
    widths = [5, 10, 25, 50]
    offsets = [-2, -1, 0, 1, 2]
    
    if width not in widths:
        # Find closest width
        width = min(widths, key=lambda w: abs(w - width))
        print(f"  (Using closest available width: {width})")
    
    # Calculate action ID: 1 + (width_idx * len(offsets)) + offset_idx
    width_idx = widths.index(width)
    offset_idx = offsets.index(0)  # offset = 0 for centered
    action_id = 1 + (width_idx * len(offsets)) + offset_idx
    
    print(f"  Action mapping: width={width}, offset=0 → action_id={action_id}")
    
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
    
    from uniswap_v3_ppo_paper import prepare_hourly_data, UniswapV3PaperEnv, tick_to_price
    from stable_baselines3 import PPO
    
    hourly_data = prepare_hourly_data(data_dir)
    env = UniswapV3PaperEnv(hourly_data, mode="test")
    model = PPO.load(model_path)
    
    # Collect trajectory
    timestamps = []
    prices = []
    actions = []
    rewards = []
    cumulative_rewards = []
    lp_windows = []  # List of (start_time, end_time, lower_price, upper_price)
    
    obs, _ = env.reset()
    done = False
    cum_reward = 0.0
    
    current_window_start = None
    current_lower = None
    current_upper = None
    
    while not done:
        t = env.timestamps[env.idx]
        price = env._get_price(t)
        
        # Check LP state before action
        had_lp = env.has_lp
        old_lower = env.lp_lower_tick
        old_upper = env.lp_upper_tick
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        timestamps.append(t)
        prices.append(price)
        actions.append(int(action))
        rewards.append(reward)
        cum_reward += reward
        cumulative_rewards.append(cum_reward)
        
        # Track LP windows
        if env.has_lp:
            lower_price = tick_to_price(env.lp_lower_tick)
            upper_price = tick_to_price(env.lp_upper_tick)
            
            # New position or position changed
            if current_window_start is None or (env.lp_lower_tick != old_lower or env.lp_upper_tick != old_upper):
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
        "action_ticks": env.action_ticks,
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
    
    # Find rebalancing events
    rebalance_times = []
    rebalance_prices = []
    for i, action in enumerate(lstm_data["actions"]):
        if action > 0:
            rebalance_times.append(timestamps[i])
            rebalance_prices.append(prices[i])
    
    # --- Plot 1: LSTM with LP Windows ---
    ax1 = axes[0]
    ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='ETH/USDT Price', zorder=10)
    
    # Draw LP windows
    for start, end, lower, upper in lstm_data["lp_windows"]:
        ax1.fill_between([start, end], [lower, lower], [upper, upper], 
                         alpha=0.3, color='purple', edgecolor='darkviolet', linewidth=1, zorder=5)
    
    # Rebalance markers
    if rebalance_times:
        ax1.scatter(rebalance_times, rebalance_prices, 
                   marker='^', c='red', s=100, zorder=15, label=f'Rebalance ({len(rebalance_times)}x)')
        for t in rebalance_times:
            ax1.axvline(x=t, color='red', alpha=0.3, linestyle='--', linewidth=1, zorder=3)
    
    ax1.set_ylabel('Price (USD)', fontsize=10)
    n_holds = sum(1 for a in lstm_data["actions"] if a == 0)
    n_rebalances = len(rebalance_times)
    ax1.set_title(f'LSTM DQN Agent: {n_holds} HOLDs, {n_rebalances} Rebalances ({len(lstm_data["lp_windows"])} LP windows)', 
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=1.5, label='ETH/USDT Price'),
        Patch(facecolor='purple', alpha=0.4, edgecolor='darkviolet', label='LP Range'),
    ]
    if rebalance_times:
        legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                                      markersize=10, label=f'Rebalance ({n_rebalances}x)'))
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
    
    # Find rebalancing events (non-zero actions)
    ppo_rebalance_times = []
    ppo_rebalance_prices = []
    ppo_rebalance_actions = []
    for i, action in enumerate(ppo_data["actions"]):
        if action > 0:  # Non-HOLD action
            ppo_rebalance_times.append(timestamps[i])
            ppo_rebalance_prices.append(prices[i])
            ppo_rebalance_actions.append(action)
    
    dqn_rebalance_times = []
    dqn_rebalance_prices = []
    dqn_rebalance_actions = []
    for i, action in enumerate(dqn_data["actions"]):
        if action > 0:  # Non-HOLD action
            dqn_rebalance_times.append(timestamps[i])
            dqn_rebalance_prices.append(prices[i])
            dqn_rebalance_actions.append(action)
    
    # --- Plot 1: PPO with LP Windows and Rebalance Markers ---
    ax1 = axes[0]
    ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='ETH/USDT Price', zorder=10)
    
    # Draw PPO LP windows as colored rectangles
    for i, (start, end, lower, upper) in enumerate(ppo_data["lp_windows"]):
        ax1.fill_between([start, end], [lower, lower], [upper, upper], 
                         alpha=0.3, color='green', edgecolor='darkgreen', linewidth=1, zorder=5)
    
    # Add rebalance markers (red triangles)
    if ppo_rebalance_times:
        ax1.scatter(ppo_rebalance_times, ppo_rebalance_prices, 
                   marker='^', c='red', s=100, zorder=15, label=f'Rebalance ({len(ppo_rebalance_times)}x)')
        # Add vertical lines at rebalance points
        for t in ppo_rebalance_times:
            ax1.axvline(x=t, color='red', alpha=0.3, linestyle='--', linewidth=1, zorder=3)
    
    ax1.set_ylabel('Price (USD)', fontsize=10)
    n_holds = sum(1 for a in ppo_data["actions"] if a == 0)
    n_rebalances = len(ppo_rebalance_times)
    ax1.set_title(f'PPO Agent: {n_holds} HOLDs, {n_rebalances} Rebalances ({len(ppo_data["lp_windows"])} LP windows)', 
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Build legend
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=1.5, label='ETH/USDT Price'),
        Patch(facecolor='green', alpha=0.4, edgecolor='darkgreen', label='LP Range (in position)'),
    ]
    if ppo_rebalance_times:
        legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                                      markersize=10, label=f'Rebalance ({n_rebalances}x)'))
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # --- Plot 2: DQN with LP Windows and Rebalance Markers ---
    ax2 = axes[1]
    ax2.plot(timestamps, prices, 'b-', linewidth=1.5, label='ETH/USDT Price', zorder=10)
    
    # Draw DQN LP windows as colored rectangles
    for i, (start, end, lower, upper) in enumerate(dqn_data["lp_windows"]):
        ax2.fill_between([start, end], [lower, lower], [upper, upper], 
                         alpha=0.3, color='orange', edgecolor='darkorange', linewidth=1, zorder=5)
    
    # Add rebalance markers (red triangles)
    if dqn_rebalance_times:
        ax2.scatter(dqn_rebalance_times, dqn_rebalance_prices, 
                   marker='^', c='red', s=100, zorder=15, label=f'Rebalance ({len(dqn_rebalance_times)}x)')
        for t in dqn_rebalance_times:
            ax2.axvline(x=t, color='red', alpha=0.3, linestyle='--', linewidth=1, zorder=3)
    
    ax2.set_ylabel('Price (USD)', fontsize=10)
    n_dqn_holds = sum(1 for a in dqn_data["actions"] if a == 0)
    n_dqn_rebalances = len(dqn_rebalance_times)
    n_dqn_windows = len(dqn_data["lp_windows"])
    if n_dqn_rebalances == 0:
        title = f'DQN Agent: Always HOLD ({n_dqn_holds} steps, no LP)'
    else:
        title = f'DQN Agent: {n_dqn_holds} HOLDs, {n_dqn_rebalances} Rebalances ({n_dqn_windows} LP windows)'
    ax2.set_title(title, fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Build legend
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=1.5, label='ETH/USDT Price'),
    ]
    if dqn_data["lp_windows"]:
        legend_elements.append(Patch(facecolor='orange', alpha=0.4, edgecolor='darkorange', label='LP Range (in position)'))
    if dqn_rebalance_times:
        legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                                      markersize=10, label=f'Rebalance ({n_dqn_rebalances}x)'))
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


def main():
    parser = argparse.ArgumentParser(description="Compare PPO vs DQN for Uniswap v3 LP")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--ppo-timesteps", type=int, default=500000, help="PPO training timesteps")
    parser.add_argument("--dqn-episodes", type=int, default=1000, help="DQN training episodes")
    parser.add_argument("--lstm-episodes", type=int, default=1000, help="LSTM DQN training episodes")
    parser.add_argument("--lstm-seq-len", type=int, default=24, help="LSTM sequence length (hours of history)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device for DQN/LSTM")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--baselines-only", action="store_true", help="Only run baselines")
    parser.add_argument("--lstm-only", action="store_true", help="Only train/evaluate LSTM DQN")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating visualization plot")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": args.data_dir,
        "algorithms": {},
        "baselines": {},
    }
    
    # Run baselines
    print("\n" + "=" * 70)
    print("🔬 RUNNING BASELINES")
    print("=" * 70)
    
    results["baselines"]["hold"] = run_baseline_hold(args.data_dir, args.eval_episodes)
    results["baselines"]["fixed_width_5"] = run_baseline_fixed_width(args.data_dir, 5, args.eval_episodes)
    results["baselines"]["fixed_width_25"] = run_baseline_fixed_width(args.data_dir, 25, args.eval_episodes)
    results["baselines"]["fixed_width_50"] = run_baseline_fixed_width(args.data_dir, 50, args.eval_episodes)
    
    if args.baselines_only:
        print("\n" + "=" * 70)
        print("📋 BASELINE RESULTS SUMMARY")
        print("=" * 70)
        for name, res in results["baselines"].items():
            print(f"  {name:20s}: {res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")
        return
    
    # LSTM-only mode
    if args.lstm_only:
        print("\n" + "=" * 70)
        print("🧠 LSTM DQN TRAINING MODE")
        print("=" * 70)
        
        if not args.skip_training:
            try:
                run_lstm_dqn_training(
                    args.data_dir, 
                    args.lstm_episodes, 
                    args.lstm_seq_len,
                    args.device
                )
            except Exception as e:
                print(f"LSTM DQN training failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Evaluate LSTM
        try:
            results["algorithms"]["lstm_dqn"] = evaluate_lstm_dqn(
                args.data_dir, 
                args.eval_episodes,
                args.lstm_seq_len, 
                args.device
            )
        except Exception as e:
            print(f"LSTM DQN evaluation failed: {e}")
            results["algorithms"]["lstm_dqn"] = {"error": str(e)}
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 LSTM RESULTS SUMMARY")
        print("=" * 70)
        print("\nBaselines:")
        for name, res in results["baselines"].items():
            if "mean_reward" in res:
                print(f"  {name:20s}: {res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")
        
        print("\nLSTM DQN:")
        if "lstm_dqn" in results["algorithms"] and "mean_reward" in results["algorithms"]["lstm_dqn"]:
            res = results["algorithms"]["lstm_dqn"]
            print(f"  lstm_dqn             : {res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")
        
        # Save results
        with open("visualizations/comparison_results_lstm.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to visualizations/comparison_results_lstm.json")
        return
    
    # Train algorithms (normal mode: PPO + DQN)
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("🏋️ TRAINING ALGORITHMS")
        print("=" * 70)
        
        try:
            run_ppo_training(args.data_dir, args.ppo_timesteps)
        except Exception as e:
            print(f"PPO training failed: {e}")
        
        try:
            run_dqn_training(args.data_dir, args.dqn_episodes, args.device)
        except Exception as e:
            print(f"DQN training failed: {e}")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("📊 EVALUATING ALGORITHMS")
    print("=" * 70)
    
    try:
        results["algorithms"]["ppo"] = evaluate_ppo(args.data_dir, args.eval_episodes)
    except Exception as e:
        print(f"PPO evaluation failed: {e}")
        results["algorithms"]["ppo"] = {"error": str(e)}
    
    try:
        results["algorithms"]["dqn"] = evaluate_dqn(args.data_dir, args.eval_episodes, args.device)
    except Exception as e:
        print(f"DQN evaluation failed: {e}")
        results["algorithms"]["dqn"] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS SUMMARY")
    print("=" * 70)
    print("\nBaselines:")
    for name, res in results["baselines"].items():
        if "mean_reward" in res:
            print(f"  {name:20s}: {res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")
    
    print("\nAlgorithms:")
    for name, res in results["algorithms"].items():
        if "mean_reward" in res:
            print(f"  {name:20s}: {res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")
        elif "error" in res:
            print(f"  {name:20s}: ERROR - {res['error']}")
    
    # Save results
    with open("visualizations/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to visualizations/comparison_results.json")
    
    # Generate visualization
    if not args.no_plot and not args.baselines_only:
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATION")
        print("=" * 70)
        
        try:
            ppo_data = visualize_ppo_decisions(args.data_dir, "models/comparison_ppo.zip")
            dqn_data = visualize_dqn_decisions(args.data_dir, "models/comparison_dqn_best.pth", args.device)
            plot_path = plot_decisions(ppo_data, dqn_data, "visualizations/test_decisions.png")
            print(f"\n✅ Visualization saved to: {plot_path}")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")


if __name__ == "__main__":
    main()
