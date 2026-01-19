"""
Evaluate trained PPO agent and visualize its LP decisions.
Generates a chart showing price, LP windows, and per-window metrics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uniswap_v3_ppo_continuous import UniswapV3ContinuousEnv, make_env_fn


def evaluate_agent(
    model_path: str = "ppo_uniswap_v3_continuous.zip",
    vec_normalize_path: str = "vec_normalize.pkl",
    data_dir: str = "/Users/ohm/Documents/GitHub/ice-senior-project/dune_pipeline/",
    num_episodes: int = 1,
    max_steps: int = 100,
    save_chart: str = "agent_evaluation_chart.png",
):
    """
    Load trained model and run evaluation with visualization.
    
    Args:
        model_path: Path to saved PPO model
        vec_normalize_path: Path to saved VecNormalize stats
        data_dir: Path to data directory
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        save_chart: Path to save the visualization
    """
    print("=" * 70)
    print("🤖 PPO AGENT EVALUATION")
    print("=" * 70)
    
    # Create environment in EVAL mode (uses last 20% of data)
    env = DummyVecEnv([make_env_fn(
        data_dir,
        total_usd=1000.0,
        window_hours=1,
        gas_per_action_usd=0.1,
        min_width_pct=0.001,
        max_width_pct=0.01,
        mode="eval",  # Use held-out evaluation data
    )])
    
    # Load normalization stats if available
    if os.path.exists(vec_normalize_path):
        try:
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ Loaded VecNormalize from {vec_normalize_path}")
        except AssertionError as e:
            print(f"⚠️  VecNormalize shape mismatch (model may be from old version)")
            print(f"   {e}")
            print(f"   Continuing without normalization...")
    else:
        print(f"⚠️  No VecNormalize found, using unnormalized env")
    
    # Load model
    model = None
    if os.path.exists(model_path):
        try:
            model = PPO.load(model_path, env=env)
            print(f"✅ Loaded model from {model_path}")
        except ValueError as e:
            print(f"⚠️  Model observation space mismatch (model trained with old env)")
            print(f"   {e}")
            model = None
    else:
        print(f"⚠️  Model not found at {model_path}")
    
    if model is None:
        print(f"\n🎲 RUNNING IN DEMO MODE (random actions)")
        print(f"   To use a trained model, run: python uniswap_v3_ppo_continuous.py")
    
    # Run evaluation
    print(f"\n📊 Running {num_episodes} episode(s)...")
    
    all_history = []
    total_reward = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_history = []
        
        for step in range(max_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Random action in demo mode
                action = np.array([env.action_space.sample()])
            obs, reward, done, info = env.step(action)
            
            # Decode action
            mode = float(action[0][0])
            width_param = float(action[0][1])
            
            if mode < -1/3:
                action_name = "BURN"
            elif mode <= 1/3:
                action_name = "HOLD"
            else:
                action_name = "ADJUST"
            
            # Extract from info dict
            inf = info[0]
            
            episode_history.append({
                'step': step,
                't0': inf.get('t0'),
                't1': inf.get('t1'),
                'price': inf.get('price_t0', 0),
                'action': action_name,
                'mode_raw': mode,
                'width_param': width_param,
                'reward': float(reward[0]),
                'capital': inf.get('current_capital', 1000),
                'lower': inf.get('current_lower'),
                'upper': inf.get('current_upper'),
                'has_lp': inf.get('has_lp', False),
            })
            
            episode_reward += float(reward[0])
            
            if done[0]:
                break
        
        all_history.extend(episode_history)
        total_reward += episode_reward
        print(f"   Episode {ep+1}: reward={episode_reward:.2f}, steps={len(episode_history)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_history)
    
    # Print summary
    print(f"\n📈 EVALUATION SUMMARY:")
    print(f"   Total Reward:     {total_reward:.2f}")
    print(f"   Final Capital:    ${df['capital'].iloc[-1]:.2f}")
    print(f"   Capital Change:   {((df['capital'].iloc[-1] / 1000) - 1) * 100:.2f}%")
    
    # Action distribution
    action_counts = df['action'].value_counts()
    print(f"\n🎯 ACTION DISTRIBUTION:")
    for action, count in action_counts.items():
        pct = count / len(df) * 100
        print(f"   {action}: {count} ({pct:.1f}%)")
    
    # Generate visualization
    print(f"\n📊 GENERATING VISUALIZATION...")
    fig = plot_agent_decisions(df, save_path=save_chart)
    
    print(f"\n✅ Chart saved to: {save_chart}")
    print("=" * 70)
    
    return df


def plot_agent_decisions(df, save_path=None):
    """
    Plot agent decisions: price line with LP windows.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Filter rows with valid timestamps and SORT by time
    df_valid = df[df['t0'].notna()].copy()
    df_valid['time'] = pd.to_datetime(df_valid['t0'])
    df_valid = df_valid.sort_values('time').reset_index(drop=True)
    
    # === Panel 1: Price with LP Windows ===
    ax1 = axes[0]
    ax1.plot(df_valid['time'], df_valid['price'], 
             color='#2196F3', linewidth=2, label='ETH Price', zorder=3)
    
    # Color code by action
    colors = {'HOLD': '#A5D6A7', 'ADJUST': '#90CAF9', 'BURN': '#FFAB91'}
    
    for idx in range(len(df_valid)):
        row = df_valid.iloc[idx]
        if row['has_lp'] and row['lower'] is not None and row['upper'] is not None:
            color = colors.get(row['action'], '#E0E0E0')
            
            # Get next row for end time
            if idx + 1 < len(df_valid):
                end_time = df_valid.iloc[idx + 1]['time']
            else:
                end_time = row['time'] + pd.Timedelta(hours=1)
            
            rect = mpatches.Rectangle(
                (row['time'], row['lower']),
                end_time - row['time'],
                row['upper'] - row['lower'],
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=0.4,
                zorder=1,
            )
            ax1.add_patch(rect)
    
    ax1.set_ylabel('ETH Price (USD)', fontsize=11)
    ax1.set_title('Trained Agent: LP Position Decisions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left')
    
    # === Panel 2: Actions over time ===
    ax2 = axes[1]
    action_map = {'HOLD': 0, 'ADJUST': 1, 'BURN': -1}
    df_valid['action_num'] = df_valid['action'].map(action_map)
    
    ax2.bar(df_valid['time'], df_valid['action_num'], 
            color=[colors.get(a, 'gray') for a in df_valid['action']], 
            width=0.03, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Action', fontsize=11)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['BURN', 'HOLD', 'ADJUST'])
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # === Panel 3: Cumulative reward / Capital ===
    ax3 = axes[2]
    ax3.plot(df_valid['time'], df_valid['capital'], 
             color='#4CAF50', linewidth=2, label='Capital ($)')
    ax3.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Initial $1000')
    ax3.fill_between(df_valid['time'], 1000, df_valid['capital'], 
                     where=df_valid['capital'] >= 1000, alpha=0.3, color='green')
    ax3.fill_between(df_valid['time'], 1000, df_valid['capital'], 
                     where=df_valid['capital'] < 1000, alpha=0.3, color='red')
    ax3.set_ylabel('Capital ($)', fontsize=11)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent')
    parser.add_argument('--model', default='ppo_uniswap_v3_continuous.zip', help='Path to model')
    parser.add_argument('--normalize', default='vec_normalize.pkl', help='Path to VecNormalize')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--output', default='agent_evaluation_chart.png', help='Output chart path')
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model,
        vec_normalize_path=args.normalize,
        num_episodes=args.episodes,
        max_steps=args.steps,
        save_chart=args.output,
    )
