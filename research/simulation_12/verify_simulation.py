
import numpy as np
import torch
import pandas as pd
from uniswap_v3_dqn_paper import UniswapV3DQNEnv, ReplayBuffer, DuelingDDQNAgent, prepare_hourly_data_extended
from uniswap_v3_ppo_paper import UniswapV3PaperEnv

def verify_dqn_features():
    print("\n--- Verifying DQN Features & Fee Cap ---")
    data_dir = "training_data"
    hourly_data = prepare_hourly_data_extended(data_dir)
    
    # Initialize Env
    env = UniswapV3DQNEnv(hourly_data, mode="train")
    print(f"DQN State Dim: {env.state_dim} (Expected 38)")
    assert env.state_dim == 38, f"Expected 38, got {env.state_dim}"
    
    # Reset
    obs, _ = env.reset()
    print(f"Observation Shape: {obs.shape}")
    assert obs.shape == (38,), f"Expected (38,), got {obs.shape}"
    
    # Step 1: HOLD
    obs, reward, terminated, _, info = env.step(0)
    print(f"Step 1 (HOLD): Reward={reward:.4f}, In-range={info['in_range']}")
    
    # Step 2: Rebalance (Action 3 = Width 5 or 10, check map)
    # Action map: {0: None, 1: 1, 2: 3, 3: 5, 4: 10, ...}
    obs, reward, terminated, _, info = env.step(3) # Width 5
    print(f"Step 2 (Rebalance width=5): Reward={reward:.4f}, Action Width={info['action_width']}")
    
    # Check Features
    dist_to_boundary = obs[36]
    hours_since_rebalance = obs[37]
    print(f"New Features: Dist={dist_to_boundary:.4f}, Hours={hours_since_rebalance:.4f}")
    
    # After 1 step, hours should be 1/24 (approx 0.0417)
    expected_1h = 1.0/24.0
    assert abs(hours_since_rebalance - expected_1h) < 1e-4, f"Expected {expected_1h}, got {hours_since_rebalance}"
    
    # Dist should be > 0 if in range
    if info['in_range']:
        assert dist_to_boundary > 0.0, "Dist should be positive if in range"
    
    # Step 3: HOLD
    obs, reward, terminated, _, info = env.step(0)
    hours_since_rebalance = obs[37]
    print(f"Step 3 (HOLD): Hours={hours_since_rebalance:.4f}")
    expected_2h = 2.0 / 24.0
    assert abs(hours_since_rebalance - expected_2h) < 1e-4, f"Expected {expected_2h}, got {hours_since_rebalance}"
    
    print("DQN Features Verified ✅")

def verify_n_step_buffer():
    print("\n--- Verifying N-Step Buffer (n=3) ---")
    buffer = ReplayBuffer(capacity=100, n_steps=3, gamma=0.9)
    
    # Push 3 steps: reward=1 each
    # Step 0
    buffer.push(np.zeros(38), 0, 1.0, np.zeros(38), False)
    assert len(buffer) == 0, "Buffer should be empty (accumulating)"
    
    # Step 1
    buffer.push(np.zeros(38), 0, 1.0, np.zeros(38), False)
    assert len(buffer) == 0
    
    # Step 2
    buffer.push(np.zeros(38), 0, 1.0, np.ones(38), False) # Next state is ones
    assert len(buffer) == 1, "Buffer should have 1 item now"
    
    # Check the stored item
    s, a, r, ns, d = buffer.buffer[0]
    # Expected return: 1 + 0.9*1 + 0.81*1 = 2.71
    expected_r = 1.0 + 0.9 + 0.81
    print(f"Stored Reward: {r:.4f} (Expected {expected_r:.4f})")
    assert abs(r - expected_r) < 1e-5
    
    # Next state should be from step 2 (np.ones)
    assert np.allclose(ns, np.ones(38)), "Next state should be from 3 steps ahead"
    
    print("N-Step Buffer Verified ✅")

def verify_ppo_features():
    print("\n--- Verifying PPO Features ---")
    data_dir = "training_data"
    from uniswap_v3_dqn_paper import prepare_hourly_data_extended
    hourly_data = prepare_hourly_data_extended(data_dir) # PPO uses same data class now
    
    env = UniswapV3PaperEnv(hourly_data)
    print(f"PPO State Dim: {env.state_dim} (Expected 38)")
    assert env.state_dim == 38
    
    obs, _ = env.reset()
    print(f"Observation Shape: {obs.shape}")
    assert obs.shape == (38,)
    
    print("PPO Features Verified ✅")

if __name__ == "__main__":
    verify_dqn_features()
    verify_n_step_buffer()
    verify_ppo_features()
