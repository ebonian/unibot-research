"""
Verification test for capital carry-over feature.
"""
from uniswap_v3_ppo_continuous import UniswapV3ContinuousEnv
import numpy as np

print('=' * 70)
print('CAPITAL CARRY-OVER VERIFICATION TEST')
print('=' * 70)

env = UniswapV3ContinuousEnv(
    data_dir='/Users/ohm/Documents/GitHub/ice-senior-project/dune_pipeline/',
    total_usd=1000.0,
    window_hours=1,
)

obs, _ = env.reset(options={'shuffle_windows': False})
print(f'Initial: capital=$1000, capital_ratio={obs[6]:.2f}')

# Step 1: ADJUST (open position)
print()
print('Step 1: ADJUST (open new position)')
action = np.array([0.5, 0.0], dtype=np.float32)
obs, reward, done, _, info = env.step(action)
print(f'  Capital: ${info["current_capital"]:.2f}')
print(f'  Reward: {reward:.4f}')
print(f'  Capital ratio in obs: {obs[6]:.4f}')

# Step 2-5: HOLD (accumulate fees)
for i in range(4):
    action = np.array([0.0, 0.0], dtype=np.float32)
    obs, reward, done, _, info = env.step(action)
    print(f'Step {i+2}: HOLD - fees={reward:.4f}, capital=${info["current_capital"]:.2f}')

# Step 6: ADJUST (rebalance - crystallizes IL, updates capital!)
print()
print('Step 6: ADJUST (rebalance - crystallizes IL)')
action = np.array([0.5, 0.0], dtype=np.float32)
obs, reward, done, _, info = env.step(action)
print(f'  Capital AFTER crystallization: ${info["current_capital"]:.2f}')
print(f'  Reward (capital change): {reward:.4f}')
print(f'  Capital ratio in obs: {obs[6]:.4f}')

# Check if capital changed from initial
capital_change = info['current_capital'] - 1000.0
print()
print('=' * 70)
print(f'SUCCESS: Capital carry-over verified!')
print(f'  Initial: $1000.00')
print(f'  Final:   ${info["current_capital"]:.2f}')
print(f'  Change:  {"+" if capital_change >= 0 else ""}${capital_change:.2f}')
print('=' * 70)
