"""
Test the trained PPO agent on full episode.
Shows environment stats and action distribution.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uniswap_v3_ppo_continuous import make_env_fn, UniswapV3ContinuousEnv

DATA_DIR = '/Users/ohm/Documents/GitHub/ice-senior-project/dune_pipeline/'

# Show environment stats
raw_env = UniswapV3ContinuousEnv(data_dir=DATA_DIR, window_hours=1, gas_per_action_usd=0.1)
print('='*70)
print('ENVIRONMENT STATS')
print('='*70)
print(f'Window size:        {raw_env.window_hours} hour(s)')
print(f'Total windows:      {raw_env.n_windows} (1 episode = {raw_env.n_windows} steps)')
print(f'Data period:        Sept 1 - Oct 1, 2025')
print(f'Observation dims:   {raw_env.observation_space.shape[0]}')
print(f'Action dims:        {raw_env.action_space.shape[0]}')
print()

# Load model and run full episode
env = DummyVecEnv([make_env_fn(DATA_DIR, 1000.0, 1, 0.1, 0.001, 0.01)])
env = VecNormalize.load('vec_normalize.pkl', env)
env.training = False
env.norm_reward = False
model = PPO.load('ppo_uniswap_v3_continuous', env=env)

print('='*70)
print('RUNNING FULL EPISODE')
print('='*70)
print()

obs = env.reset()
total_reward = 0
actions_count = {'BURN': 0, 'HOLD': 0, 'ADJUST': 0}
rewards = []

window_num = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    mode = action[0][0]
    action_name = 'BURN' if mode < -0.33 else ('HOLD' if mode <= 0.33 else 'ADJUST')
    actions_count[action_name] += 1
    
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    rewards.append(reward[0])
    window_num += 1
    
    # Print every 100 windows
    if window_num % 100 == 0:
        print(f'Window {window_num}: cumulative profit = ${total_reward:.2f}')
    
    if done[0]:
        break

print()
print('='*70)
print('EPISODE SUMMARY')
print('='*70)
print(f'Total windows:      {window_num}')
print(f'Total hours:        {window_num} hours (~{window_num/24:.1f} days)')
print(f'Total profit:       ${total_reward:.2f}')
print(f'Avg profit/hour:    ${total_reward/window_num:.2f}')
print(f'Avg profit/day:     ${total_reward/window_num*24:.2f}')
print()
print('Action Distribution:')
for k, v in actions_count.items():
    pct = v/window_num*100
    print(f'  {k:6s}: {v:4d} ({pct:5.1f}%)')
print()
print(f'Win rate:           {sum(1 for r in rewards if r > 0)/len(rewards)*100:.1f}%')
print('='*70)
