# uniswap_v3_ppo_continuous.py
import os
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fee_simulator import UniswapV3FeeSimulator, sqrt_price_x96_to_price


class UniswapV3ContinuousEnv(gym.Env):
    """
    Simple Uniswap v3 LP env for PPO with continuous actions.

    State (obs): shape (6,)
        [0] log(price_t0)
        [1] current width % of range (0 if no LP)
        [2] has_lp flag (1 if in LP, 0 otherwise)
        [3] volatility_24h (normalized 0-1)
        [4] price_change_pct (normalized -1 to 1)
        [5] in_range_flag (1 if price in LP bounds)

    Action: Box(-1, 1, shape=(2,))
        a[0] = mode selector in [-1, 1]
            if a[0] < -1/3:  BURN / NO LP  (exit; no LP this step)
            if -1/3 <= a[0] <= 1/3: HOLD (reuse current range if exists)
            if a[0] > 1/3:  ADJUST (set new range around current price)

        a[1] = width parameter in [-1, 1] used only when ADJUST:
            map to width % in [min_width_pct, max_width_pct], e.g. [0.1%, 1%]
            lower = price * (1 - width_pct)
            upper = price * (1 + width_pct)

    Reward per step:
        r = fees_usd + il_usd - gas (il_usd is negative for losses)

    Notes:
    - Each step is a fixed time window [t0, t1] (e.g. 24h).
    - We always deploy the same notional `total_usd` when we are in LP.
    - No capital carry-over; this is intended as a minimal, testable environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str,
        total_usd: float = 1000.0,
        window_hours: int = 1,  # 1-hour windows for more granular control
        gas_per_action_usd: float = 0.1,  # Realistic L2 (Arbitrum) gas cost
        min_width_pct: float = 0.001,  # 0.1%
        max_width_pct: float = 0.01,   # 1%
        mode: str = "train",  # "train" = first 80%, "eval" = last 20%, "all" = full data
        train_ratio: float = 0.8,  # Fraction of data for training
    ):
        super().__init__()

        # Load pool data
        pool_cfg = pd.read_csv(os.path.join(data_dir, "pool_config_eth_usdt_0p3.csv"))
        tokens = pd.read_csv(os.path.join(data_dir, "token_metadata_eth_usdt_0p3.csv"))
        slot0 = pd.read_csv(os.path.join(data_dir, "slot0_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv"))
        swaps = pd.read_csv(os.path.join(data_dir, "swaps_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv"))
        mints = pd.read_csv(os.path.join(data_dir, "mints_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv"))
        burns = pd.read_csv(os.path.join(data_dir, "burns_2025_09_01_to_2025_10_01_eth_usdt_0p3.csv"))
        ethusdt = pd.read_csv(os.path.join(os.path.dirname(__file__), "ETHUSDT_hourly_data_20241101_20251101.csv"))

        self.sim = UniswapV3FeeSimulator(
            pool_cfg=pool_cfg,
            tokens=tokens,
            slot0=slot0,
            swaps=swaps,
            mints=mints,
            burns=burns,
            eth_usdt_prices=ethusdt,
        )

        self.total_usd = float(total_usd)
        self.window_hours = int(window_hours)
        self.gas_per_action_usd = float(gas_per_action_usd)
        self.min_width_pct = float(min_width_pct)
        self.max_width_pct = float(max_width_pct)
        self.mode = mode
        self.train_ratio = train_ratio

        # Build time windows from swaps
        t_min = pd.to_datetime(swaps["evt_block_time"].min(), utc=True)
        t_max = pd.to_datetime(swaps["evt_block_time"].max(), utc=True)
        starts = pd.date_range(start=t_min, end=t_max, freq=f"{self.window_hours}h", tz="UTC")

        all_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s in starts[:-1]:
            e = s + pd.Timedelta(hours=self.window_hours)
            if e > t_max:
                break
            all_windows.append((s, e))

        # Apply train/eval split
        n_total = len(all_windows)
        split_idx = int(n_total * train_ratio)
        
        if mode == "train":
            windows = all_windows[:split_idx]
            print(f"📊 TRAIN mode: using windows 0-{split_idx} ({len(windows)} windows)")
        elif mode == "eval":
            windows = all_windows[split_idx:]
            print(f"📊 EVAL mode: using windows {split_idx}-{n_total} ({len(windows)} windows)")
        else:  # "all"
            windows = all_windows
            print(f"📊 ALL mode: using all {len(windows)} windows")

        self.windows = windows
        self.n_windows = len(windows)

        # Store swaps for volatility calculation
        self.swaps_df = swaps.copy()
        self.swaps_df['evt_block_time'] = pd.to_datetime(self.swaps_df['evt_block_time'], utc=True)
        
        # Gym spaces - EXPANDED observation (7 dimensions with capital)
        # obs: [log_price, width_pct, has_lp, volatility_24h, price_change_pct, in_range_flag, capital_ratio]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32),
        )

        # action: [mode, width_param] in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state
        self.idx = 0
        self.has_lp = False
        self.current_lower = None
        self.current_upper = None
        self.current_width_pct = 0.0
        self.prev_price = None  # Track previous price for change calculation
        
        # Realistic IL tracking: only crystallize on rebalance/burn
        self.position_entry_time = None      # When LP was minted
        self.position_entry_price = None     # Price at entry
        self.accumulated_fees = 0.0          # Fees earned since entry (unrealized)
        
        # Capital carry-over: track actual portfolio value
        self.initial_capital = float(total_usd)  # Starting capital for reference
        self.current_capital = float(total_usd)  # Actual portfolio value (changes with IL/fees)

    # ---------- helpers ----------

    def _get_price_at(self, ts: pd.Timestamp) -> float:
        init_state = self.sim._init_state_at_t0(ts)
        price_dec = sqrt_price_x96_to_price(
            init_state.sqrt_price_x96,
            self.sim.cfg.decimals0,
            self.sim.cfg.decimals1,
        )
        return float(price_dec)

    def _get_volatility_24h(self, ts: pd.Timestamp) -> float:
        """Calculate 24h price volatility (std of log returns) ending at ts."""
        t_start = ts - pd.Timedelta(hours=24)
        mask = (self.swaps_df['evt_block_time'] >= t_start) & (self.swaps_df['evt_block_time'] < ts)
        swaps_window = self.swaps_df[mask]
        
        if len(swaps_window) < 10:
            return 0.0
        
        # Calculate price from sqrtPriceX96
        Q96 = 2**96
        prices = swaps_window['sqrtPriceX96'].apply(
            lambda x: (float(x) / Q96) ** 2 * (10 ** (self.sim.cfg.decimals0 - self.sim.cfg.decimals1))
        )
        
        # Log returns volatility
        log_returns = np.log(prices).diff().dropna()
        if len(log_returns) < 2:
            return 0.0
        
        vol = float(log_returns.std())
        # Normalize: typical intraday log return std is 0.0001-0.001, scale up
        return min(vol * 1000, 1.0)  # Scale and cap at 1.0

    def _get_obs(self) -> np.ndarray:
        if self.idx >= self.n_windows:
            # Terminal - return 7 zeros
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        t0, _ = self.windows[self.idx]
        price = self._get_price_at(t0)
        log_price = math.log(price)
        has_lp_flag = 1.0 if self.has_lp else 0.0
        
        # NEW: Volatility (normalized 0-1)
        volatility = self._get_volatility_24h(t0)
        
        # NEW: Price change since last step (normalized, clipped to [-1, 1])
        if self.prev_price is not None and self.prev_price > 0:
            price_change_pct = (price - self.prev_price) / self.prev_price
            price_change_pct = max(-1.0, min(1.0, price_change_pct * 10))  # Scale by 10x, clip
        else:
            price_change_pct = 0.0
        
        # NEW: In-range indicator (1.0 if current price is within LP bounds)
        if self.has_lp and self.current_lower and self.current_upper:
            in_range = 1.0 if self.current_lower <= price <= self.current_upper else 0.0
        else:
            in_range = 0.0

        # Capital ratio: current_capital / initial_capital (capped at 2.0 for obs space)
        capital_ratio = min(self.current_capital / self.initial_capital, 2.0) if self.initial_capital > 0 else 1.0

        obs = np.array([
            log_price,           # [0] Current log price
            self.current_width_pct,  # [1] Current width (0-1)
            has_lp_flag,         # [2] Has LP position
            volatility,          # [3] 24h volatility (0-1)
            price_change_pct,    # [4] Price change since last step (-1 to 1)
            in_range,            # [5] Is current price in LP range
            capital_ratio,       # [6] Current capital / initial capital (0-2)
        ], dtype=np.float32)
        return obs

    def _width_from_action(self, a_width: float) -> float:
        """
        Map a_width in [-1,1] to width_pct in [min_width_pct, max_width_pct].
        Then clip to [0,1] for observation.
        """
        frac = (a_width + 1.0) / 2.0  # in [0,1]
        width_pct = self.min_width_pct + frac * (self.max_width_pct - self.min_width_pct)
        return float(width_pct)

    # ---------- gymnasium API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            options = {}
        shuffle = options.get("shuffle_windows", True)
        if shuffle:
            self.np_random.shuffle(self.windows)  # Use seeded RNG for reproducibility

        self.idx = 0
        self.has_lp = False
        self.current_lower = None
        self.current_upper = None
        self.current_width_pct = 0.0
        self.prev_price = None  # Reset previous price
        
        # Reset realistic IL tracking
        self.position_entry_time = None
        self.position_entry_price = None
        self.accumulated_fees = 0.0
        
        # Reset capital to initial value
        self.current_capital = self.initial_capital

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), "Invalid action"
        if self.idx >= self.n_windows:
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        t0, t1 = self.windows[self.idx]
        price_t0 = self._get_price_at(t0)

        mode = float(action[0])
        a_width = float(action[1])
        reward = 0.0

        # Decide mode:
        #   mode < -1/3   -> BURN / NO LP
        #   -1/3..+1/3    -> HOLD
        #   mode > +1/3   -> ADJUST (new continuous range)
        burn_threshold = -1.0 / 3.0
        adjust_threshold = 1.0 / 3.0

        if mode < burn_threshold:
            # BURN / NO LP - Crystallize IL + accumulated fees
            if self.has_lp and self.position_entry_time is not None:
                # Calculate crystallized IL from entry to now
                try:
                    crystallize_summary = self.sim.simulate(
                        price_lower=self.current_lower,
                        price_upper=self.current_upper,
                        start=self.position_entry_time.isoformat(),
                        end=t0.isoformat(),
                        liquidity=None,
                        total_usd=self.current_capital,  # Use current capital
                        validate=False,
                        use_swap_liquidity=False,
                        accounting_mode="growth",
                        protocol_fee_encoding="base256",
                    )
                    crystallized_il = crystallize_summary.impermanent_loss.usd
                except Exception:
                    # Fallback if simulation fails (e.g., time range issues)
                    crystallized_il = 0.0
                
                # Update capital: add fees and IL (IL is negative for losses)
                capital_change = self.accumulated_fees + crystallized_il - self.gas_per_action_usd
                self.current_capital += capital_change
                self.current_capital = max(0.0, self.current_capital)  # Can't go negative
                
                # Reward = capital change
                reward = float(capital_change)
            elif self.has_lp:
                # Had LP but no entry time (shouldn't happen, but handle gracefully)
                capital_change = self.accumulated_fees - self.gas_per_action_usd
                self.current_capital += capital_change
                self.current_capital = max(0.0, self.current_capital)
                reward = float(capital_change)
            
            # Reset position state
            self.has_lp = False
            self.current_lower = None
            self.current_upper = None
            self.current_width_pct = 0.0
            self.position_entry_time = None
            self.position_entry_price = None
            self.accumulated_fees = 0.0

        elif mode <= adjust_threshold:
            # HOLD - Only count fees, IL is unrealized (paper loss)
            if self.has_lp and self.current_lower is not None and self.current_upper is not None:
                # Simulate this window to get fees
                summary = self.sim.simulate(
                    price_lower=self.current_lower,
                    price_upper=self.current_upper,
                    start=t0.isoformat(),
                    end=t1.isoformat(),
                    liquidity=None,
                    total_usd=self.current_capital,  # Use current capital
                    validate=False,
                    use_swap_liquidity=False,
                    accounting_mode="growth",
                    protocol_fee_encoding="base256",
                )
                fees_usd = summary.fees_usd.token1
                
                # REALISTIC: Only count fees, IL is unrealized
                self.accumulated_fees += float(fees_usd)
                reward = float(fees_usd)  # Immediate reward is just fees
            else:
                # no LP -> nothing happens
                pass

        else:
            # ADJUST: Crystallize old position IL + accumulated fees, open new position
            width_pct = self._width_from_action(a_width)
            lower = price_t0 * (1.0 - width_pct)
            upper = price_t0 * (1.0 + width_pct)
            
            # First: If we had an existing position, crystallize it
            if self.has_lp and self.position_entry_time is not None:
                try:
                    crystallize_summary = self.sim.simulate(
                        price_lower=self.current_lower,
                        price_upper=self.current_upper,
                        start=self.position_entry_time.isoformat(),
                        end=t0.isoformat(),
                        liquidity=None,
                        total_usd=self.current_capital,  # Use current capital
                        validate=False,
                        use_swap_liquidity=False,
                        accounting_mode="growth",
                        protocol_fee_encoding="base256",
                    )
                    crystallized_il = crystallize_summary.impermanent_loss.usd
                except Exception:
                    crystallized_il = 0.0
                
                # Update capital and reward from closing old position
                capital_change = self.accumulated_fees + crystallized_il - self.gas_per_action_usd
                self.current_capital += capital_change
                self.current_capital = max(0.0, self.current_capital)
                reward += float(capital_change)
            elif self.has_lp:
                # Had LP but no entry time
                capital_change = self.accumulated_fees - self.gas_per_action_usd
                self.current_capital += capital_change
                self.current_capital = max(0.0, self.current_capital)
                reward += float(capital_change)
            else:
                # First time opening LP, just pay gas
                self.current_capital -= self.gas_per_action_usd
                self.current_capital = max(0.0, self.current_capital)
                reward -= self.gas_per_action_usd
            
            # Now simulate this window for the NEW position to get initial fees
            summary = self.sim.simulate(
                price_lower=lower,
                price_upper=upper,
                start=t0.isoformat(),
                end=t1.isoformat(),
                liquidity=None,
                total_usd=self.current_capital,  # Use updated capital
                validate=False,
                use_swap_liquidity=False,
                accounting_mode="growth",
                protocol_fee_encoding="base256",
            )
            fees_usd = summary.fees_usd.token1

            # Update position state - new position starts NOW
            self.has_lp = True
            self.current_lower = lower
            self.current_upper = upper
            self.current_width_pct = min(width_pct / self.max_width_pct, 1.0)
            self.position_entry_time = t0
            self.position_entry_price = price_t0
            self.accumulated_fees = float(fees_usd)  # Start accumulating from this window

        # Move to next window
        self.idx += 1
        self.prev_price = price_t0  # Track price for next observation
        terminated = self.idx >= self.n_windows
        truncated = False
        obs = self._get_obs()
        info = {
            "t0": t0, 
            "t1": t1, 
            "price_t0": price_t0, 
            "accumulated_fees": self.accumulated_fees, 
            "current_capital": self.current_capital,
            "current_lower": self.current_lower,
            "current_upper": self.current_upper,
            "has_lp": self.has_lp,
        }

        return obs, reward, terminated, truncated, info


def make_env_fn(
    data_dir: str,
    total_usd: float,
    window_hours: int,
    gas_per_action_usd: float,
    min_width_pct: float = 0.001,  # 0.1%
    max_width_pct: float = 0.01,   # 1%
    mode: str = "train",  # "train", "eval", or "all"
):
    def _init():
        return UniswapV3ContinuousEnv(
            data_dir=data_dir,
            total_usd=total_usd,
            window_hours=window_hours,
            gas_per_action_usd=gas_per_action_usd,
            min_width_pct=min_width_pct,
            max_width_pct=max_width_pct,
            mode=mode,
        )
    return _init


if __name__ == "__main__":
    # Adjust to your data path (where CSVs + fee_simulator.py live)
    DATA_DIR = "/Users/ohm/Documents/GitHub/ice-senior-project/dune_pipeline/"  # e.g. "/mnt/data"

    env = DummyVecEnv(
        [make_env_fn(
            DATA_DIR,
            total_usd=1000.0,
            window_hours=1,  # 1-hour windows for more opportunities
            gas_per_action_usd=0.1,  # Realistic L2 gas cost (Arbitrum)
            min_width_pct=0.001,  # 0.1%
            max_width_pct=0.01,   # 1%
        )]
    )
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )

    # Simple training run (more steps needed for 1h windows = 720 steps/month)
    model.learn(total_timesteps=100_000)
    model.save("ppo_uniswap_v3_continuous")
    env.save("vec_normalize.pkl")

    # Example inference after training:
    # from stable_baselines3.common.vec_env import VecNormalize
    #
    # # 1. Re-create the base environment
    # env = DummyVecEnv([make_env_fn(DATA_DIR, total_usd=1000.0, window_hours=24, gas_per_action_usd=0.0)])
    #
    # # 2. Load the saved statistics
    # env = VecNormalize.load("vec_normalize.pkl", env)
    #
    # # 3. Disable training-only behavior for inference
    # env.training = False
    # env.norm_reward = False
    #
    # # 4. Load the model
    # model = PPO.load("ppo_uniswap_v3_continuous", env=env)
    #
    # obs, _ = env.reset()
    # action, _ = model.predict(obs, deterministic=True)
    # print("Action:", action)
