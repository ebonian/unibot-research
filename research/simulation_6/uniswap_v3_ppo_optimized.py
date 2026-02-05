# uniswap_v3_ppo_optimized.py
"""
Optimized Uniswap v3 PPO Training

Key optimizations:
1. SubprocVecEnv with multiple parallel environments (8x speedup)
2. Precomputed prices and volatility (eliminates redundant calculations)
3. Optional: use full fee_simulator for exact fees/IL (--full-sim), or fast approximations
4. Better hyperparameters for faster convergence
"""

import os
import math
import glob
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from fee_simulator import UniswapV3FeeSimulator

Q96 = 2 ** 96


def _load_full_simulator(data_dir: str) -> UniswapV3FeeSimulator:
    """Load UniswapV3FeeSimulator from data_dir (same files as original env)."""
    pool_cfg = pd.read_csv(os.path.join(data_dir, "pool_config_eth_usdt_0p3.csv"))
    tokens = pd.read_csv(os.path.join(data_dir, "token_metadata_eth_usdt_0p3.csv"))
    slot0_files = glob.glob(os.path.join(data_dir, "slot0_*_eth_usdt_0p3.csv"))
    swaps_files = glob.glob(os.path.join(data_dir, "swaps_*_eth_usdt_0p3.csv"))
    mints_files = glob.glob(os.path.join(data_dir, "mints_*_eth_usdt_0p3.csv"))
    burns_files = glob.glob(os.path.join(data_dir, "burns_*_eth_usdt_0p3.csv"))
    ethusdt_files = glob.glob(os.path.join(data_dir, "ETHUSDT_hourly_data_*.csv"))

    if not all([slot0_files, swaps_files, mints_files, burns_files]):
        raise FileNotFoundError(
            f"Missing required data in {data_dir}: need slot0_*, swaps_*, mints_*, burns_*"
        )
    slot0 = pd.read_csv(slot0_files[0])
    swaps = pd.read_csv(swaps_files[0], low_memory=False)
    mints = pd.read_csv(mints_files[0], low_memory=False)
    burns = pd.read_csv(burns_files[0], low_memory=False)
    ethusdt = pd.read_csv(ethusdt_files[0]) if ethusdt_files else None
    return UniswapV3FeeSimulator(
        pool_cfg=pool_cfg,
        tokens=tokens,
        slot0=slot0,
        swaps=swaps,
        mints=mints,
        burns=burns,
        eth_usdt_prices=ethusdt,
    )


@dataclass
class PrecomputedData:
    """Precomputed data shared across all environments."""
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
    prices: Dict[pd.Timestamp, float]  # t0 -> price
    volatilities: Dict[pd.Timestamp, float]  # t0 -> 24h volatility
    fee_rates: Dict[pd.Timestamp, float]  # t0 -> estimated fee rate per USD per hour
    decimals0: int
    decimals1: int
    pool_fee_bps: int


def sqrt_price_x96_to_price_fast(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Fast float version of price conversion (good enough for training)."""
    p = float(sqrt_price_x96) / Q96
    return (p * p) * (10 ** (decimals0 - decimals1))


def precompute_training_data(data_dir: str, window_hours: int = 1) -> PrecomputedData:
    """
    Precompute all prices, volatilities, and fee estimates upfront.
    This is done ONCE before training starts.
    """
    print("🔄 Precomputing training data (this runs once)...")
    
    # Load data
    pool_cfg = pd.read_csv(os.path.join(data_dir, "pool_config_eth_usdt_0p3.csv"))
    tokens = pd.read_csv(os.path.join(data_dir, "token_metadata_eth_usdt_0p3.csv"))
    
    slot0_files = glob.glob(os.path.join(data_dir, "slot0_*_eth_usdt_0p3.csv"))
    swaps_files = glob.glob(os.path.join(data_dir, "swaps_*_eth_usdt_0p3.csv"))
    
    if not slot0_files or not swaps_files:
        raise FileNotFoundError(f"Missing required data files in {data_dir}")
    
    slot0 = pd.read_csv(slot0_files[0])
    swaps = pd.read_csv(swaps_files[0], low_memory=False)
    
    # Get decimals
    tokens['contract_address'] = tokens['contract_address'].str.lower()
    t0_addr = pool_cfg.loc[0, 'token0'].lower()
    t1_addr = pool_cfg.loc[0, 'token1'].lower()
    t0 = tokens.set_index('contract_address').loc[t0_addr]
    t1 = tokens.set_index('contract_address').loc[t1_addr]
    decimals0 = int(t0['decimals'])
    decimals1 = int(t1['decimals'])
    pool_fee_bps = int(pool_cfg.loc[0, 'fee'])
    
    # Parse timestamps
    swaps['evt_block_time'] = pd.to_datetime(swaps['evt_block_time'], utc=True)
    swaps = swaps.sort_values('evt_block_time').reset_index(drop=True)
    
    # Build time windows
    t_min = swaps['evt_block_time'].min()
    t_max = swaps['evt_block_time'].max()
    starts = pd.date_range(start=t_min, end=t_max, freq=f"{window_hours}h", tz="UTC")
    
    windows = []
    for s in starts[:-1]:
        e = s + pd.Timedelta(hours=window_hours)
        if e > t_max:
            break
        windows.append((s, e))
    
    print(f"  📊 Built {len(windows)} time windows")
    
    # Precompute prices for each window start
    prices = {}
    swaps_indexed = swaps.set_index('evt_block_time').sort_index()
    
    for t0_ts, _ in windows:
        # Get price from most recent swap strictly before t0 (match original _init_state_at_t0)
        swaps_before = swaps_indexed[swaps_indexed.index < t0_ts]
        if len(swaps_before) > 0:
            sqrt_price = int(swaps_before.iloc[-1]['sqrtPriceX96'])
        else:
            sqrt_price = int(swaps_indexed.iloc[0]['sqrtPriceX96'])
        prices[t0_ts] = sqrt_price_x96_to_price_fast(sqrt_price, decimals0, decimals1)
    
    print(f"  💰 Precomputed {len(prices)} prices")
    
    # Precompute 24h volatility for each window
    volatilities = {}
    swaps['price'] = swaps['sqrtPriceX96'].apply(
        lambda x: sqrt_price_x96_to_price_fast(int(x), decimals0, decimals1)
    )
    
    for t0_ts, _ in windows:
        t_start = t0_ts - pd.Timedelta(hours=24)
        mask = (swaps['evt_block_time'] >= t_start) & (swaps['evt_block_time'] < t0_ts)
        swaps_window = swaps[mask]
        
        if len(swaps_window) < 10:
            volatilities[t0_ts] = 0.0
        else:
            log_returns = np.log(swaps_window['price']).diff().dropna()
            if len(log_returns) < 2:
                volatilities[t0_ts] = 0.0
            else:
                vol = float(log_returns.std())
                volatilities[t0_ts] = min(vol * 1000, 1.0)  # Normalize
    
    print(f"  📈 Precomputed {len(volatilities)} volatilities")
    
    # Estimate fee rates for each window (fees per $1 of liquidity per hour)
    # This is an approximation based on swap volume
    fee_rates = {}
    fee_rate = pool_fee_bps / 1_000_000  # Convert bps to decimal
    
    for t0_ts, t1_ts in windows:
        mask = (swaps['evt_block_time'] >= t0_ts) & (swaps['evt_block_time'] < t1_ts)
        swaps_window = swaps[mask]
        
        if len(swaps_window) == 0:
            # No volume => no fees; correct to use 0 (you earn 0 when not in pool, and 0 in this window if you were)
            fee_rates[t0_ts] = 0.0
        else:
            # Estimate: fee per USD per window. Use amount1 (USDT) in token units.
            total_volume_usd = swaps_window['amount1'].abs().sum() / (10 ** decimals1)
            # Pool fee from volume; assume our position gets a tiny share (e.g. $1000 in $10M pool)
            # So fee_per_1000_usd = total_volume_usd * fee_rate * (1000 / pool_liquidity_approx)
            # Use conservative cap: max 0.001 USD per USD per window (~0.1% of capital per step)
            raw_per_usd = (total_volume_usd * fee_rate) / max(total_volume_usd, 1e6)
            fee_rates[t0_ts] = min(float(raw_per_usd), 0.001)  # Cap to prevent explosion
    
    print(f"  💵 Precomputed {len(fee_rates)} fee rate estimates")
    print("✅ Precomputation complete!")
    
    return PrecomputedData(
        windows=windows,
        prices=prices,
        volatilities=volatilities,
        fee_rates=fee_rates,
        decimals0=decimals0,
        decimals1=decimals1,
        pool_fee_bps=pool_fee_bps,
    )


class UniswapV3FastEnv(gym.Env):
    """
    Uniswap v3 LP environment for PPO training.
    
    - use_full_sim=False: fast approximations for fees/IL (precomputed fee_rates).
    - use_full_sim=True: exact fees/IL via fee_simulator.simulate() (slower, correct).
    Observations always use precomputed prices/volatility for speed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        precomputed: PrecomputedData,
        total_usd: float = 1000.0,
        gas_per_action_usd: float = 0.1,
        min_width_pct: float = 0.001,
        max_width_pct: float = 0.01,
        mode: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        data_dir: Optional[str] = None,
        use_full_sim: bool = False,
    ):
        super().__init__()
        
        self.precomputed = precomputed
        self.total_usd = float(total_usd)
        self.gas_per_action_usd = float(gas_per_action_usd)
        self.min_width_pct = float(min_width_pct)
        self.max_width_pct = float(max_width_pct)
        self.use_full_sim = use_full_sim
        self.sim: Optional[UniswapV3FeeSimulator] = None
        if use_full_sim and data_dir:
            self.sim = _load_full_simulator(data_dir)
        
        # 80/10/10 split: train / val / test
        n_total = len(precomputed.windows)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        if mode == "train":
            self.windows = precomputed.windows[:train_end]
        elif mode == "eval":
            self.windows = precomputed.windows[train_end:val_end]
        elif mode == "test":
            self.windows = precomputed.windows[val_end:]
        else:
            self.windows = precomputed.windows
        
        self.n_windows = len(self.windows)
        
        # Observation space: 7 dimensions
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32),
        )
        
        # Action space: [mode, width]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Episode state
        self._reset_state()

    def _reset_state(self):
        self.idx = 0
        self.has_lp = False
        self.current_lower = None
        self.current_upper = None
        self.current_width_pct = 0.0
        self.prev_price = None
        self.position_entry_price = None
        self.position_entry_time = None  # for full sim crystallize
        self.accumulated_fees = 0.0
        self.initial_capital = self.total_usd
        self.current_capital = self.total_usd

    def _get_price(self, t0: pd.Timestamp) -> float:
        return self.precomputed.prices.get(t0, 0.0)

    def _get_volatility(self, t0: pd.Timestamp) -> float:
        return self.precomputed.volatilities.get(t0, 0.0)

    def _get_fee_rate(self, t0: pd.Timestamp) -> float:
        return self.precomputed.fee_rates.get(t0, 0.0)

    def _estimate_fees(self, t0: pd.Timestamp, width_pct: float, capital: float) -> float:
        """
        Fast fee estimation based on precomputed data.
        
        Narrower ranges = higher fee share (more concentrated liquidity)
        Uses: fee_rate * capital * concentration_bonus
        """
        base_fee_rate = self._get_fee_rate(t0)
        
        # Concentration bonus: narrower range = higher share of fees
        # At 1% width, bonus = 1.0; at 0.1% width, bonus = ~3x
        concentration_bonus = 1.0 / (width_pct * 100 + 0.1)
        concentration_bonus = min(concentration_bonus, 5.0)  # Cap at 5x
        
        estimated_fees = base_fee_rate * capital * concentration_bonus
        # Cap at 1% of capital per step to keep rewards stable and prevent explosion
        estimated_fees = min(estimated_fees, capital * 0.01)
        return max(0.0, estimated_fees)

    def _estimate_il(self, entry_price: float, current_price: float, 
                     width_pct: float, capital: float) -> float:
        """
        Fast IL estimation using simplified formula.
        
        IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        Adjusted for concentrated liquidity (narrower = higher IL risk)
        """
        if entry_price <= 0 or current_price <= 0:
            return 0.0
        
        price_ratio = current_price / entry_price
        
        # Basic IL calculation
        sqrt_ratio = math.sqrt(price_ratio)
        il_pct = 2 * sqrt_ratio / (1 + price_ratio) - 1
        
        # Concentrated liquidity amplifies IL when price moves out of range
        # Narrower range = more severe IL
        amplification = 1.0 / (width_pct * 10 + 0.1)
        amplification = min(amplification, 10.0)
        
        il_usd = il_pct * capital * amplification
        return il_usd  # Negative for losses

    def _get_obs(self) -> np.ndarray:
        if self.idx >= self.n_windows:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        t0, _ = self.windows[self.idx]
        price = self._get_price(t0)
        log_price = math.log(price) if price > 0 else 0.0
        has_lp_flag = 1.0 if self.has_lp else 0.0
        
        volatility = self._get_volatility(t0)
        
        if self.prev_price is not None and self.prev_price > 0:
            price_change_pct = (price - self.prev_price) / self.prev_price
            price_change_pct = max(-1.0, min(1.0, price_change_pct * 10))
        else:
            price_change_pct = 0.0
        
        if self.has_lp and self.current_lower and self.current_upper:
            in_range = 1.0 if self.current_lower <= price <= self.current_upper else 0.0
        else:
            in_range = 0.0

        capital_ratio = min(self.current_capital / self.initial_capital, 2.0)

        return np.array([
            log_price,
            self.current_width_pct,
            has_lp_flag,
            volatility,
            price_change_pct,
            in_range,
            capital_ratio,
        ], dtype=np.float32)

    def _width_from_action(self, a_width: float) -> float:
        frac = (a_width + 1.0) / 2.0
        return self.min_width_pct + frac * (self.max_width_pct - self.min_width_pct)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            options = {}
        
        if options.get("shuffle_windows", True):
            self.windows = self.windows.copy()
            self.np_random.shuffle(self.windows)

        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self.idx >= self.n_windows:
            return self._get_obs(), 0.0, True, False, {}

        t0, t1 = self.windows[self.idx]
        price_t0 = self._get_price(t0)

        mode = float(action[0])
        a_width = float(action[1])
        reward = 0.0

        burn_threshold = -1.0 / 3.0
        adjust_threshold = 1.0 / 3.0

        if self.sim is not None:
            # --- Full simulator: exact fees and IL (same logic as uniswap_v3_ppo_continuous) ---
            if mode < burn_threshold:
                if self.has_lp and self.position_entry_time is not None:
                    try:
                        crystallize_summary = self.sim.simulate(
                            price_lower=self.current_lower,
                            price_upper=self.current_upper,
                            start=self.position_entry_time.isoformat(),
                            end=t0.isoformat(),
                            liquidity=None,
                            total_usd=self.current_capital,
                            validate=False,
                            use_swap_liquidity=False,
                            accounting_mode="growth",
                            protocol_fee_encoding="base256",
                        )
                        crystallized_il = crystallize_summary.impermanent_loss.usd
                    except Exception:
                        crystallized_il = 0.0
                    capital_change = self.accumulated_fees + crystallized_il - self.gas_per_action_usd
                    self.current_capital += capital_change
                    self.current_capital = max(0.0, self.current_capital)
                    reward = float(capital_change)
                elif self.has_lp:
                    capital_change = self.accumulated_fees - self.gas_per_action_usd
                    self.current_capital += capital_change
                    self.current_capital = max(0.0, self.current_capital)
                    reward = float(capital_change)
                self.has_lp = False
                self.current_lower = None
                self.current_upper = None
                self.current_width_pct = 0.0
                self.position_entry_time = None
                self.position_entry_price = None
                self.accumulated_fees = 0.0

            elif mode <= adjust_threshold:
                if self.has_lp and self.current_lower is not None and self.current_upper is not None:
                    summary = self.sim.simulate(
                        price_lower=self.current_lower,
                        price_upper=self.current_upper,
                        start=t0.isoformat(),
                        end=t1.isoformat(),
                        liquidity=None,
                        total_usd=self.current_capital,
                        validate=False,
                        use_swap_liquidity=False,
                        accounting_mode="growth",
                        protocol_fee_encoding="base256",
                    )
                    fees_usd = summary.fees_usd.token1
                    self.accumulated_fees += float(fees_usd)
                    reward = float(fees_usd)

            else:
                width_pct = self._width_from_action(a_width)
                lower = price_t0 * (1.0 - width_pct)
                upper = price_t0 * (1.0 + width_pct)
                if self.has_lp and self.position_entry_time is not None:
                    try:
                        crystallize_summary = self.sim.simulate(
                            price_lower=self.current_lower,
                            price_upper=self.current_upper,
                            start=self.position_entry_time.isoformat(),
                            end=t0.isoformat(),
                            liquidity=None,
                            total_usd=self.current_capital,
                            validate=False,
                            use_swap_liquidity=False,
                            accounting_mode="growth",
                            protocol_fee_encoding="base256",
                        )
                        crystallized_il = crystallize_summary.impermanent_loss.usd
                    except Exception:
                        crystallized_il = 0.0
                    capital_change = self.accumulated_fees + crystallized_il - self.gas_per_action_usd
                    self.current_capital += capital_change
                    self.current_capital = max(0.0, self.current_capital)
                    reward += float(capital_change)
                elif self.has_lp:
                    capital_change = self.accumulated_fees - self.gas_per_action_usd
                    self.current_capital += capital_change
                    self.current_capital = max(0.0, self.current_capital)
                    reward += float(capital_change)
                else:
                    self.current_capital -= self.gas_per_action_usd
                    self.current_capital = max(0.0, self.current_capital)
                    reward -= self.gas_per_action_usd
                summary = self.sim.simulate(
                    price_lower=lower,
                    price_upper=upper,
                    start=t0.isoformat(),
                    end=t1.isoformat(),
                    liquidity=None,
                    total_usd=self.current_capital,
                    validate=False,
                    use_swap_liquidity=False,
                    accounting_mode="growth",
                    protocol_fee_encoding="base256",
                )
                fees_usd = summary.fees_usd.token1
                self.has_lp = True
                self.current_lower = lower
                self.current_upper = upper
                self.current_width_pct = min(width_pct / self.max_width_pct, 1.0)
                self.position_entry_time = t0
                self.position_entry_price = price_t0
                self.accumulated_fees = float(fees_usd)

        else:
            # --- Fast approximations (original optimized logic) ---
            if mode < burn_threshold:
                if self.has_lp and self.position_entry_price is not None:
                    il = self._estimate_il(
                        self.position_entry_price, price_t0,
                        self.current_width_pct * self.max_width_pct, self.current_capital
                    )
                    capital_change = self.accumulated_fees + il - self.gas_per_action_usd
                    self.current_capital += capital_change
                    self.current_capital = max(0.0, self.current_capital)
                    reward = float(capital_change)
                self.has_lp = False
                self.current_lower = None
                self.current_upper = None
                self.current_width_pct = 0.0
                self.position_entry_price = None
                self.position_entry_time = None
                self.accumulated_fees = 0.0

            elif mode <= adjust_threshold:
                if self.has_lp and self.current_lower is not None:
                    width_pct = self.current_width_pct * self.max_width_pct
                    fees = self._estimate_fees(t0, width_pct, self.current_capital)
                    self.accumulated_fees += fees
                    reward = float(fees)

            else:
                width_pct = self._width_from_action(a_width)
                lower = price_t0 * (1.0 - width_pct)
                upper = price_t0 * (1.0 + width_pct)
                if self.has_lp and self.position_entry_price is not None:
                    il = self._estimate_il(
                        self.position_entry_price, price_t0,
                        self.current_width_pct * self.max_width_pct, self.current_capital
                    )
                    capital_change = self.accumulated_fees + il - self.gas_per_action_usd
                    self.current_capital += capital_change
                    self.current_capital = max(0.0, self.current_capital)
                    reward += float(capital_change)
                else:
                    self.current_capital -= self.gas_per_action_usd
                    self.current_capital = max(0.0, self.current_capital)
                    reward -= self.gas_per_action_usd
                fees = self._estimate_fees(t0, width_pct, self.current_capital)
                self.has_lp = True
                self.current_lower = lower
                self.current_upper = upper
                self.current_width_pct = min(width_pct / self.max_width_pct, 1.0)
                self.position_entry_price = price_t0
                self.position_entry_time = None
                self.accumulated_fees = fees

        self.idx += 1
        self.prev_price = price_t0
        terminated = self.idx >= self.n_windows
        
        return self._get_obs(), reward, terminated, False, {
            "t0": t0,
            "current_capital": self.current_capital,
        }


def make_fast_env_fn(
    precomputed: PrecomputedData,
    total_usd: float = 1000.0,
    gas_per_action_usd: float = 0.1,
    min_width_pct: float = 0.001,
    max_width_pct: float = 0.05,  # 5% so agent can choose "wide range" = low IL
    mode: str = "train",
    data_dir: Optional[str] = None,
    use_full_sim: bool = False,
):
    def _init():
        return UniswapV3FastEnv(
            precomputed=precomputed,
            total_usd=total_usd,
            gas_per_action_usd=gas_per_action_usd,
            min_width_pct=min_width_pct,
            max_width_pct=max_width_pct,
            mode=mode,
            data_dir=data_dir,
            use_full_sim=use_full_sim,
        )
    return _init


def train_optimized(
    data_dir: str,
    num_envs: int = 8,
    total_timesteps: int = 100_000,
    save_path: str = "ppo_uniswap_v3_optimized",
    eval_freq: int = 10_000,
    use_full_sim: bool = False,
):
    """
    Train PPO with optimized settings.
    
    Args:
        data_dir: Path to training data
        num_envs: Number of parallel environments (default 8)
        total_timesteps: Total training steps
        save_path: Model save path
        eval_freq: Evaluation frequency
        use_full_sim: If True, use exact fees/IL from fee_simulator (slower, correct)
    """
    print("=" * 60)
    print("🚀 Optimized Uniswap v3 PPO Training")
    print("=" * 60)
    print(f"  Data dir: {data_dir}")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Full simulator (exact fees/IL): {use_full_sim}")
    print()
    
    # Precompute data ONCE (prices, volatilities; fee_rates only used when not use_full_sim)
    precomputed = precompute_training_data(data_dir, window_hours=1)
    
    print()
    print("🏋️ Creating training environments...")
    
    train_fn = make_fast_env_fn(
        precomputed, mode="train",
        data_dir=data_dir, use_full_sim=use_full_sim,
    )
    eval_fn = make_fast_env_fn(
        precomputed, mode="eval",
        data_dir=data_dir, use_full_sim=use_full_sim,
    )
    
    if num_envs > 1:
        env = SubprocVecEnv([train_fn for _ in range(num_envs)])
    else:
        env = DummyVecEnv([train_fn])
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    eval_env = DummyVecEnv([eval_fn])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    
    print(f"  ✅ {num_envs} training envs, 1 eval env created")
    print()
    
    # Optimized PPO hyperparameters
    # n_steps * num_envs must be divisible by batch_size to avoid truncated mini-batches (SB3 warning)
    batch_size = 256
    target_total = (2048 // num_envs) * num_envs
    # Rollout size must be multiple of LCM(batch_size, num_envs) so n_steps is int and batch_size divides
    lcm = batch_size * num_envs // math.gcd(batch_size, num_envs)
    total_per_update = max(lcm, (target_total // lcm) * lcm)
    n_steps = total_per_update // num_envs
    assert (n_steps * num_envs) % batch_size == 0, "n_steps * num_envs must be divisible by batch_size"
    
    print("🧠 Creating PPO model...")
    print(f"  n_steps per env: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  samples per update: {n_steps * num_envs}")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,  # Encourage exploration
        clip_range=0.2,
        n_epochs=10,
        gae_lambda=0.95,
        tensorboard_log=None,  # Set to "./ppo_uniswap_tb/" if tensorboard is installed
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1000),
        save_path="./checkpoints/",
        name_prefix="ppo_uniswap"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=max(eval_freq // num_envs, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )
    
    print()
    print("🏃 Starting training...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,  # Set True if tqdm/rich installed: pip install stable-baselines3[extra]
    )
    
    print()
    print("💾 Saving model...")
    model.save(save_path)
    env.save(f"{save_path}_vec_normalize.pkl")
    
    print()
    print("=" * 60)
    print("✅ Training complete!")
    print(f"  Model saved to: {save_path}.zip")
    print(f"  VecNormalize saved to: {save_path}_vec_normalize.pkl")
    print("=" * 60)
    
    return model, env


def _action_to_label(mode: float, a_width: float, min_width_pct: float = 0.001, max_width_pct: float = 0.05) -> Tuple[str, Optional[float]]:
    """Map raw action (mode, a_width) to human-readable label and width_pct if MINT. Same thresholds as env."""
    burn_threshold = -1.0 / 3.0
    adjust_threshold = 1.0 / 3.0
    if mode < burn_threshold:
        return "BURN", None  # close LP position
    if mode <= adjust_threshold:
        return "HOLD", None  # keep current position (or do nothing if no LP)
    frac = (a_width + 1.0) / 2.0
    width_pct = min_width_pct + frac * (max_width_pct - min_width_pct)
    return "MINT", width_pct  # open new position with this width


def evaluate_on_test(
    data_dir: str,
    model_path: str = "ppo_uniswap_v3_optimized.zip",
    vec_normalize_path: Optional[str] = None,
    n_episodes: int = 10,
    use_full_sim: bool = False,
    deterministic: bool = True,
    max_steps_per_episode: Optional[int] = 500,
    show_actions: bool = True,
) -> dict:
    """
    Evaluate trained model on the held-out 10% test set (80/10/10 split).
    If max_steps_per_episode is set, each episode is truncated after that many steps (faster eval).
    If show_actions is True, print action summary (BURN / HOLD / MINT counts) and first-episode step log.
    """
    if vec_normalize_path is None:
        vec_normalize_path = model_path.replace(".zip", "_vec_normalize.pkl")
    
    print("=" * 60)
    print("📊 Evaluation on TEST set (last 10% of data)")
    print("=" * 60)
    print(f"  Data dir: {data_dir}")
    print(f"  Model: {model_path}")
    print(f"  VecNormalize: {vec_normalize_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Full sim: {use_full_sim}")
    print(f"  Max steps/episode: {max_steps_per_episode}")
    print()
    
    precomputed = precompute_training_data(data_dir, window_hours=1)
    n_total = len(precomputed.windows)
    test_start = int(n_total * 0.9)
    n_test = n_total - test_start
    print(f"  Test windows: {n_test} (indices {test_start}-{n_total})")
    print()
    
    eval_fn = make_fast_env_fn(
        precomputed,
        total_usd=1000.0,
        gas_per_action_usd=0.1,
        mode="test",
        data_dir=data_dir if use_full_sim else None,
        use_full_sim=use_full_sim,
    )
    env = DummyVecEnv([eval_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(model_path, env=env)
    
    rewards = []
    capitals = []
    all_actions: List[Tuple[str, Optional[float]]] = []
    first_episode_log: List[Tuple[int, str, Optional[float], float]] = []  # step, label, width_pct, capital
    for ep in range(n_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, (tuple, list)) else reset_result
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                break
            action, _ = model.predict(obs, deterministic=deterministic)
            mode, a_width = float(action[0][0]), float(action[0][1])
            label, width_pct = _action_to_label(mode, a_width)
            all_actions.append((label, width_pct))
            capital = 1000.0
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated[0] or truncated[0])
            else:
                obs, reward, done, info = step_result
                done = bool(done[0])
            capital = info[0].get("current_capital", 1000.0)
            if ep == 0 and len(first_episode_log) < 25:
                first_episode_log.append((steps, label, width_pct, float(capital)))
            total_reward += float(reward[0])
            steps += 1
        rewards.append(total_reward)
        capitals.append(info[0].get("current_capital", 1000.0))
    
    if show_actions and all_actions:
        burn_count = sum(1 for lab, _ in all_actions if lab == "BURN")
        hold_count = sum(1 for lab, _ in all_actions if lab == "HOLD")
        mint_count = sum(1 for lab, _ in all_actions if lab == "MINT")
        mint_widths = [w for _, w in all_actions if w is not None]
        print("Action summary (what the agent did):")
        print(f"  BURN (close LP):  {burn_count:5d}  ({100*burn_count/len(all_actions):.1f}%)")
        print(f"  HOLD (keep/do nothing): {hold_count:5d}  ({100*hold_count/len(all_actions):.1f}%)")
        print(f"  MINT (open LP):   {mint_count:5d}  ({100*mint_count/len(all_actions):.1f}%)")
        if mint_widths:
            print(f"  When MINT, width %% around price: mean {100*np.mean(mint_widths):.3f}%  min {100*min(mint_widths):.3f}%  max {100*max(mint_widths):.3f}%")
        if mint_count == 0 and all_actions:
            print()
            print("  (Agent never opened a position. Likely causes: model was trained when many windows had 0 fee rate"
                  " or IL risk dominated; retrain with wider max range and min fee rate for no-swap windows to encourage wide-range LP.)")
        print()
        print("First episode (first 25 steps):")
        print("  step   action   width_pct   capital")
        for step, label, width_pct, cap in first_episode_log:
            w_str = f"{100*width_pct:.3f}%" if width_pct is not None else "  -"
            print(f"  {step:4d}   {label:4s}     {w_str:>8s}   ${cap:.2f}")
        print()
    
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_capital = float(np.mean(capitals))
    std_capital = float(np.std(capitals))
    return_pct = (mean_capital - 1000.0) / 1000.0 * 100.0  # % on $1000
    
    print("Results:")
    print(f"  Mean reward:    {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mean final cap: ${mean_capital:.2f} ± ${std_capital:.2f}")
    print(f"  Return:         {return_pct:.2f}%")
    print("=" * 60)
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_capital": mean_capital,
        "std_capital": std_capital,
        "return_pct": return_pct,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized PPO training for Uniswap v3 LP")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data (default: simulation_6/training_data)")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default="ppo_uniswap_v3_optimized", help="Model save path")
    parser.add_argument("--full-sim", action="store_true", help="Use exact fees/IL from fee_simulator (correct, slower)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on 10%% test set (80/10/10 split)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes for test evaluation")
    parser.add_argument("--eval-max-steps", type=int, default=500, help="Max steps per episode (None = full test set, use 0 for no limit in script)")
    parser.add_argument("--no-show-actions", action="store_true", help="Disable action summary and step log during evaluation")
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        # Try several locations so it works when code/data are not in a subfolder (e.g. all in home dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        candidates = [
            os.path.join(script_dir, "training_data"),
            os.path.join(cwd, "training_data"),
            cwd,  # data CSVs directly in current directory
        ]
        required = "pool_config_eth_usdt_0p3.csv"
        for d in candidates:
            if d and os.path.isfile(os.path.join(d, required)):
                args.data_dir = os.path.abspath(d)
                break
        else:
            args.data_dir = os.path.join(script_dir, "training_data")  # default for error message
            if not os.path.isfile(os.path.join(args.data_dir, required)):
                raise FileNotFoundError(
                    f"Data not found. Tried: {candidates}. "
                    f"Put training_data (with pool_config_*.csv, swaps_*.csv, etc.) in the same directory as this script, "
                    f"or in current directory, or pass --data-dir /path/to/training_data"
                )
    
    if args.evaluate:
        evaluate_on_test(
            data_dir=args.data_dir,
            model_path=f"{args.save_path}.zip" if not args.save_path.endswith(".zip") else args.save_path,
            n_episodes=args.eval_episodes,
            use_full_sim=args.full_sim,
            max_steps_per_episode=args.eval_max_steps if args.eval_max_steps > 0 else None,
            show_actions=not args.no_show_actions,
        )
    else:
        train_optimized(
            data_dir=args.data_dir,
            num_envs=args.num_envs,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            use_full_sim=args.full_sim,
        )
