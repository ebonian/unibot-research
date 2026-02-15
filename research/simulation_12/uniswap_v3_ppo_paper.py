# uniswap_v3_ppo_paper.py
"""
Uniswap v3 PPO Training - Paper-Based Approach (Exact Per-Swap)

Implements the methodology from:
  Xu & Brini (2025) - "Improving DeFi Accessibility through Efficient Liquidity
                       Provisioning with Deep Reinforcement Learning" (AAAI 2025)
  arXiv:2501.07508

Key features:
1. EXACT per-swap fee calculation (summing Equations 5-6 over all swaps per hour)
2. LVR (Loss-Versus-Rebalancing) penalty in reward (Equations 15-17)
3. Hourly decision steps with swap-level fee accuracy
4. State space: 36-dim (31 tech features + 5 position features) - matches DQN
5. Discrete action space: tick widths for LP interval
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

Q96 = 2 ** 96


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Convert Uniswap v3 sqrtPriceX96 to human-readable price."""
    p = float(sqrt_price_x96) / Q96
    return (p * p) * (10 ** (decimals0 - decimals1))


def price_to_tick(price: float) -> int:
    """
    Convert price to tick index (Equation 1 from paper).
    i = floor(log(p_t) / log(1.0001))
    """
    if price <= 0:
        return 0
    return int(math.floor(math.log(price) / math.log(1.0001)))


def tick_to_price(tick: int) -> float:
    """Convert tick index to price: p(i) = 1.0001^i"""
    return math.pow(1.0001, tick)


@dataclass
class HourlyData:
    """Hourly resampled data for training (paper approach)."""
    timestamps: List[pd.Timestamp]
    prices: Dict[pd.Timestamp, float]  # hourly close price
    volumes: Dict[pd.Timestamp, float]  # hourly swap volume in USD
    volatilities: Dict[pd.Timestamp, float]  # exponentially weighted volatility
    ma_24h: Dict[pd.Timestamp, float]  # 24-hour moving average
    ma_168h: Dict[pd.Timestamp, float]  # 168-hour (1 week) moving average
    decimals0: int
    decimals1: int
    pool_fee: float  # δ in paper (e.g. 0.003 for 0.3%)
    tick_spacing: int
    # Per-swap prices for each hour, precomputed from swap-level data.
    # Used for exact per-swap fee calculation following Zhang et al. (2023).
    swap_prices_per_hour: Optional[Dict[pd.Timestamp, np.ndarray]] = None


def prepare_hourly_data(data_dir: str) -> HourlyData:
    """
    Prepare hourly resampled data (paper methodology).
    Resamples swap data to hourly OHLCV format.
    """
    print("🔄 Preparing hourly data (paper methodology)...")
    
    # Load data files
    pool_cfg = pd.read_csv(os.path.join(data_dir, "pool_config_eth_usdt_0p3.csv"))
    tokens = pd.read_csv(os.path.join(data_dir, "token_metadata_eth_usdt_0p3.csv"))
    swaps_files = glob.glob(os.path.join(data_dir, "swaps_*_eth_usdt_0p3.csv"))
    
    if not swaps_files:
        raise FileNotFoundError(f"Missing swaps_*_eth_usdt_0p3.csv in {data_dir}")
    
    swaps = pd.read_csv(swaps_files[0], low_memory=False)
    
    # Get token decimals
    tokens['contract_address'] = tokens['contract_address'].str.lower()
    t0_addr = pool_cfg.loc[0, 'token0'].lower()
    t1_addr = pool_cfg.loc[0, 'token1'].lower()
    t0 = tokens.set_index('contract_address').loc[t0_addr]
    t1 = tokens.set_index('contract_address').loc[t1_addr]
    decimals0 = int(t0['decimals'])
    decimals1 = int(t1['decimals'])
    
    # Pool parameters
    pool_fee_bps = int(pool_cfg.loc[0, 'fee'])
    pool_fee = pool_fee_bps / 1_000_000  # e.g. 3000 -> 0.003 (0.3%)
    tick_spacing = int(pool_cfg.loc[0, 'tickSpacing'])
    
    print(f"  Pool fee: {pool_fee*100:.2f}% ({pool_fee_bps} bps)")
    print(f"  Tick spacing: {tick_spacing}")
    
    # Parse timestamps and compute prices
    swaps['evt_block_time'] = pd.to_datetime(swaps['evt_block_time'], utc=True)
    swaps = swaps.sort_values('evt_block_time').reset_index(drop=True)
    swaps['price'] = swaps['sqrtPriceX96'].apply(
        lambda x: sqrt_price_x96_to_price(int(x), decimals0, decimals1)
    )
    swaps['volume_usd'] = swaps['amount1'].abs() / (10 ** decimals1)
    
    # Precompute per-swap prices for each hour from swap-level data.
    # Following Zhang et al. (2023): fees are summed over every swap per hour.
    swaps_indexed_for_prices = swaps.set_index('evt_block_time')
    swap_prices_per_hour_raw = {}
    for hour, group in swaps_indexed_for_prices.groupby(pd.Grouper(freq='1h')):
        if len(group) >= 2:
            swap_prices_per_hour_raw[hour] = group['price'].values.astype(np.float64)
        elif len(group) == 1:
            swap_prices_per_hour_raw[hour] = group['price'].values.astype(np.float64)
    
    # Resample to hourly OHLCV
    swaps.set_index('evt_block_time', inplace=True)
    hourly = swaps.resample('1h').agg({
        'price': ['first', 'last', 'max', 'min'],
        'volume_usd': 'sum'
    })
    hourly.columns = ['open', 'close', 'high', 'low', 'volume']
    hourly = hourly.dropna(subset=['close'])
    
    # Forward-fill missing hours (no swaps in that hour)
    full_range = pd.date_range(start=hourly.index.min(), end=hourly.index.max(), freq='1h', tz='UTC')
    hourly = hourly.reindex(full_range)
    hourly['close'] = hourly['close'].ffill()
    hourly['open'] = hourly['open'].ffill()
    hourly['high'] = hourly['high'].ffill()
    hourly['low'] = hourly['low'].ffill()
    hourly['volume'] = hourly['volume'].fillna(0)
    
    print(f"  📊 {len(hourly)} hourly candles")
    
    # Compute volatility (exponentially weighted std of log returns)
    # Paper uses smoothing factor α = 0.05
    hourly['log_return'] = np.log(hourly['close']).diff()
    hourly['volatility'] = hourly['log_return'].ewm(alpha=0.05, min_periods=1).std()
    hourly['volatility'] = hourly['volatility'].fillna(0)
    
    # Moving averages
    hourly['ma_24h'] = hourly['close'].rolling(window=24, min_periods=1).mean()
    hourly['ma_168h'] = hourly['close'].rolling(window=168, min_periods=1).mean()
    
    print(f"  📈 Computed volatility and moving averages")
    
    # Convert to dictionaries for fast lookup
    timestamps = list(hourly.index)
    prices = hourly['close'].to_dict()
    volumes = hourly['volume'].to_dict()
    volatilities = hourly['volatility'].to_dict()
    ma_24h = hourly['ma_24h'].to_dict()
    ma_168h = hourly['ma_168h'].to_dict()
    
    # Map per-swap prices to the full (forward-filled) timestamp index
    swap_prices_per_hour = {}
    for ts in timestamps:
        swap_prices_per_hour[ts] = swap_prices_per_hour_raw.get(ts, None)
    
    print("✅ Data preparation complete!")
    
    return HourlyData(
        timestamps=timestamps,
        prices=prices,
        volumes=volumes,
        volatilities=volatilities,
        ma_24h=ma_24h,
        ma_168h=ma_168h,
        decimals0=decimals0,
        decimals1=decimals1,
        pool_fee=pool_fee,
        tick_spacing=tick_spacing,
        swap_prices_per_hour=swap_prices_per_hour,
    )


class UniswapV3PaperEnv(gym.Env):
    """
    Uniswap v3 LP environment following paper methodology.
    
    State space (paper Section 5):
      - Market price p_t (log scale)
      - Tick index i (normalized)
      - Interval width w_t (as multiple of tick_spacing)
      - Current liquidity L_t (normalized by initial)
      - Volatility σ_t (exponentially weighted)
      - 24h moving average (normalized by price)
      - 168h moving average (normalized by price)
      - In-range flag (1 if price in LP range)
    
    Action space (discrete tick widths):
      - 0: do nothing / hold
      - 1-N: deploy/adjust LP with width = action * tick_spacing * 10
    
    Reward (Equation 17 from paper):
      R_{t+1} = f_t - ℓ_t(σ, p) - gas_cost
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        hourly_data,  # HourlyData or HourlyDataExtended
        initial_capital_usd: float = 1000.0,  # $1,000 initial capital
        gas_cost_usd: float = 0.05,  # Arbitrum L2 gas cost (~$0.05 per rebalance)
        action_ticks: List[int] = [0, 1, 3, 5, 10, 20, 40],  # 0=hold, rest=widths in tick_spacing units
        mode: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        
        self.hourly_data = hourly_data
        self.initial_capital = float(initial_capital_usd)
        self.gas_cost_usd = float(gas_cost_usd)
        self.action_ticks = action_ticks  # 0 = no action, others = tick widths
        self.pool_fee = hourly_data.pool_fee
        self.tick_spacing = hourly_data.tick_spacing
        
        # Split data: 80/10/10 train/val/test
        n_total = len(hourly_data.timestamps)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        if mode == "train":
            self.timestamps = hourly_data.timestamps[:train_end]
        elif mode == "eval":
            self.timestamps = hourly_data.timestamps[train_end:val_end]
        elif mode == "test":
            self.timestamps = hourly_data.timestamps[val_end:]
        else:
            self.timestamps = hourly_data.timestamps
        
        self.n_steps = len(self.timestamps) - 1  # Need t and t+1
        
        # Check if we have extended features (from HourlyDataExtended)
        self.has_extended_features = hasattr(hourly_data, 'features') and hourly_data.features is not None
        
        if self.has_extended_features:
            # 36-dim observation: 31 tech features + 7 position features (matches DQN)
            # Position features:
            # 1. cash_ratio
            # 2. width_normalized
            # 3. in_range
            # 4. position_value_ratio
            # 5. price_momentum
            # 6. dist_to_boundary (NEW)
            # 7. hours_since_rebalance (NEW)
            n_tech_features = 31
            self.state_dim = n_tech_features + 7
            self.max_width = max(self.action_ticks[1:]) if len(self.action_ticks) > 1 else 1
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_dim,), dtype=np.float32
            )
        else:
            # Legacy 8-dim observation (fallback)
            self.max_width = max(self.action_ticks[1:]) if len(self.action_ticks) > 1 else 1
            self.observation_space = spaces.Box(
                low=np.array([-np.inf, -1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([np.inf, 1e6, 100.0, 10.0, 1.0, 2.0, 2.0, 1.0], dtype=np.float32),
            )
        
        # Discrete action space
        self.action_space = spaces.Discrete(len(action_ticks))
        
        # Episode state
        self._reset_state()

    def _reset_state(self):
        self.idx = 0
        self.has_lp = False
        self.lp_lower_tick = None
        self.lp_upper_tick = None
        self.lp_width_ticks = 0
        self.liquidity = 0.0  # L_t in paper
        self.entry_price = None
        self.initial_value_usd = None  # For HODL comparison
        self.position_entry_idx = 0  # Track position age

    def _get_price(self, t: pd.Timestamp) -> float:
        return self.hourly_data.prices.get(t, 0.0)

    def _get_volatility(self, t: pd.Timestamp) -> float:
        return self.hourly_data.volatilities.get(t, 0.0)

    def _get_ma_24h(self, t: pd.Timestamp) -> float:
        return self.hourly_data.ma_24h.get(t, 0.0)

    def _get_ma_168h(self, t: pd.Timestamp) -> float:
        return self.hourly_data.ma_168h.get(t, 0.0)

    def _get_volume(self, t: pd.Timestamp) -> float:
        return self.hourly_data.volumes.get(t, 0.0)

    def _compute_liquidity(self, price: float, price_lower: float, price_upper: float, x: float, y: float) -> float:
        """
        Compute liquidity L from token amounts and price range.
        From Equation 3 in paper:
          x_t = L_t * (1/√p_t - 1/√p_t^u)
          y_t = L_t * (√p_t - √p_t^l)
        """
        if price <= 0 or price_lower <= 0 or price_upper <= 0:
            return 0.0
        
        sqrt_p = math.sqrt(price)
        sqrt_pl = math.sqrt(price_lower)
        sqrt_pu = math.sqrt(price_upper)
        
        if price <= price_lower:
            # All in token X
            if sqrt_pl > 0 and sqrt_pu > 0 and sqrt_pl != sqrt_pu:
                return x / (1.0 / sqrt_pl - 1.0 / sqrt_pu)
        elif price >= price_upper:
            # All in token Y
            return y / (sqrt_pu - sqrt_pl) if sqrt_pu > sqrt_pl else 0.0
        else:
            # In range: use both
            L_from_x = x / (1.0 / sqrt_p - 1.0 / sqrt_pu) if sqrt_p != sqrt_pu else 0.0
            L_from_y = y / (sqrt_p - sqrt_pl) if sqrt_p > sqrt_pl else 0.0
            return min(L_from_x, L_from_y) if L_from_x > 0 and L_from_y > 0 else max(L_from_x, L_from_y)
        return 0.0

    def _compute_position_value(self, price: float, price_lower: float, price_upper: float, L: float) -> float:
        """
        Compute LP position value V_t(p_t) from Equation 15 (first part).
        V_t(p_t) = L_t * (2√p_t - p_t/√p_t^u - √p_t^l)  [when p ∈ [p_l, p_u]]
        """
        if L <= 0 or price <= 0:
            return 0.0
        
        sqrt_p = math.sqrt(price)
        sqrt_pl = math.sqrt(price_lower)
        sqrt_pu = math.sqrt(price_upper)
        
        if price <= price_lower:
            # All in X: value = x * price
            x = L * (1.0 / sqrt_pl - 1.0 / sqrt_pu)
            return x * price
        elif price >= price_upper:
            # All in Y: value = y
            y = L * (sqrt_pu - sqrt_pl)
            return y
        else:
            # In range: Equation 15
            return L * (2.0 * sqrt_p - price / sqrt_pu - sqrt_pl)

    def _compute_fee(self, price_t: float, price_t1: float, L: float, price_lower: float, price_upper: float) -> float:
        """
        Compute trading fee by summing Equations 5-6 over every swap in this hour.
        Exact per-swap computation following Zhang et al. (2023) Section 3.2.
        """
        if L <= 0:
            return 0.0
        
        delta = self.pool_fee
        fee_mult = delta / (1.0 - delta)
        
        # Get precomputed per-swap prices for this hour
        t0 = self.timestamps[self.idx]
        swap_prices = self.hourly_data.swap_prices_per_hour.get(t0, None) if self.hourly_data.swap_prices_per_hour else None
        
        if swap_prices is not None and len(swap_prices) >= 2:
            # Exact per-swap fee computation (paper method)
            total_fee = 0.0
            sqrt_pl = math.sqrt(price_lower)
            sqrt_pu = math.sqrt(price_upper)
            
            for i in range(len(swap_prices) - 1):
                p0 = swap_prices[i]
                p1 = swap_prices[i + 1]
                
                # Skip if both outside range on same side
                if (p0 < price_lower and p1 < price_lower) or (p0 > price_upper and p1 > price_upper):
                    continue
                
                # Clamp prices to LP range
                p0_c = max(price_lower, min(price_upper, p0))
                p1_c = max(price_lower, min(price_upper, p1))
                sqrt_p0_c = math.sqrt(p0_c)
                sqrt_p1_c = math.sqrt(p1_c)
                
                if sqrt_p0_c <= sqrt_p1_c:
                    fee_i = fee_mult * L * (sqrt_p1_c - sqrt_p0_c)
                else:
                    fee_i = fee_mult * L * (1.0 / sqrt_p1_c - 1.0 / sqrt_p0_c) * p1_c
                
                total_fee += max(0.0, fee_i)
            
            return total_fee
        else:
            # Fallback to open-close formula if no swap data available
            if (price_t < price_lower and price_t1 < price_lower) or \
               (price_t > price_upper and price_t1 > price_upper):
                return 0.0
            
            p_t_clamped = max(price_lower, min(price_upper, price_t))
            p_t1_clamped = max(price_lower, min(price_upper, price_t1))
            
            if p_t_clamped <= p_t1_clamped:
                fee = fee_mult * L * (math.sqrt(p_t1_clamped) - math.sqrt(p_t_clamped))
            else:
                fee = fee_mult * L * (1.0 / math.sqrt(p_t1_clamped) - 1.0 / math.sqrt(p_t_clamped)) * p_t1_clamped
            
            return max(0.0, fee)

    def _position_value_at(self, price: float, L: float, p_lower: float, p_upper: float,
                           sqrt_pl: float, sqrt_pu: float) -> float:
        """Compute position value at a given price (helper for per-swap LVR)."""
        if L <= 0:
            return 0.0
        sqrt_p = math.sqrt(price)
        if price <= p_lower:
            x = L * (1.0 / sqrt_pl - 1.0 / sqrt_pu)
            return x * price
        elif price >= p_upper:
            y = L * (sqrt_pu - sqrt_pl)
            return y
        else:
            return L * (2.0 * sqrt_p - price / sqrt_pu - sqrt_pl)

    def _compute_lvr(self, price_t0: float, price_t1: float, L: float, 
                     p_lower: float, p_upper: float) -> float:
        """
        Compute LVR (Loss-Versus-Rebalancing) using Zhang et al. (2023) discrete formula.
        Sums over every swap: LVR = Σ {V(p_{i+1}) - V(p_i) - x(p_i) × Δp}
        This is always <= 0 (it's a cost).
        """
        if L <= 0:
            return 0.0
        
        sqrt_pl = math.sqrt(p_lower)
        sqrt_pu = math.sqrt(p_upper)
        
        # Get precomputed per-swap prices for this hour
        t0 = self.timestamps[self.idx]
        swap_prices = self.hourly_data.swap_prices_per_hour.get(t0, None) if self.hourly_data.swap_prices_per_hour else None
        
        if swap_prices is not None and len(swap_prices) >= 2:
            # Exact per-swap LVR computation (Zhang et al. Equation 5)
            total_lvr = 0.0
            
            for i in range(len(swap_prices) - 1):
                pi = swap_prices[i]
                pi1 = swap_prices[i + 1]
                
                # V(p) = position value at price p
                V_i = self._position_value_at(pi, L, p_lower, p_upper, sqrt_pl, sqrt_pu)
                V_i1 = self._position_value_at(pi1, L, p_lower, p_upper, sqrt_pl, sqrt_pu)
                
                # x(p_i) = amount of token X at price p_i
                if pi <= p_lower:
                    x_i = L * (1.0 / sqrt_pl - 1.0 / sqrt_pu)
                elif pi >= p_upper:
                    x_i = 0.0
                else:
                    x_i = L * (1.0 / math.sqrt(pi) - 1.0 / sqrt_pu)
                
                # LVR_i = ΔV - x(p_i) × Δp
                lvr_i = (V_i1 - V_i) - x_i * (pi1 - pi)
                total_lvr += lvr_i
            
            return total_lvr
        else:
            # Fallback to endpoint-based LVR
            V_t0 = self._position_value_at(price_t0, L, p_lower, p_upper, sqrt_pl, sqrt_pu)
            V_t1 = self._position_value_at(price_t1, L, p_lower, p_upper, sqrt_pl, sqrt_pu)
            
            if price_t0 <= p_lower:
                x_t0 = L * (1.0 / sqrt_pl - 1.0 / sqrt_pu)
            elif price_t0 >= p_upper:
                x_t0 = 0.0
            else:
                x_t0 = L * (1.0 / math.sqrt(price_t0) - 1.0 / sqrt_pu)
            
            return (V_t1 - V_t0) - x_t0 * (price_t1 - price_t0)

    def _get_obs(self) -> np.ndarray:
        if self.has_extended_features:
            return self._get_obs_extended()
        return self._get_obs_legacy()
    
    def _get_obs_extended(self) -> np.ndarray:
        """36-dim observation matching DQN's state space."""
        if self.idx >= len(self.timestamps):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        t = self.timestamps[self.idx]
        price = self._get_price(t)
        
        # 31 technical features (same as DQN)
        tech_features = self.hourly_data.features.get(t, np.zeros(31, dtype=np.float32))
        
        # 5 position features (matching DQN)
        # cash_ratio: PPO always re-deploys full capital, so this is 0 when has_lp, 1 otherwise
        cash_ratio = 0.0 if self.has_lp else 1.0
        width_normalized = self.lp_width_ticks / max(self.max_width, 1)
        
        # Position value ratio
        if self.has_lp and self.liquidity > 0 and self.lp_lower_tick is not None:
            lp_lower_price = tick_to_price(self.lp_lower_tick)
            lp_upper_price = tick_to_price(self.lp_upper_tick)
            pos_val = self._compute_position_value(price, lp_lower_price, lp_upper_price, self.liquidity)
            position_value_ratio = pos_val / max(self.initial_capital, 1e-10)
        else:
            position_value_ratio = 0.0
        
        # In-range check
        in_range = 0.0
        if self.has_lp and self.lp_lower_tick is not None:
            lp_lower_price = tick_to_price(self.lp_lower_tick)
            lp_upper_price = tick_to_price(self.lp_upper_tick)
            in_range = 1.0 if lp_lower_price <= price <= lp_upper_price else 0.0
        
        # Price momentum
        if self.idx >= 1:
            prev_t = self.timestamps[self.idx - 1]
            prev_price = self._get_price(prev_t)
            price_momentum = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        else:
            price_momentum = 0.0
            
        # Dist to boundary (NEW)
        dist_to_boundary = 0.0
        if self.has_lp and self.lp_lower_tick is not None and in_range > 0.5:
            tick = price_to_tick(price)
            # Distance in ticks
            dist = min(abs(tick - self.lp_lower_tick), abs(self.lp_upper_tick - tick))
            half_width_ticks = self.lp_width_ticks / 2.0
            dist_to_boundary = dist / half_width_ticks if half_width_ticks > 0 else 0.0
            
        # Hours since rebalance (NEW)
        hours_since_rebalance = 0.0
        if self.has_lp:
            hours_since_rebalance = min((self.idx - self.position_entry_idx) / 24.0, 1.0)
        
        position_features = np.array([
            cash_ratio, width_normalized, in_range, position_value_ratio,
            price_momentum, dist_to_boundary, hours_since_rebalance
        ], dtype=np.float32)
        
        return np.concatenate([tech_features, position_features])
    
    def _get_obs_legacy(self) -> np.ndarray:
        """Original 8-dim observation (fallback)."""
        if self.idx >= len(self.timestamps):
            return np.zeros(8, dtype=np.float32)
        
        t = self.timestamps[self.idx]
        price = self._get_price(t)
        tick = price_to_tick(price) if price > 0 else 0
        volatility = self._get_volatility(t)
        ma_24h = self._get_ma_24h(t)
        ma_168h = self._get_ma_168h(t)
        
        log_price = math.log(price) if price > 0 else 0.0
        tick_normalized = tick / 10000.0
        width_normalized = self.lp_width_ticks / 100.0
        liquidity_normalized = self.liquidity / 1e6 if self.liquidity > 0 else 0.0
        volatility_normalized = min(volatility * 100, 1.0)
        ma_24h_ratio = ma_24h / price if price > 0 and ma_24h > 0 else 1.0
        ma_168h_ratio = ma_168h / price if price > 0 and ma_168h > 0 else 1.0
        
        in_range = 0.0
        if self.has_lp and self.lp_lower_tick is not None and self.lp_upper_tick is not None:
            lp_lower_price = tick_to_price(self.lp_lower_tick)
            lp_upper_price = tick_to_price(self.lp_upper_tick)
            in_range = 1.0 if lp_lower_price <= price <= lp_upper_price else 0.0
        
        return np.array([
            log_price, tick_normalized, width_normalized,
            liquidity_normalized, volatility_normalized,
            ma_24h_ratio, ma_168h_ratio, in_range,
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        if self.idx >= self.n_steps:
            return self._get_obs(), 0.0, True, False, {}
        
        t0 = self.timestamps[self.idx]
        t1 = self.timestamps[self.idx + 1]
        
        price_t0 = self._get_price(t0)
        price_t1 = self._get_price(t1)
        sigma = self._get_volatility(t0)
        
        reward = 0.0
        tick_width = self.action_ticks[action]
        
        if tick_width == 0:
            # Action 0: HOLD - keep existing position, no gas cost
            # Agent must learn to use this when staying put is optimal
            if self.has_lp and self.liquidity > 0:
                lp_lower_price = tick_to_price(self.lp_lower_tick)
                lp_upper_price = tick_to_price(self.lp_upper_tick)
                fee = self._compute_fee(price_t0, price_t1, self.liquidity, lp_lower_price, lp_upper_price)
                
                # CAP FEE to total pool fees
                volume_usd = self._get_volume(t0)
                max_fee = volume_usd * self.pool_fee
                fee = min(fee, max_fee)

                lvr = self._compute_lvr(price_t0, price_t1, self.liquidity, lp_lower_price, lp_upper_price)
                reward = fee + lvr  # LVR is already negative, so we add it
        else:
            # Action > 0: Deploy/rebalance to specified width (always costs gas)
            # Agent learns when rebalancing is worth the gas cost
            # Even selecting same width = recenter at current price = costs gas
            
            tick_t0 = price_to_tick(price_t0)
            aligned_tick = (tick_t0 // self.tick_spacing) * self.tick_spacing
            half_width = tick_width * self.tick_spacing // 2
            
            new_lower_tick = aligned_tick - half_width
            new_upper_tick = aligned_tick + half_width
            new_lower_price = tick_to_price(new_lower_tick)
            new_upper_price = tick_to_price(new_upper_tick)
            
            # Compute liquidity from initial_capital USD (matching DQN approach)
            sqrt_p = math.sqrt(price_t0)
            sqrt_pl = math.sqrt(new_lower_price)
            sqrt_pu = math.sqrt(new_upper_price)
            
            if price_t0 <= new_lower_price:
                # Price below range: all in token X
                value_per_L = (1.0 / sqrt_pl - 1.0 / sqrt_pu) * price_t0
            elif price_t0 >= new_upper_price:
                # Price above range: all in token Y
                value_per_L = sqrt_pu - sqrt_pl
            else:
                # Price in range
                value_per_L = 2.0 * sqrt_p - price_t0 / sqrt_pu - sqrt_pl
            
            if value_per_L > 0:
                new_L = self.initial_capital / value_per_L
            else:
                new_L = 0.0
            
            self.initial_value_usd = self.initial_capital
            
            self.has_lp = True
            self.lp_width_ticks = tick_width
            self.lp_lower_tick = new_lower_tick
            self.lp_upper_tick = new_upper_tick
            self.entry_price = price_t0
            self.position_entry_idx = self.idx  # Track entry
            
            fee = self._compute_fee(price_t0, price_t1, new_L, new_lower_price, new_upper_price)
            
            # CAP FEE to total pool fees
            volume_usd = self._get_volume(t0)
            max_fee = volume_usd * self.pool_fee
            fee = min(fee, max_fee)

            lvr = self._compute_lvr(price_t0, price_t1, new_L, new_lower_price, new_upper_price)
            reward = fee + lvr  # LVR is already negative, so we add it
            
            # Gas cost for any rebalance - agent learns when it's worth it
            reward -= self.gas_cost_usd
            
            # Swap fee cost: ~50% of position needs swapping during rebalance
            # Only a portion of assets need to be swapped to achieve new token ratio
            swap_fee_rate = self.pool_fee  # Use pool's fee tier (0.0005 = 0.05%)
            swap_fee_cost = 0.5 * swap_fee_rate * self.initial_value_usd
            reward -= swap_fee_cost
        
        # Opportunity cost penalty for out-of-range positions
        # Capital locked in LP but earning 0 fees has an opportunity cost
        # ~5% APY = 0.006% per hour (risk-free rate that could be earned elsewhere)
        if self.has_lp and self.lp_lower_tick is not None:
            lp_lower_price = tick_to_price(self.lp_lower_tick)
            lp_upper_price = tick_to_price(self.lp_upper_tick)
            in_range = lp_lower_price <= price_t1 <= lp_upper_price
            
            if not in_range:
                # Opportunity cost: capital locked in LP earning 0 fees could earn ~5% APY elsewhere
                position_value = self.initial_value_usd if hasattr(self, 'initial_value_usd') else self.initial_capital
                opportunity_rate = 0.0000057  # 5% APY / 8760 hours
                opportunity_cost = position_value * opportunity_rate
                reward -= opportunity_cost
        
        self.idx += 1
        terminated = self.idx >= self.n_steps
        
        return self._get_obs(), reward, terminated, False, {
            "t0": t0,
            "price": price_t1,
            "liquidity": self.liquidity,
        }


def make_env_fn(
    hourly_data,  # HourlyData or HourlyDataExtended
    initial_capital_usd: float = 1000.0,
    gas_cost_usd: float = 0.05,
    action_ticks: List[int] = [0, 1, 3, 5, 10, 20, 40],
    mode: str = "train",
):
    def _init():
        return UniswapV3PaperEnv(
            hourly_data=hourly_data,
            initial_capital_usd=initial_capital_usd,
            gas_cost_usd=gas_cost_usd,
            action_ticks=action_ticks,
            mode=mode,
        )
    return _init


def train_paper_method(
    data_dir: str,
    num_envs: int = 8,
    total_timesteps: int = 1_000_000,
    initial_capital_usd: float = 1000.0,
    save_path: str = "ppo_uniswap_v3_paper",
    eval_freq: int = 10_000,
    action_ticks: List[int] = [0, 1, 3, 5, 10, 20, 40],
):
    """
    Train PPO using the paper's methodology.
    
    This follows Xu & Brini (2025) - arXiv:2501.07508:
    - Hourly resampled data
    - Formula-based fee calculation (Equations 5-6)
    - LVR penalty in reward (Equations 15-17)
    - Discrete action space (tick widths)
    """
    print("=" * 60)
    print("🚀 Paper-Based Uniswap v3 PPO Training")
    print("   Following Xu & Brini (2025) - arXiv:2501.07508")
    print("=" * 60)
    print(f"  Data dir: {data_dir}")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Initial Capital: ${initial_capital_usd}")
    print(f"  Action ticks: {action_ticks}")
    print()
    
    # Prepare hourly data - use extended data if available
    try:
        from uniswap_v3_dqn_paper import prepare_hourly_data_extended
        hourly_data = prepare_hourly_data_extended(data_dir)
        print("  Using extended features (36-dim observation, matching DQN)")
    except ImportError:
        hourly_data = prepare_hourly_data(data_dir)
        print("  Using basic features (8-dim observation)")
    
    print()
    print("🏋️ Creating training environments...")
    
    train_fn = make_env_fn(hourly_data, initial_capital_usd=initial_capital_usd, action_ticks=action_ticks, mode="train")
    eval_fn = make_env_fn(hourly_data, initial_capital_usd=initial_capital_usd, action_ticks=action_ticks, mode="eval")
    
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
    
    # PPO hyperparameters (similar to paper: optimize via grid search)
    n_steps = max(2048 // num_envs, 64)
    batch_size = min(256, n_steps * num_envs)
    
    # Ensure n_steps * num_envs >= batch_size
    if n_steps * num_envs < batch_size:
        batch_size = n_steps * num_envs
    
    print("🧠 Creating PPO model...")
    print(f"  n_steps per env: {n_steps}")
    print(f"  batch_size: {batch_size}")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,  # Paper uses 0.9, 0.99, 0.999, 0.9999 (optimized)
        learning_rate=3e-4,  # Paper range: 1e-5 to 1e-2
        ent_coef=0.01,  # Paper range: 1e-5 to 0.01
        clip_range=0.2,  # Paper range: 0.05 to 0.4
        n_epochs=10,
        gae_lambda=0.95,
        policy_kwargs=dict(
            net_arch=[128, 128],  # Scaled up for 38-dim state space
        ),
        tensorboard_log=None,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1000),
        save_path="./checkpoints_paper/",
        name_prefix="ppo_paper"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_paper/",
        log_path="./eval_logs_paper/",
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
        progress_bar=False,
    )
    
    print()
    if save_path is not None:
        print("💾 Saving model...")
        model.save(save_path)
        env.save(f"{save_path}_vec_normalize.pkl")
        print(f"  Model saved to: {save_path}.zip")
        print(f"  VecNormalize saved to: {save_path}_vec_normalize.pkl")
    
    print()
    print("=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    
    return model, env


def evaluate_paper_method(
    data_dir: str,
    model_path: str = "ppo_uniswap_v3_paper.zip",
    vec_normalize_path: Optional[str] = None,
    n_episodes: int = 10,
    action_ticks: List[int] = [0, 1, 3, 5, 10, 20, 40],
) -> dict:
    """
    Evaluate trained model on test set.
    """
    if vec_normalize_path is None:
        vec_normalize_path = model_path.replace(".zip", "_vec_normalize.pkl")
    
    print("=" * 60)
    print("📊 Evaluation on TEST set (paper methodology)")
    print("=" * 60)
    print(f"  Data dir: {data_dir}")
    print(f"  Model: {model_path}")
    print()
    
    try:
        from uniswap_v3_dqn_paper import prepare_hourly_data_extended
        hourly_data = prepare_hourly_data_extended(data_dir)
    except ImportError:
        hourly_data = prepare_hourly_data(data_dir)
    
    eval_fn = make_env_fn(hourly_data, action_ticks=action_ticks, mode="test")
    env = DummyVecEnv([eval_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(model_path, env=env)
    
    rewards = []
    action_counts = {i: 0 for i in range(len(action_ticks))}
    
    for ep in range(n_episodes):
        # VecNormalize doesn't support seed argument, use plain reset
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, (tuple, list)) else reset_result
        done = False
        total_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action[0])
            action_counts[action_int] += 1
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated[0] or truncated[0])
            else:
                obs, reward, done, info = step_result
                done = bool(done[0])
            
            total_reward += float(reward[0])
        
        rewards.append(total_reward)
    
    # Print action distribution
    total_actions = sum(action_counts.values())
    print("Action distribution:")
    for i, count in action_counts.items():
        tick_width = action_ticks[i]
        label = "HOLD" if tick_width == 0 else f"WIDTH={tick_width}"
        pct = 100 * count / total_actions if total_actions > 0 else 0
        print(f"  {label:15s}: {count:5d} ({pct:.1f}%)")
    print()
    
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    
    print("Results:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("=" * 60)
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "action_distribution": action_counts,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper-based PPO training for Uniswap v3 LP")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default="ppo_uniswap_v3_paper", help="Model save path")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes for evaluation")
    parser.add_argument("--action-ticks", type=str, default="0,10,20,30,40", 
                        help="Comma-separated tick width options (0=hold)")
    
    args = parser.parse_args()
    
    # Parse action ticks
    action_ticks = [int(x) for x in args.action_ticks.split(",")]
    
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        candidates = [
            os.path.join(script_dir, "training_data"),
            os.path.join(cwd, "training_data"),
            # Also check simulation_6 (since we share data)
            os.path.join(os.path.dirname(script_dir), "simulation_6", "training_data"),
            cwd,
        ]
        required = "pool_config_eth_usdt_0p3.csv"
        for d in candidates:
            if d and os.path.isfile(os.path.join(d, required)):
                args.data_dir = os.path.abspath(d)
                break
        else:
            raise FileNotFoundError(
                f"Data not found. Tried: {candidates}. "
                f"Pass --data-dir /path/to/training_data"
            )
    
    if args.evaluate:
        evaluate_paper_method(
            data_dir=args.data_dir,
            model_path=f"{args.save_path}.zip" if not args.save_path.endswith(".zip") else args.save_path,
            n_episodes=args.eval_episodes,
            action_ticks=action_ticks,
        )
    else:
        train_paper_method(
            data_dir=args.data_dir,
            num_envs=args.num_envs,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            action_ticks=action_ticks,
        )
