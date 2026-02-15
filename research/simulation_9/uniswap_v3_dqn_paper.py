# uniswap_v3_dqn_paper.py
"""
Uniswap v3 Dueling DDQN Training - Paper-Based Approach (Exact Per-Swap)

Implements the methodology from:
  Zhang, Chen & Yang (2023) - "Adaptive Liquidity Provision in Uniswap V3 
                               with Deep Reinforcement Learning"
  arXiv:2309.10129

Key features:
1. Dueling Double DQN (not PPO) - value-based RL
2. LVR (Loss-Versus-Rebalancing) in reward function
3. Extended state space with 32 technical indicators
4. Discrete action space for tick widths
5. EXACT per-swap fee & LVR calculation (following paper Section 3.2)
   - Fees and LVR are summed over every individual swap within each hourly step
   - Not approximated from OHLCV candles
"""

import os
import math
import glob
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import random

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Q96 = 2 ** 96


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Convert Uniswap v3 sqrtPriceX96 to human-readable price."""
    p = float(sqrt_price_x96) / Q96
    return (p * p) * (10 ** (decimals0 - decimals1))


def price_to_tick(price: float) -> int:
    """Convert price to tick index: i = floor(log(p) / log(1.0001))"""
    if price <= 0:
        return 0
    return int(math.floor(math.log(price) / math.log(1.0001)))


def tick_to_price(tick: int) -> float:
    """Convert tick index to price: p(i) = 1.0001^i"""
    return math.pow(1.0001, tick)


# =============================================================================
# Technical Indicators (from Zhang et al. paper - Table 2)
# =============================================================================

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators from OHLCV data.
    Based on Zhang et al. (2023) Table 2 - 28 features.
    """
    # Ensure we have required columns
    if 'close' not in df.columns:
        return df
    
    close = df['close'].values
    high = df['high'].values if 'high' in df.columns else close
    low = df['low'].values if 'low' in df.columns else close
    open_price = df['open'].values if 'open' in df.columns else close
    volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
    
    n = len(close)
    
    # Basic OHLC ratios (4 features)
    df['high_open_ratio'] = high / np.maximum(open_price, 1e-10)
    df['low_open_ratio'] = low / np.maximum(open_price, 1e-10)
    df['close_open_ratio'] = close / np.maximum(open_price, 1e-10)
    
    # Double Exponential Moving Average (DEMA)
    def ema(data, period):
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    ema_12 = ema(close, 12)
    ema_12_12 = ema(ema_12, 12)
    dema = 2 * ema_12 - ema_12_12
    df['dema_ratio'] = dema / np.maximum(open_price, 1e-10)
    
    # Momentum indicators
    # Simple momentum (price change over n periods)
    momentum_12 = np.zeros(n)
    momentum_12[12:] = close[12:] - close[:-12]
    df['momentum_12'] = momentum_12
    
    # Rate of Change
    roc_12 = np.zeros(n)
    roc_12[12:] = (close[12:] - close[:-12]) / np.maximum(close[:-12], 1e-10)
    df['roc_12'] = roc_12
    
    # Average True Range (ATR) - volatility indicator
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)),
                              np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    df['atr_14'] = pd.Series(tr).rolling(14, min_periods=1).mean().values
    df['natr_14'] = df['atr_14'] / np.maximum(close, 1e-10)
    
    # Average Directional Index (ADX)
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with EMA
    atr_smooth = ema(tr, 14)
    plus_di = 100 * ema(plus_dm, 14) / np.maximum(atr_smooth, 1e-10)
    minus_di = 100 * ema(minus_dm, 14) / np.maximum(atr_smooth, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    df['adx_14'] = ema(dx, 14) / 100.0  # Normalize to [0, 1]
    df['plus_di'] = plus_di / 100.0
    df['minus_di'] = minus_di / 100.0
    
    # Commodity Channel Index (CCI)
    typical_price = (high + low + close) / 3.0
    tp_sma = pd.Series(typical_price).rolling(20, min_periods=1).mean().values
    tp_mad = pd.Series(typical_price).rolling(20, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    ).fillna(1).values
    df['cci_20'] = (typical_price - tp_sma) / np.maximum(0.015 * tp_mad, 1e-10) / 200.0  # Normalize
    
    # Relative Strength Index (RSI)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = ema(gain, 14)
    avg_loss = ema(loss, 14)
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    df['rsi_14'] = 1 - 1 / (1 + rs)  # Already [0, 1]
    
    # MACD
    ema_12_close = ema(close, 12)
    ema_26_close = ema(close, 26)
    macd = ema_12_close - ema_26_close
    signal = ema(macd, 9)
    df['macd'] = macd / np.maximum(close, 1e-10)
    df['macd_signal'] = signal / np.maximum(close, 1e-10)
    df['macd_hist'] = (macd - signal) / np.maximum(close, 1e-10)
    
    # Bollinger Bands
    sma_20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std_20 = pd.Series(close).rolling(20, min_periods=1).std().fillna(0).values
    df['bb_upper'] = (sma_20 + 2 * std_20) / np.maximum(close, 1e-10)
    df['bb_lower'] = (sma_20 - 2 * std_20) / np.maximum(close, 1e-10)
    df['bb_width'] = (4 * std_20) / np.maximum(close, 1e-10)
    
    # Stochastic Oscillator
    low_14 = pd.Series(low).rolling(14, min_periods=1).min().values
    high_14 = pd.Series(high).rolling(14, min_periods=1).max().values
    df['stoch_k'] = (close - low_14) / np.maximum(high_14 - low_14, 1e-10)
    df['stoch_d'] = pd.Series(df['stoch_k']).rolling(3, min_periods=1).mean().values
    
    # Volume indicators
    df['volume_sma_ratio'] = volume / np.maximum(
        pd.Series(volume).rolling(20, min_periods=1).mean().values, 1e-10
    )
    
    # ==========================================================================
    # TREND FEATURES - Critical for directional positioning
    # ==========================================================================
    
    # Price returns over different horizons (helps detect trend direction)
    return_1h = np.zeros(n)
    return_1h[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
    df['return_1h'] = return_1h
    
    return_24h = np.zeros(n)
    return_24h[24:] = (close[24:] - close[:-24]) / np.maximum(close[:-24], 1e-10)
    df['return_24h'] = return_24h
    
    return_7d = np.zeros(n)
    return_7d[168:] = (close[168:] - close[:-168]) / np.maximum(close[:-168], 1e-10)
    df['return_7d'] = return_7d
    
    # Moving average trend signals
    ma_50 = pd.Series(close).rolling(50, min_periods=1).mean().values
    ma_200 = pd.Series(close).rolling(200, min_periods=1).mean().values
    
    df['price_vs_ma50'] = (close - ma_50) / np.maximum(ma_50, 1e-10)  # >0 = above MA (bullish)
    df['price_vs_ma200'] = (close - ma_200) / np.maximum(ma_200, 1e-10)
    df['ma50_vs_ma200'] = (ma_50 - ma_200) / np.maximum(ma_200, 1e-10)  # Golden/death cross signal
    
    # Market regime indicator (explicit classification)
    # Based on 7-day return: >3% bull, <-3% bear, else sideways
    regime = np.zeros(n)
    for i in range(168, n):
        ret_7d = (close[i] - close[i-168]) / close[i-168]
        if ret_7d > 0.03:
            regime[i] = 1.0   # Bull
        elif ret_7d < -0.03:
            regime[i] = -1.0  # Bear
        else:
            regime[i] = 0.0   # Sideways
    df['market_regime'] = regime
    
    # Trend strength (absolute return - higher = stronger trend regardless of direction)
    df['trend_strength_24h'] = np.abs(df['return_24h'])
    df['trend_strength_7d'] = np.abs(df['return_7d'])
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df


@dataclass
class HourlyDataExtended:
    """Extended hourly data with technical indicators."""
    timestamps: List[pd.Timestamp]
    prices: Dict[pd.Timestamp, float]
    features: Dict[pd.Timestamp, np.ndarray]  # Technical indicator features
    volumes: Dict[pd.Timestamp, float]
    decimals0: int
    decimals1: int
    pool_fee: float
    tick_spacing: int
    # Per-swap prices for each hour, precomputed from swap-level data.
    # Used for exact per-swap fee & LVR calculation following Zhang et al. (2023).
    # Key: hourly timestamp, Value: numpy array of swap prices within that hour.
    swap_prices_per_hour: Optional[Dict[pd.Timestamp, np.ndarray]] = None


def prepare_hourly_data_extended(data_dir: str) -> HourlyDataExtended:
    """
    Prepare hourly data with extended technical indicators.
    Following Zhang et al. (2023) methodology.
    """
    print("🔄 Preparing hourly data with technical indicators...")
    
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
    pool_fee = pool_fee_bps / 1_000_000
    tick_spacing = int(pool_cfg.loc[0, 'tickSpacing'])
    
    print(f"  Pool fee: {pool_fee*100:.2f}%")
    print(f"  Tick spacing: {tick_spacing}")
    
    # Parse timestamps and compute prices
    swaps['evt_block_time'] = pd.to_datetime(swaps['evt_block_time'], utc=True)
    swaps = swaps.sort_values('evt_block_time').reset_index(drop=True)
    swaps['price'] = swaps['sqrtPriceX96'].apply(
        lambda x: sqrt_price_x96_to_price(int(x), decimals0, decimals1)
    )
    swaps['volume_usd'] = swaps['amount1'].abs() / (10 ** decimals1)
    
    # Precompute per-swap prices for each hour from swap-level data.
    # Following Zhang et al. (2023): fees and LVR are summed over every swap per hour.
    # We store the array of swap prices per hour for exact computation at runtime.
    swaps_indexed_for_prices = swaps.set_index('evt_block_time')
    swap_prices_per_hour_raw = {}
    for hour, group in swaps_indexed_for_prices.groupby(pd.Grouper(freq='1h')):
        if len(group) >= 2:
            swap_prices_per_hour_raw[hour] = group['price'].values.astype(np.float64)
        elif len(group) == 1:
            swap_prices_per_hour_raw[hour] = group['price'].values.astype(np.float64)
        # else: no swaps this hour → not stored (will use close prices as fallback)
    
    # Resample to hourly OHLCV
    swaps.set_index('evt_block_time', inplace=True)
    hourly = swaps.resample('1h').agg({
        'price': ['first', 'last', 'max', 'min'],
        'volume_usd': 'sum'
    })
    hourly.columns = ['open', 'close', 'high', 'low', 'volume']
    hourly = hourly.dropna(subset=['close'])
    
    # Forward-fill missing hours
    full_range = pd.date_range(start=hourly.index.min(), end=hourly.index.max(), freq='1h', tz='UTC')
    hourly = hourly.reindex(full_range)
    hourly['close'] = hourly['close'].ffill()
    hourly['open'] = hourly['open'].ffill()
    hourly['high'] = hourly['high'].ffill()
    hourly['low'] = hourly['low'].ffill()
    hourly['volume'] = hourly['volume'].fillna(0)
    
    print(f"  📊 {len(hourly)} hourly candles")
    
    # Compute technical indicators
    hourly = compute_technical_indicators(hourly)
    
    # Feature columns (22 original + 10 trend features = 32 features)
    feature_cols = [
        # Original technical features (22)
        'high_open_ratio', 'low_open_ratio', 'close_open_ratio',
        'dema_ratio', 'momentum_12', 'roc_12', 'atr_14', 'natr_14',
        'adx_14', 'plus_di', 'minus_di', 'cci_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width',
        'stoch_k', 'stoch_d', 'volume_sma_ratio',
        # NEW: Trend features (10) - critical for directional positioning
        'return_1h', 'return_24h', 'return_7d',
        'price_vs_ma50', 'price_vs_ma200', 'ma50_vs_ma200',
        'market_regime',  # -1 bear, 0 sideways, +1 bull
        'trend_strength_24h', 'trend_strength_7d'
    ]
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in hourly.columns:
            hourly[col] = 0.0
    
    print(f"  📈 Computed {len(feature_cols)} features (22 technical + 9 trend)")
    
    # Convert to dictionaries
    timestamps = list(hourly.index)
    prices = hourly['close'].to_dict()
    volumes = hourly['volume'].to_dict()
    
    # Map per-swap prices to the full (forward-filled) timestamp index
    swap_prices_per_hour = {}
    for ts in timestamps:
        swap_prices_per_hour[ts] = swap_prices_per_hour_raw.get(ts, None)
    
    # Build feature vectors
    features = {}
    for ts in timestamps:
        feat_vec = hourly.loc[ts, feature_cols].values.astype(np.float32)
        # Clip extreme values
        feat_vec = np.clip(feat_vec, -10, 10)
        features[ts] = feat_vec
    
    print("✅ Data preparation complete!")
    
    return HourlyDataExtended(
        timestamps=timestamps,
        prices=prices,
        features=features,
        volumes=volumes,
        decimals0=decimals0,
        decimals1=decimals1,
        pool_fee=pool_fee,
        tick_spacing=tick_spacing,
        swap_prices_per_hour=swap_prices_per_hour,
    )


# =============================================================================
# Dueling DQN Network Architecture
# =============================================================================

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture from Wang et al. (2016).
    Used in Zhang et al. (2023) for Uniswap v3 LP.
    
    Separates value stream V(s) and advantage stream A(s,a).
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        
        # Shared feature layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.feature_layer = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN with N-step returns."""
    
    def __init__(self, capacity: int = 100000, n_steps: int = 1, gamma: float = 0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)
    
    def push(self, state, action, reward, next_state, done):
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_steps:
            # Compute n-step return for the oldest step
            r, n_s, d = self._get_n_step_info()
            s, a = self.n_step_buffer[0][:2]
            self.buffer.append((s, a, r, n_s, d))
            self.n_step_buffer.popleft()
            
        if done:
            # Drain buffer
            while len(self.n_step_buffer) > 0:
                r, n_s, d = self._get_n_step_info()
                s, a = self.n_step_buffer[0][:2]
                self.buffer.append((s, a, r, n_s, d))
                self.n_step_buffer.popleft()
    
    def _get_n_step_info(self):
        reward, next_state, done = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for i, transition in enumerate(self.n_step_buffer):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward += (self.gamma ** i) * r
            if d:
                next_state = n_s
                done = True
                break
        return reward, next_state, done
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# LSTM-based Dueling DQN (Sequence-aware)
# =============================================================================

class LSTMDuelingDQN(nn.Module):
    """
    LSTM-based Dueling DQN for sequence processing.
    
    Instead of seeing a single state, the network sees a sequence of past states
    and learns temporal patterns directly without manual feature engineering.
    
    Architecture:
        Input: (batch, seq_len, state_dim) - sequence of past states
        LSTM: Processes sequence, outputs hidden state
        FC: Feature extraction from LSTM output
        Value + Advantage streams: Dueling architecture
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        seq_len: int = 24,
        lstm_hidden: int = 64,
        fc_hidden: int = 64,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden
        
        # LSTM layer - processes sequences
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,  # Input shape: (batch, seq, features)
            dropout=0.0 if num_lstm_layers == 1 else 0.1
        )
        
        # Feature layer after LSTM
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU()
        )
        
        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(fc_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream (action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, state_dim)
               OR (seq_len, state_dim) for single sample
        
        Returns:
            Q-values of shape (batch, action_dim)
        """
        # Handle single sample (no batch dimension)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last timestep's output
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        
        # Feature extraction
        features = self.fc(last_hidden)
        
        # Dueling architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class SequenceReplayBuffer:
    """
    Replay buffer that stores sequences of states for LSTM training.
    
    Instead of storing single (s, a, r, s', done) tuples, stores:
    (state_sequence, action, reward, next_state_sequence, done)
    """
    
    def __init__(self, capacity: int = 50000, seq_len: int = 24):
        self.buffer = deque(maxlen=capacity)
        self.seq_len = seq_len
    
    def push(self, state_seq, action, reward, next_state_seq, done):
        """
        Push a transition with state sequences.
        
        Args:
            state_seq: numpy array of shape (seq_len, state_dim)
            action: int
            reward: float
            next_state_seq: numpy array of shape (seq_len, state_dim)
            done: bool
        """
        self.buffer.append((state_seq, action, reward, next_state_seq, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state_seqs, actions, rewards, next_state_seqs, dones = zip(*batch)
        return (
            np.array(state_seqs),      # (batch, seq_len, state_dim)
            np.array(actions),          # (batch,)
            np.array(rewards),          # (batch,)
            np.array(next_state_seqs),  # (batch, seq_len, state_dim)
            np.array(dones)             # (batch,)
        )
    
    def __len__(self):
        return len(self.buffer)


class SequenceStateWrapper:
    """
    Wrapper that maintains a sequence of past states for LSTM input.
    
    Keeps a rolling window of the last `seq_len` states.
    """
    
    def __init__(self, state_dim: int, seq_len: int = 24):
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.reset()
    
    def reset(self):
        """Reset the state history (fill with zeros)."""
        self.history = deque(maxlen=self.seq_len)
        # Initialize with zeros
        for _ in range(self.seq_len):
            self.history.append(np.zeros(self.state_dim, dtype=np.float32))
    
    def push(self, state: np.ndarray):
        """Add a new state to the history."""
        self.history.append(state.copy())
    
    def get_sequence(self) -> np.ndarray:
        """Get the current state sequence as numpy array."""
        return np.array(list(self.history), dtype=np.float32)  # (seq_len, state_dim)


# =============================================================================
# Environment
# =============================================================================

class UniswapV3DQNEnv(gym.Env):
    """
    Uniswap v3 LP environment following Zhang et al. (2023).
    
    Key differences from PPO version:
    1. Extended state space (28+ features)
    2. LVR in reward function
    3. Designed for discrete action DQN
    
    Enhanced Action Space (v2):
    - HOLD (action=0): Keep current position
    - Width + Offset combinations: Allow agent to position LP range
      above or below current price to express directional views
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        hourly_data: HourlyDataExtended,
        initial_capital_usd: float = 1000.0,  # $1000 initial capital
        gas_cost_usd: float = 0.50,  # Arbitrum L2 gas cost (~$0.50 per rebalance)
        mode: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        
        self.hourly_data = hourly_data
        self.initial_capital = float(initial_capital_usd)
        self.gas_cost = float(gas_cost_usd)
        self.pool_fee = hourly_data.pool_fee
        self.tick_spacing = hourly_data.tick_spacing
        
        # =====================================================================
        # ACTION SPACE: HOLD + Width selection (always centered)
        # =====================================================================
        # Widths: Different LP range sizes (in tick_spacing units)
        self.widths = [1, 3, 5, 10, 20, 40]  # ±10 to ±400 ticks (~0.1% to ~4% ranges)
        
        # Build action mapping: action_id -> width (None = HOLD)
        # Action 0 = HOLD, Action 1-5 = widths, always centered at current price
        self.action_map = {0: None}  # HOLD
        for i, width in enumerate(self.widths):
            self.action_map[i + 1] = width
        
        self.n_actions = len(self.action_map)
        self.max_width = max(self.widths)
        
        # Data split
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
        
        self.n_steps = len(self.timestamps) - 1
        
        # State space: technical features (31) + position info (5) = 36
        n_tech_features = 31
        self.state_dim = n_tech_features + 7  # features + (new features)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32
        )
        
        # Discrete action space: HOLD + 5 widths = 6 actions
        self.action_space = spaces.Discrete(self.n_actions)
        
        self._reset_state()
        
        # Print action space info on first creation
        print(f"🎮 Action Space: {self.n_actions} actions")
        print(f"   - HOLD (action=0)")
        print(f"   - Widths: {self.widths} ticks (always centered)")

    def _reset_state(self):
        self.idx = 0
        self.cash = self.initial_capital
        self.position_value = 0.0
        self.has_position = False
        self.position_width = 0
        self.position_center_tick = 0
        self.position_entry_price = 0.0
        self.position_entry_idx = 0  # Track when position was opened
        self.liquidity = 0.0

    def _get_price(self, t: pd.Timestamp) -> float:
        return self.hourly_data.prices.get(t, 0.0)

    def _get_ma_168h(self, t: pd.Timestamp) -> float:
        return self.hourly_data.ma_168h.get(t, 0.0)

    def _get_volume(self, t: pd.Timestamp) -> float:
        return self.hourly_data.volumes.get(t, 0.0)

    def _get_features(self, t: pd.Timestamp) -> np.ndarray:
        return self.hourly_data.features.get(t, np.zeros(31, dtype=np.float32))

    def _compute_position_value(self, price: float) -> float:
        """Compute current position value using Uniswap v3 formula."""
        if not self.has_position or self.liquidity <= 0:
            return 0.0
        
        tick = price_to_tick(price)
        lower_tick = self.position_center_tick - self.position_width * self.tick_spacing
        upper_tick = self.position_center_tick + self.position_width * self.tick_spacing
        
        p_lower = tick_to_price(lower_tick)
        p_upper = tick_to_price(upper_tick)
        
        L = self.liquidity
        sqrt_p = math.sqrt(price)
        sqrt_pl = math.sqrt(p_lower)
        sqrt_pu = math.sqrt(p_upper)
        
        if price <= p_lower:
            # All in token X
            x = L * (1.0 / sqrt_pl - 1.0 / sqrt_pu)
            return x * price
        elif price >= p_upper:
            # All in token Y
            y = L * (sqrt_pu - sqrt_pl)
            return y
        else:
            # In range: V = L * (2√p - p/√p_u - √p_l)
            return L * (2.0 * sqrt_p - price / sqrt_pu - sqrt_pl)

    def _compute_fee(self, price_t0: float, price_t1: float) -> float:
        """
        Compute trading fee by summing Equation (3) over every swap in this hour.
        Exact per-swap computation following Zhang et al. (2023) Section 3.2.
        
        For each consecutive swap price pair (p_i, p_{i+1}):
          - Clamp both to LP range [p_lower, p_upper]
          - fee_i = (δ/(1-δ)) × L × |√p_{i+1}_c - √p_i_c|  (adjusted for direction)
        Total fee = Σ fee_i
        """
        if not self.has_position or self.liquidity <= 0:
            return 0.0
        
        lower_tick = self.position_center_tick - self.position_width * self.tick_spacing
        upper_tick = self.position_center_tick + self.position_width * self.tick_spacing
        p_lower = tick_to_price(lower_tick)
        p_upper = tick_to_price(upper_tick)
        
        delta = self.pool_fee
        fee_mult = delta / (1.0 - delta)
        L = self.liquidity
        
        # Get precomputed per-swap prices for this hour
        t0 = self.timestamps[self.idx]
        swap_prices = self.hourly_data.swap_prices_per_hour.get(t0, None) if self.hourly_data.swap_prices_per_hour else None
        
        if swap_prices is not None and len(swap_prices) >= 2:
            # Exact per-swap fee computation (paper method)
            total_fee = 0.0
            sqrt_pl = math.sqrt(p_lower)
            sqrt_pu = math.sqrt(p_upper)
            
            for i in range(len(swap_prices) - 1):
                p0 = swap_prices[i]
                p1 = swap_prices[i + 1]
                
                # Skip if both outside range on same side
                if (p0 < p_lower and p1 < p_lower) or (p0 > p_upper and p1 > p_upper):
                    continue
                
                # Clamp prices to LP range
                p0_c = max(p_lower, min(p_upper, p0))
                p1_c = max(p_lower, min(p_upper, p1))
                
                sqrt_p0_c = math.sqrt(p0_c)
                sqrt_p1_c = math.sqrt(p1_c)
                
                if sqrt_p0_c <= sqrt_p1_c:
                    # Price increase: Equation 5
                    fee_i = fee_mult * L * (sqrt_p1_c - sqrt_p0_c)
                else:
                    # Price decrease: Equation 6
                    fee_i = fee_mult * L * (1.0 / sqrt_p1_c - 1.0 / sqrt_p0_c) * p1_c
                
                total_fee += max(0.0, fee_i)
            
            return total_fee
        else:
            # Fallback to open-close formula if no swap data available
            if (price_t0 < p_lower and price_t1 < p_lower) or \
               (price_t0 > p_upper and price_t1 > p_upper):
                return 0.0
            
            p0_clamped = max(p_lower, min(p_upper, price_t0))
            p1_clamped = max(p_lower, min(p_upper, price_t1))
            
            if p0_clamped <= p1_clamped:
                fee = fee_mult * L * (math.sqrt(p1_clamped) - math.sqrt(p0_clamped))
            else:
                fee = fee_mult * L * (1.0/math.sqrt(p1_clamped) - 1.0/math.sqrt(p0_clamped)) * p1_clamped
            
            return max(0.0, fee)

    def _compute_lvr(self, price_t0: float, price_t1: float) -> float:
        """
        Compute LVR (Loss-Versus-Rebalancing) by summing Equation (5) over every swap.
        Exact per-swap computation following Zhang et al. (2023) Section 3.2.
        
        LVR = Σ {V(p_{i+1}) - V(p_i) - x(p_i) × (p_{i+1} - p_i)}
        
        This is always <= 0 (it's a cost).
        """
        if not self.has_position or self.liquidity <= 0:
            return 0.0
        
        lower_tick = self.position_center_tick - self.position_width * self.tick_spacing
        upper_tick = self.position_center_tick + self.position_width * self.tick_spacing
        p_lower = tick_to_price(lower_tick)
        p_upper = tick_to_price(upper_tick)
        
        L = self.liquidity
        sqrt_pl = math.sqrt(p_lower)
        sqrt_pu = math.sqrt(p_upper)
        
        # Get precomputed per-swap prices for this hour
        t0 = self.timestamps[self.idx]
        swap_prices = self.hourly_data.swap_prices_per_hour.get(t0, None) if self.hourly_data.swap_prices_per_hour else None
        
        if swap_prices is not None and len(swap_prices) >= 2:
            # Exact per-swap LVR computation (paper Equation 5)
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
            V_t0 = self._compute_position_value(price_t0)
            V_t1 = self._compute_position_value(price_t1)
            
            if price_t0 <= p_lower:
                x_t0 = L * (1.0 / sqrt_pl - 1.0 / sqrt_pu)
            elif price_t0 >= p_upper:
                x_t0 = 0.0
            else:
                x_t0 = L * (1.0 / math.sqrt(price_t0) - 1.0 / sqrt_pu)
            
            return (V_t1 - V_t0) - x_t0 * (price_t1 - price_t0)

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

    def _get_obs(self) -> np.ndarray:
        if self.idx >= len(self.timestamps):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        t = self.timestamps[self.idx]
        price = self._get_price(t)
        
        # Technical features (31)
        tech_features = self._get_features(t)
        
        # Position features (7)
        total_value = self.cash + self._compute_position_value(price)
        cash_ratio = self.cash / max(self.initial_capital, 1e-10)
        width_normalized = self.position_width / max(self.max_width, 1)
        position_value_ratio = self._compute_position_value(price) / max(self.initial_capital, 1e-10)
        
        # In-range check
        in_range = 0.0
        if self.has_position:
            tick = price_to_tick(price)
            lower_tick = self.position_center_tick - self.position_width * self.tick_spacing
            upper_tick = self.position_center_tick + self.position_width * self.tick_spacing
            in_range = 1.0 if lower_tick <= tick <= upper_tick else 0.0
        
        # Price momentum: short-term direction indicator
        if self.idx >= 1:
            prev_t = self.timestamps[self.idx - 1]
            prev_price = self._get_price(prev_t)
            price_momentum = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        else:
            price_momentum = 0.0
            
        # Dist to boundary (NEW)
        dist_to_boundary = 0.0
        if self.has_position and in_range > 0.5:
            tick = price_to_tick(price)
            lower_tick = self.position_center_tick - self.position_width * self.tick_spacing
            upper_tick = self.position_center_tick + self.position_width * self.tick_spacing
            dist = min(abs(tick - lower_tick), abs(upper_tick - tick))
            half_width_ticks = (self.position_width * self.tick_spacing) / 2.0
            # Normalize by half the total width of the position in ticks
            dist_to_boundary = dist / half_width_ticks if half_width_ticks > 0 else 0.0
            
        # Hours since rebalance (NEW)
        hours_since_rebalance = 0.0
        if self.has_position:
            # Normalized by 24 hours, saturated at 1.0
            hours_since_rebalance = min((self.idx - self.position_entry_idx) / 24.0, 1.0)
        
        position_features = np.array([
            cash_ratio, width_normalized, in_range, position_value_ratio,
            price_momentum, dist_to_boundary, hours_since_rebalance
        ], dtype=np.float32)
        
        return np.concatenate([tech_features, position_features])

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
        
        reward = 0.0
        
        # Decode action using action_map
        width = self.action_map.get(action, None)
        
        if width is None:
            # Action 0: HOLD - keep existing position, no gas cost
            if self.has_position:
                fee = self._compute_fee(price_t0, price_t1)
                
                # CAP FEE to total pool fees
                volume_usd = self._get_volume(t0)
                max_fee = volume_usd * self.pool_fee
                fee = min(fee, max_fee)

                lvr = self._compute_lvr(price_t0, price_t1)
                reward = fee + lvr  # LVR is negative
                self.cash += fee
        else:
            # Reallocate with width, always centered at current price
            
            # First, close existing position
            if self.has_position:
                position_val = self._compute_position_value(price_t0)
                self.cash += position_val
                self.has_position = False
            
            # Gas cost
            self.cash -= self.gas_cost
            reward -= self.gas_cost
            
            # Swap fee cost: ~50% of position needs swapping during rebalance
            swap_fee_rate = self.pool_fee
            swap_fee_cost = 0.5 * swap_fee_rate * self.initial_capital
            self.cash -= swap_fee_cost
            reward -= swap_fee_cost
            
            # Calculate center tick (always centered at current price)
            tick_t0 = price_to_tick(price_t0)
            center_tick = (tick_t0 // self.tick_spacing) * self.tick_spacing
            
            lower_tick = center_tick - width * self.tick_spacing
            upper_tick = center_tick + width * self.tick_spacing
            p_lower = tick_to_price(lower_tick)
            p_upper = tick_to_price(upper_tick)
            
            # Invest all cash into position
            invest_amount = max(0.0, self.cash)
            
            # Compute liquidity from invest amount
            sqrt_p = math.sqrt(price_t0)
            sqrt_pl = math.sqrt(p_lower)
            sqrt_pu = math.sqrt(p_upper)
            
            if price_t0 <= p_lower:
                value_per_L = (1.0 / sqrt_pl - 1.0 / sqrt_pu) * price_t0
            elif price_t0 >= p_upper:
                value_per_L = sqrt_pu - sqrt_pl
            else:
                value_per_L = 2.0 * sqrt_p - price_t0 / sqrt_pu - sqrt_pl
            
            if value_per_L > 0:
                new_L = invest_amount / value_per_L
            else:
                new_L = 0.0
            
            self.has_position = True
            self.position_width = width
            self.position_center_tick = center_tick
            self.position_entry_price = price_t0
            self.position_entry_idx = self.idx  # Track entry time
            self.liquidity = new_L
            self.cash = 0.0
            
            # Collect fees for this step
            fee = self._compute_fee(price_t0, price_t1)
            
            # CAP FEE to total pool fees (fix for reward explosion)
            # You cannot earn more than the total fees generated by the pool in that hour
            volume_usd = self._get_volume(t0)
            max_fee = volume_usd * self.pool_fee
            fee = min(fee, max_fee)

            lvr = self._compute_lvr(price_t0, price_t1)
            reward += fee + lvr
            self.cash += fee
        
        # Update position value
        self.position_value = self._compute_position_value(price_t1)
        
        # Track in_range for info
        in_range = False
        if self.has_position:
            tick = price_to_tick(price_t1)
            lower_tick = self.position_center_tick - self.position_width * self.tick_spacing
            upper_tick = self.position_center_tick + self.position_width * self.tick_spacing
            in_range = lower_tick <= tick <= upper_tick
        
        # Opportunity cost when out of range
        if self.has_position and not in_range:
            opportunity_rate = 0.0000057  # 5% APY / 8760 hours
            opportunity_cost = self.position_value * opportunity_rate
            reward -= opportunity_cost
        
        self.idx += 1
        terminated = self.idx >= self.n_steps
        
        return self._get_obs(), reward, terminated, False, {
            "price": price_t1,
            "total_value": self.cash + self.position_value,
            "in_range": in_range,
            "action_width": width,
        }


# =============================================================================
# Dueling DDQN Agent
# =============================================================================

class DuelingDDQNAgent:
    """
    Dueling Double DQN agent following Zhang et al. (2023).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,  # Per-episode decay (not per-step)
        target_update_rate: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 256,
        n_steps: int = 1,  # Added n_steps
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.gamma_n = gamma ** n_steps  # For n-step updates
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.q_network = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, n_steps=n_steps, gamma=gamma)
        
        self.train_step = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            # Use gamma_n (gamma ^ n_steps) for target calculation
            target_q = rewards + self.gamma_n * next_q * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.7)
        self.optimizer.step()
        
        # Soft update target network
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.target_update_rate * local_param.data + 
                (1.0 - self.target_update_rate) * target_param.data
            )
        
        # Note: epsilon decay moved to per-episode in training loop
        
        self.train_step += 1
        return loss.item()

    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']


# =============================================================================
# LSTM-based Agent
# =============================================================================

class LSTMDDQNAgent:
    """
    LSTM-based Dueling Double DQN agent.
    
    Uses sequences of past states instead of single states for temporal awareness.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 24,  # Look back 24 hours
        lstm_hidden: int = 64,
        fc_hidden: int = 64,
        lr: float = 1e-4,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,
        target_update_rate: float = 0.01,
        buffer_size: int = 50000,  # Smaller buffer (sequences take more memory)
        batch_size: int = 64,  # Smaller batch (sequences take more memory)
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # LSTM Networks
        self.q_network = LSTMDuelingDQN(
            state_dim, action_dim, seq_len, lstm_hidden, fc_hidden
        ).to(self.device)
        self.target_network = LSTMDuelingDQN(
            state_dim, action_dim, seq_len, lstm_hidden, fc_hidden
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = SequenceReplayBuffer(buffer_size, seq_len)
        
        # State sequence wrapper for maintaining history
        self.state_wrapper = SequenceStateWrapper(state_dim, seq_len)
        
        self.train_step = 0

    def reset_sequence(self):
        """Reset state history for new episode."""
        self.state_wrapper.reset()

    def update_sequence(self, state: np.ndarray):
        """Add new state to history."""
        self.state_wrapper.push(state)

    def get_current_sequence(self) -> np.ndarray:
        """Get current state sequence."""
        return self.state_wrapper.get_sequence()

    def select_action(self, state_seq: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action based on state sequence.
        
        Args:
            state_seq: numpy array of shape (seq_len, state_dim)
            deterministic: if True, always pick best action (no exploration)
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            # Shape: (1, seq_len, state_dim)
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state_seq, action, reward, next_state_seq, done):
        """Store sequence-based transition."""
        self.replay_buffer.push(state_seq, action, reward, next_state_seq, done)

    def update(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        state_seqs, actions, rewards, next_state_seqs, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors - shapes: (batch, seq_len, state_dim)
        state_seqs = torch.FloatTensor(state_seqs).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_seqs = torch.FloatTensor(next_state_seqs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.q_network(state_seqs).gather(1, actions)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_state_seqs).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_state_seqs).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.7)
        self.optimizer.step()
        
        # Soft update target network
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.target_update_rate * local_param.data + 
                (1.0 - self.target_update_rate) * target_param.data
            )
        
        self.train_step += 1
        return loss.item()

    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'seq_len': self.seq_len,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_lstm_dqn(
    data_dir: str,
    n_episodes: int = 1000,
    seq_len: int = 24,
    max_steps_per_episode: Optional[int] = None,
    save_path: str = "lstm_dqn_uniswap_v3",
    eval_freq: int = 50,
    device: str = "cpu",
):
    """
    Train LSTM-based Dueling DDQN.
    
    Uses sequences of past states for temporal pattern learning.
    """
    print("=" * 60)
    print("🧠 LSTM Dueling DDQN Training")
    print("=" * 60)
    print(f"  Data dir: {data_dir}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Sequence length: {seq_len} hours")
    print(f"  Device: {device}")
    print()
    
    # Prepare data
    hourly_data = prepare_hourly_data_extended(data_dir)
    
    # Create environments
    train_env = UniswapV3DQNEnv(hourly_data, mode="train")
    eval_env = UniswapV3DQNEnv(hourly_data, mode="eval")
    
    print(f"  State dim: {train_env.state_dim}")
    print(f"  Action dim: {train_env.action_space.n}")
    print(f"  Network input shape: ({seq_len}, {train_env.state_dim})")
    print()
    
    # Create LSTM agent
    agent = LSTMDDQNAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_space.n,
        seq_len=seq_len,
        lstm_hidden=64,
        fc_hidden=64,
        lr=1e-4,
        gamma=0.9,
        device=device,
    )
    
    print("🏃 Starting LSTM training...", flush=True)
    print("=" * 60)
    
    best_eval_reward = -float('inf')
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = train_env.reset()
        agent.reset_sequence()  # Reset state history
        agent.update_sequence(state)  # Add initial state
        
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done:
            # Get sequence and select action
            state_seq = agent.get_current_sequence()
            action = agent.select_action(state_seq)
            
            # Environment step
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Update sequence with new state
            agent.update_sequence(next_state)
            next_state_seq = agent.get_current_sequence()
            
            # Store sequence-based transition
            agent.store_transition(state_seq, action, reward, next_state_seq, done)
            loss = agent.update()
            
            episode_reward += reward
            step += 1
            
            if max_steps_per_episode and step >= max_steps_per_episode:
                break
        
        episode_rewards.append(episode_reward)
        
        # Decay epsilon per episode
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}", flush=True)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_lstm_agent(agent, eval_env, n_episodes=5)
            print(f"  [EVAL] Episode {episode+1} | Eval Reward: {eval_reward:.2f}", flush=True)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                if save_path is not None:
                    agent.save(f"{save_path}_best.pth")
                    print(f"  [BEST] New best model saved!")
    
    # Save final model
    if save_path is not None:
        agent.save(f"{save_path}_final.pth")
        print(f"  Model saved to: {save_path}_final.pth")
    
    print()
    print("=" * 60)
    print("✅ LSTM Training complete!")
    print(f"  Best eval reward: {best_eval_reward:.2f}")
    print("=" * 60)
    
    return agent


def evaluate_lstm_agent(agent: LSTMDDQNAgent, env: UniswapV3DQNEnv, n_episodes: int = 10) -> float:
    """Evaluate LSTM agent on environment."""
    total_rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        agent.reset_sequence()
        agent.update_sequence(state)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            state_seq = agent.get_current_sequence()
            action = agent.select_action(state_seq, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update_sequence(next_state)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def train_dqn(
    data_dir: str,
    n_episodes: int = 500,
    max_steps_per_episode: Optional[int] = None,
    save_path: str = "dqn_uniswap_v3_paper",
    eval_freq: int = 50,
    device: str = "cpu",
    n_steps: int = 3,  # Default n=3 step returns
):
    """
    Train Dueling DDQN agent.
    """
    # Zhang et al. (2023) methodology is referenced in the print statement below.
    print("=" * 60)
    print("🚀 Dueling DDQN Training (Zhang et al. 2023 methodology)")
    print("=" * 60)
    print(f"  Data dir: {data_dir}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Device: {device}")
    print()
    
    # Prepare data
    hourly_data = prepare_hourly_data_extended(data_dir)
    
    # Create environments
    train_env = UniswapV3DQNEnv(hourly_data, mode="train")
    eval_env = UniswapV3DQNEnv(hourly_data, mode="eval")
    
    print(f"  State dim: {train_env.state_dim}")
    print(f"  Action dim: {train_env.action_space.n}")
    print()
    
    # Create agent
    agent = DuelingDDQNAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_space.n,
        hidden_dims=[128, 128],
        lr=1e-4,
        gamma=0.99,
        n_steps=n_steps,
        device=device,
    )
    
    print("🏃 Starting training...")
    print("=" * 60)
    
    best_eval_reward = -float('inf')
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = train_env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if max_steps_per_episode and step >= max_steps_per_episode:
                break
        
        episode_rewards.append(episode_reward)
        
        # Decay epsilon per episode
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}", flush=True)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(agent, eval_env, n_episodes=5)
            print(f"  [EVAL] Episode {episode+1} | Eval Reward: {eval_reward:.2f}", flush=True)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                if save_path is not None:
                    agent.save(f"{save_path}_best.pth")
                    print(f"  [BEST] New best model saved!")
    
    # Save final model
    if save_path is not None:
        agent.save(f"{save_path}_final.pth")
        print(f"  Model saved to: {save_path}_final.pth")
    
    print()
    print("=" * 60)
    print("✅ Training complete!")
    print(f"  Best eval reward: {best_eval_reward:.2f}")
    print("=" * 60)
    
    return agent


def evaluate_agent(agent: DuelingDDQNAgent, env: UniswapV3DQNEnv, n_episodes: int = 10) -> float:
    """Evaluate agent on environment."""
    total_rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def evaluate_on_test(
    data_dir: str,
    model_path: str = "dqn_uniswap_v3_paper_best.pth",
    n_episodes: int = 10,
    device: str = "cpu",
) -> dict:
    """Evaluate trained model on test set."""
    print("=" * 60)
    print("📊 Evaluation on TEST set (Dueling DDQN)")
    print("=" * 60)
    
    hourly_data = prepare_hourly_data_extended(data_dir)
    test_env = UniswapV3DQNEnv(hourly_data, mode="test")
    
    agent = DuelingDDQNAgent(
        state_dim=test_env.state_dim,
        action_dim=test_env.action_space.n,
        device=device,
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration
    
    rewards = []
    action_counts = {i: 0 for i in range(test_env.action_space.n)}
    
    for ep in range(n_episodes):
        state, _ = test_env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            action_counts[action] += 1
            state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    # Print results
    total_actions = sum(action_counts.values())
    print("\nAction distribution:")
    for action, count in action_counts.items():
        label = "HOLD" if action == 0 else f"WIDTH={action}"
        pct = 100 * count / total_actions if total_actions > 0 else 0
        print(f"  {label:10s}: {count:5d} ({pct:.1f}%)")
    
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    
    print(f"\nResults:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("=" * 60)
    
    return {"mean_reward": mean_reward, "std_reward": std_reward}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dueling DDQN for Uniswap v3 LP (Zhang et al. 2023)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--save-path", type=str, default="dqn_uniswap_v3_paper")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "training_data"),
            os.path.join(os.path.dirname(script_dir), "simulation_6", "training_data"),
            os.path.join(os.path.dirname(script_dir), "simulation_8", "training_data"),
        ]
        for d in candidates:
            if os.path.isfile(os.path.join(d, "pool_config_eth_usdt_0p3.csv")):
                args.data_dir = d
                break
        else:
            raise FileNotFoundError("Data not found")
    
    if args.evaluate:
        evaluate_on_test(args.data_dir, f"{args.save_path}_best.pth", device=args.device)
    else:
        train_dqn(args.data_dir, args.episodes, save_path=args.save_path, device=args.device)
