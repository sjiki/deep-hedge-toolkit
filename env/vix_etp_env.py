"""VIX ETP Reinforcement Learning Environment with comprehensive features."""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any
from gymnasium import spaces

from env.execution_models import ExecutionCostModel
from env.portfolio_constraints import PortfolioConstraints


class VIXETPEnv(gym.Env):
    """
    Gymnasium environment for VIX ETP trading with RL.
    
    Features:
    - Regime-aware cooldown
    - Dynamic slippage & borrow costs
    - Rebalance frequency control
    - Availability mask for IPO/new listings
    - Position bands & turnover caps
    - Risk budgeting using rolling covariance
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None,
        execution_model: Optional[ExecutionCostModel] = None,
        constraints: Optional[PortfolioConstraints] = None,
        initial_cash: float = 100000.0,
        rebalance_frequency: int = 1,
        min_history_days: int = 60,
        observation_window: int = 30,
        reward_type: str = 'sharpe'
    ):
        """
        Initialize VIX ETP environment.
        
        Args:
            price_data: DataFrame with prices (tickers as columns)
            volume_data: DataFrame with volumes (tickers as columns)
            regime_data: DataFrame with regime probabilities (optional)
            execution_model: Execution cost model
            constraints: Portfolio constraints
            initial_cash: Initial portfolio value
            rebalance_frequency: Days between rebalances
            min_history_days: Minimum history required for asset availability
            observation_window: Window for computing observations
            reward_type: Type of reward ('sharpe', 'sortino', 'returns')
        """
        super().__init__()
        
        # Data
        self.price_data = price_data
        self.volume_data = volume_data
        self.regime_data = regime_data
        self.tickers = list(price_data.columns)
        self.n_assets = len(self.tickers)
        
        # Models
        self.execution_model = execution_model or ExecutionCostModel()
        self.constraints = constraints or PortfolioConstraints()
        
        # Environment parameters
        self.initial_cash = initial_cash
        self.rebalance_frequency = rebalance_frequency
        self.min_history_days = min_history_days
        self.observation_window = observation_window
        self.reward_type = reward_type
        
        # State variables
        self.current_step = 0
        self.current_weights = np.zeros(self.n_assets)
        self.portfolio_value = initial_cash
        self.cash = initial_cash
        self.returns_history = []
        self.portfolio_values = []
        self.actions_history = []
        
        # Compute returns for risk budgeting
        self.returns_df = price_data.pct_change().dropna()
        
        # Define action and observation spaces
        # Actions: target weights for each asset
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observations: prices, volumes, regimes, current weights
        obs_dim = (
            self.n_assets * 3 +  # Price returns, volumes, current weights
            (4 if regime_data is not None else 0) +  # Regime probabilities
            3  # Portfolio metrics: value, return, volatility
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Episode data
        self.start_idx = self.min_history_days
        self.max_steps = len(price_data) - self.start_idx - 1
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_weights = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_cash
        self.cash = self.initial_cash
        self.returns_history = []
        self.portfolio_values = [self.initial_cash]
        self.actions_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Target portfolio weights (will be normalized)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to sum to 1
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0, 1)
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = np.ones(self.n_assets) / self.n_assets
        
        # Get current date
        current_date_idx = self.start_idx + self.current_step
        
        # Get availability mask
        availability_mask = self._get_availability_mask(current_date_idx)
        action = action * availability_mask
        
        # Re-normalize after masking
        if action.sum() > 0:
            action = action / action.sum()
        
        # Update spike cooldown
        if self.regime_data is not None:
            regime_proba = self._get_regime_proba(current_date_idx)
            is_spike = regime_proba[2] > 0.5  # Spike is index 2
            self.constraints.update_spike_cooldown(is_spike)
        
        # Apply constraints
        recent_returns = self.returns_df.iloc[max(0, current_date_idx - 60):current_date_idx]
        constrained_action = self.constraints.project_weights(
            action,
            self.current_weights,
            recent_returns if len(recent_returns) > 0 else None
        )
        
        # Compute transaction costs
        old_weights = self.current_weights
        new_weights = constrained_action
        
        prices = self.price_data.iloc[current_date_idx].values
        volumes = self.volume_data.iloc[current_date_idx].values
        
        transaction_cost = self.execution_model.compute_total_transaction_cost(
            pd.Series(old_weights, index=self.tickers),
            pd.Series(new_weights, index=self.tickers),
            pd.Series(prices, index=self.tickers),
            pd.Series(volumes, index=self.tickers),
            self.portfolio_value
        )
        
        # Apply transaction cost
        self.portfolio_value *= (1 - transaction_cost)
        
        # Update positions
        self.current_weights = new_weights
        self.actions_history.append(new_weights.copy())
        
        # Move to next step
        self.current_step += 1
        next_date_idx = self.start_idx + self.current_step
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if not terminated:
            # Compute returns
            old_prices = prices
            new_prices = self.price_data.iloc[next_date_idx].values
            
            asset_returns = (new_prices - old_prices) / old_prices
            portfolio_return = np.sum(self.current_weights * asset_returns)
            
            # Apply borrow costs
            borrow_cost = self.execution_model.compute_borrow_cost(
                pd.DataFrame([self.current_weights], columns=self.tickers),
                pd.DataFrame([new_prices], columns=self.tickers),
                days_held=self.rebalance_frequency
            ).iloc[0]
            
            portfolio_return -= borrow_cost
            
            # Update portfolio value
            self.portfolio_value *= (1 + portfolio_return)
            self.returns_history.append(portfolio_return)
            self.portfolio_values.append(self.portfolio_value)
        else:
            portfolio_return = 0.0
        
        # Compute reward
        reward = self._compute_reward()
        
        # Get observation
        observation = self._get_observation()
        info = self._get_info()
        info['transaction_cost'] = transaction_cost
        info['portfolio_return'] = portfolio_return
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        current_date_idx = self.start_idx + self.current_step
        
        # Price returns (last observation_window days)
        start_idx = max(0, current_date_idx - self.observation_window)
        price_window = self.price_data.iloc[start_idx:current_date_idx + 1]
        
        if len(price_window) > 1:
            returns = price_window.pct_change().iloc[-1].values
        else:
            returns = np.zeros(self.n_assets)
        
        # Normalized volumes
        volume_window = self.volume_data.iloc[start_idx:current_date_idx + 1]
        if len(volume_window) > 0:
            avg_volume = volume_window.mean().values
            current_volume = self.volume_data.iloc[current_date_idx].values
            volume_ratio = current_volume / (avg_volume + 1)
        else:
            volume_ratio = np.ones(self.n_assets)
        
        # Current weights
        weights = self.current_weights
        
        # Portfolio metrics
        if len(self.returns_history) > 0:
            recent_returns = self.returns_history[-self.observation_window:]
            portfolio_return = np.mean(recent_returns)
            portfolio_vol = np.std(recent_returns) if len(recent_returns) > 1 else 0.0
        else:
            portfolio_return = 0.0
            portfolio_vol = 0.0
        
        portfolio_value_scaled = self.portfolio_value / self.initial_cash
        
        # Combine features
        obs = np.concatenate([
            returns,
            volume_ratio,
            weights,
            [portfolio_value_scaled, portfolio_return, portfolio_vol]
        ])
        
        # Add regime probabilities if available
        if self.regime_data is not None:
            regime_proba = self._get_regime_proba(current_date_idx)
            obs = np.concatenate([obs, regime_proba])
        
        return obs.astype(np.float32)
    
    def _get_regime_proba(self, date_idx: int) -> np.ndarray:
        """Get regime probabilities at given date index."""
        if self.regime_data is None:
            return np.zeros(4)
        
        date = self.price_data.index[date_idx]
        if date in self.regime_data.index:
            regime_cols = [col for col in self.regime_data.columns if 'regime_proba' in col]
            return self.regime_data.loc[date, regime_cols].values
        else:
            return np.ones(4) / 4  # Uniform if not available
    
    def _get_availability_mask(self, date_idx: int) -> np.ndarray:
        """Get mask for assets with sufficient history."""
        mask = np.ones(self.n_assets)
        
        for i, ticker in enumerate(self.tickers):
            # Check if we have min_history_days of data
            ticker_data = self.price_data[ticker].iloc[:date_idx + 1]
            valid_data = ticker_data.dropna()
            
            if len(valid_data) < self.min_history_days:
                mask[i] = 0.0
        
        return mask
    
    def _compute_reward(self) -> float:
        """Compute reward based on reward type."""
        if len(self.returns_history) < 2:
            return 0.0
        
        recent_returns = self.returns_history[-self.observation_window:]
        
        if self.reward_type == 'sharpe':
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)
            else:
                sharpe = 0.0
            return sharpe
        
        elif self.reward_type == 'sortino':
            mean_return = np.mean(recent_returns)
            downside_returns = [r for r in recent_returns if r < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino = mean_return / downside_std * np.sqrt(252)
                else:
                    sortino = mean_return * np.sqrt(252) * 10  # High reward for no downside
            else:
                sortino = mean_return * np.sqrt(252) * 10
            return sortino
        
        else:  # returns
            return np.sum(recent_returns)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
            'total_return': (self.portfolio_value / self.initial_cash - 1.0) if self.initial_cash > 0 else 0.0
        }
        
        if len(self.returns_history) > 0:
            info['returns'] = np.array(self.returns_history)
            info['mean_return'] = np.mean(self.returns_history)
            info['volatility'] = np.std(self.returns_history)
            
            if info['volatility'] > 0:
                info['sharpe_ratio'] = info['mean_return'] / info['volatility'] * np.sqrt(252)
            else:
                info['sharpe_ratio'] = 0.0
        
        return info
