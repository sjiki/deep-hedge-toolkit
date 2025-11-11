"""Portfolio constraints including risk budgeting, turnover, and position bands."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy.optimize import minimize


class PortfolioConstraints:
    """Enforce portfolio constraints: leverage, position bands, turnover, risk budget."""
    
    def __init__(
        self,
        max_leverage: float = 1.0,
        max_position: float = 0.4,
        min_position: float = 0.0,
        max_turnover_daily: float = 0.2,
        position_bands: Optional[Dict[str, float]] = None,
        use_risk_budgeting: bool = True,
        covariance_window: int = 60,
        spike_cooldown_days: int = 5,
        cooldown_turnover_scale: float = 0.5
    ):
        """
        Initialize portfolio constraints.
        
        Args:
            max_leverage: Maximum leverage (1.0 = long-only)
            max_position: Maximum position size per asset
            min_position: Minimum position size (0.0 = long-only)
            max_turnover_daily: Maximum daily turnover (fraction of portfolio)
            position_bands: Dict with 'min' and 'max' position bands
            use_risk_budgeting: Whether to use covariance-based risk budgeting
            covariance_window: Window for covariance estimation
            spike_cooldown_days: Days to reduce trading after spike
            cooldown_turnover_scale: Turnover scaling during cooldown
        """
        self.max_leverage = max_leverage
        self.max_position = max_position
        self.min_position = min_position
        self.max_turnover_daily = max_turnover_daily
        self.position_bands = position_bands or {'min': 0.05, 'max': 0.35}
        self.use_risk_budgeting = use_risk_budgeting
        self.covariance_window = covariance_window
        self.spike_cooldown_days = spike_cooldown_days
        self.cooldown_turnover_scale = cooldown_turnover_scale
        
        self.days_since_spike = 0
        
    def update_spike_cooldown(self, is_spike: bool):
        """
        Update spike cooldown counter.
        
        Args:
            is_spike: Whether current regime is spike
        """
        if is_spike:
            self.days_since_spike = 0
        else:
            self.days_since_spike += 1
    
    def is_in_cooldown(self) -> bool:
        """Check if in spike cooldown period."""
        return self.days_since_spike < self.spike_cooldown_days
    
    def get_effective_turnover_limit(self) -> float:
        """Get turnover limit adjusted for cooldown."""
        if self.is_in_cooldown():
            return self.max_turnover_daily * self.cooldown_turnover_scale
        return self.max_turnover_daily
    
    def apply_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply position size limits.
        
        Args:
            weights: Target position weights
            
        Returns:
            Constrained weights
        """
        # Clip to position limits
        weights = np.clip(weights, self.min_position, self.max_position)
        
        # Apply position bands
        band_min = self.position_bands['min']
        band_max = self.position_bands['max']
        
        # Zero out positions below minimum band
        weights = np.where(np.abs(weights) < band_min, 0.0, weights)
        
        # Clip to maximum band
        weights = np.clip(weights, -band_max, band_max)
        
        return weights
    
    def apply_leverage_constraint(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply leverage constraint by normalizing weights.
        
        Args:
            weights: Target position weights
            
        Returns:
            Normalized weights
        """
        total_leverage = np.abs(weights).sum()
        
        if total_leverage > self.max_leverage:
            # Scale down to meet leverage constraint
            weights = weights * (self.max_leverage / total_leverage)
        
        return weights
    
    def apply_turnover_constraint(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply turnover constraint by limiting position changes.
        
        Args:
            old_weights: Current position weights
            new_weights: Target position weights
            
        Returns:
            Constrained weights respecting turnover limit
        """
        turnover_limit = self.get_effective_turnover_limit()
        
        # Compute desired changes
        changes = new_weights - old_weights
        current_turnover = np.abs(changes).sum()
        
        if current_turnover > turnover_limit:
            # Scale changes to meet turnover constraint
            scale = turnover_limit / current_turnover
            changes = changes * scale
            new_weights = old_weights + changes
        
        return new_weights
    
    def compute_risk_budget_weights(
        self,
        returns: pd.DataFrame,
        equal_weight: bool = True
    ) -> np.ndarray:
        """
        Compute risk-budgeted weights using covariance matrix.
        
        Args:
            returns: DataFrame of asset returns
            equal_weight: Whether to use equal risk budgeting
            
        Returns:
            Risk-budgeted weights
        """
        if len(returns) < self.covariance_window:
            # Not enough data, use equal weights
            n_assets = returns.shape[1]
            return np.ones(n_assets) / n_assets
        
        # Use recent window for covariance
        recent_returns = returns.iloc[-self.covariance_window:]
        
        # Compute covariance matrix
        cov_matrix = recent_returns.cov().values
        
        # Add small regularization to ensure positive definite
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8
        
        if equal_weight:
            # Equal risk contribution
            weights = self._equal_risk_contribution(cov_matrix)
        else:
            # Minimum variance
            weights = self._minimum_variance(cov_matrix)
        
        return weights
    
    def _equal_risk_contribution(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Compute equal risk contribution weights."""
        n_assets = len(cov_matrix)
        
        # Start with equal weights
        x0 = np.ones(n_assets) / n_assets
        
        def objective(w):
            # Risk contribution: w * (Cov @ w)
            portfolio_var = w @ cov_matrix @ w
            risk_contrib = w * (cov_matrix @ w)
            target_risk = portfolio_var / n_assets
            
            # Minimize squared deviation from equal risk
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        bounds = [(0, self.max_position) for _ in range(n_assets)]
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return x0
    
    def _minimum_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Compute minimum variance weights."""
        n_assets = len(cov_matrix)
        
        def objective(w):
            return w @ cov_matrix @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]
        
        bounds = [(0, self.max_position) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            return x0
    
    def project_weights(
        self,
        target_weights: np.ndarray,
        old_weights: np.ndarray,
        returns: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Project target weights through all constraints.
        
        Args:
            target_weights: Desired position weights
            old_weights: Current position weights
            returns: Recent returns for risk budgeting (optional)
            
        Returns:
            Constrained weights
        """
        weights = target_weights.copy()
        
        # Apply risk budgeting if enabled and returns available
        if self.use_risk_budgeting and returns is not None:
            risk_weights = self.compute_risk_budget_weights(returns)
            # Blend with target weights
            weights = 0.5 * weights + 0.5 * risk_weights
        
        # Apply position limits
        weights = self.apply_position_limits(weights)
        
        # Apply turnover constraint
        weights = self.apply_turnover_constraint(old_weights, weights)
        
        # Apply leverage constraint
        weights = self.apply_leverage_constraint(weights)
        
        # Ensure weights sum to <= max_leverage
        total = weights.sum()
        if total > self.max_leverage:
            weights = weights * (self.max_leverage / total)
        
        return weights


def create_constraints_from_config(config: Dict) -> PortfolioConstraints:
    """
    Create PortfolioConstraints from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PortfolioConstraints instance
    """
    constraints_config = config.get('constraints', {})
    risk_config = config.get('risk', {})
    
    return PortfolioConstraints(
        max_leverage=constraints_config.get('max_leverage', 1.0),
        max_position=constraints_config.get('max_position', 0.4),
        min_position=constraints_config.get('min_position', 0.0),
        max_turnover_daily=constraints_config.get('max_turnover_daily', 0.2),
        position_bands=constraints_config.get('position_bands', {'min': 0.05, 'max': 0.35}),
        use_risk_budgeting=risk_config.get('use_risk_budgeting', True),
        covariance_window=risk_config.get('covariance_window', 60),
        spike_cooldown_days=risk_config.get('spike_cooldown_days', 5),
        cooldown_turnover_scale=risk_config.get('cooldown_turnover_scale', 0.5)
    )
