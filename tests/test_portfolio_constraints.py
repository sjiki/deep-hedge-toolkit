"""Tests for portfolio constraints."""

import pytest
import numpy as np
import pandas as pd

from env.portfolio_constraints import PortfolioConstraints, create_constraints_from_config


def test_portfolio_constraints_init():
    """Test initialization."""
    constraints = PortfolioConstraints(
        max_leverage=1.0,
        max_position=0.4,
        min_position=0.0,
        max_turnover_daily=0.2
    )
    
    assert constraints.max_leverage == 1.0
    assert constraints.max_position == 0.4


def test_apply_position_limits():
    """Test position limits."""
    constraints = PortfolioConstraints(
        max_position=0.4,
        min_position=0.0,
        position_bands={'min': 0.05, 'max': 0.35}
    )
    
    # Test clipping
    weights = np.array([0.6, 0.3, -0.1, 0.04])
    constrained = constraints.apply_position_limits(weights)
    
    # Check max limit
    assert all(constrained <= 0.35)
    # Check min limit (small positions zeroed)
    assert constrained[3] == 0.0
    # Check non-negative
    assert all(constrained >= 0.0)


def test_apply_leverage_constraint():
    """Test leverage constraint."""
    constraints = PortfolioConstraints(max_leverage=1.0)
    
    # Test over-leveraged weights
    weights = np.array([0.5, 0.4, 0.3])  # Sum = 1.2
    constrained = constraints.apply_leverage_constraint(weights)
    
    # Should scale down to sum = 1.0
    assert np.isclose(np.abs(constrained).sum(), 1.0)
    
    # Proportions should be maintained
    assert np.allclose(constrained / constrained.sum(), weights / weights.sum())


def test_apply_turnover_constraint():
    """Test turnover constraint."""
    constraints = PortfolioConstraints(max_turnover_daily=0.2)
    
    old_weights = np.array([0.3, 0.4, 0.3])
    new_weights = np.array([0.5, 0.2, 0.3])
    
    constrained = constraints.apply_turnover_constraint(old_weights, new_weights)
    
    # Check turnover
    turnover = np.abs(constrained - old_weights).sum()
    assert turnover <= 0.2 + 1e-6  # Small tolerance


def test_spike_cooldown():
    """Test spike cooldown mechanism."""
    constraints = PortfolioConstraints(
        spike_cooldown_days=5,
        cooldown_turnover_scale=0.5,
        max_turnover_daily=0.2
    )
    
    # Not in cooldown initially
    assert not constraints.is_in_cooldown()
    assert constraints.get_effective_turnover_limit() == 0.2
    
    # Trigger spike
    constraints.update_spike_cooldown(is_spike=True)
    assert constraints.is_in_cooldown()
    assert constraints.get_effective_turnover_limit() == 0.1  # Scaled
    
    # Wait out cooldown
    for _ in range(6):  # One more than cooldown days
        constraints.update_spike_cooldown(is_spike=False)
    
    assert not constraints.is_in_cooldown()
    assert constraints.get_effective_turnover_limit() == 0.2


def test_compute_risk_budget_weights(sample_price_data):
    """Test risk budgeting."""
    constraints = PortfolioConstraints(
        use_risk_budgeting=True,
        covariance_window=60
    )
    
    returns = sample_price_data.pct_change().dropna()
    
    weights = constraints.compute_risk_budget_weights(returns, equal_weight=True)
    
    assert len(weights) == len(sample_price_data.columns)
    assert all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_project_weights(sample_price_data):
    """Test full weight projection."""
    constraints = PortfolioConstraints(
        max_leverage=1.0,
        max_position=0.4,
        max_turnover_daily=0.2,
        use_risk_budgeting=False
    )
    
    n_assets = len(sample_price_data.columns)
    target_weights = np.array([0.5, 0.3, 0.15, 0.05])
    old_weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    returns = sample_price_data.pct_change().dropna()
    
    projected = constraints.project_weights(target_weights, old_weights, returns)
    
    # Check all constraints satisfied
    assert np.abs(projected).sum() <= constraints.max_leverage + 1e-6
    assert all(projected <= constraints.max_position)
    turnover = np.abs(projected - old_weights).sum()
    assert turnover <= constraints.max_turnover_daily + 1e-6


def test_create_constraints_from_config(sample_config):
    """Test factory function."""
    constraints = create_constraints_from_config(sample_config['environment'])
    
    assert isinstance(constraints, PortfolioConstraints)
    assert constraints.max_leverage == 1.0
    assert constraints.max_position == 0.4


def test_minimum_variance_weights(sample_price_data):
    """Test minimum variance portfolio."""
    constraints = PortfolioConstraints(
        use_risk_budgeting=True,
        covariance_window=60
    )
    
    returns = sample_price_data.pct_change().dropna()
    
    weights = constraints.compute_risk_budget_weights(returns, equal_weight=False)
    
    assert len(weights) == len(sample_price_data.columns)
    assert all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
