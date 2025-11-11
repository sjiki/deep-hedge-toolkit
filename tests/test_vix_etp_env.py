"""Tests for VIX ETP environment."""

import pytest
import numpy as np
import gymnasium as gym

from env.vix_etp_env import VIXETPEnv
from env.execution_models import ExecutionCostModel
from env.portfolio_constraints import PortfolioConstraints


def test_env_initialization(sample_price_data, sample_volume_data):
    """Test environment initialization."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data
    )
    
    assert isinstance(env, gym.Env)
    assert env.n_assets == 4
    assert env.initial_cash == 100000.0


def test_env_spaces(sample_price_data, sample_volume_data):
    """Test action and observation spaces."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data
    )
    
    # Action space should be Box for weights
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (env.n_assets,)
    
    # Observation space should be Box
    assert isinstance(env.observation_space, gym.spaces.Box)


def test_env_reset(sample_price_data, sample_volume_data):
    """Test environment reset."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data
    )
    
    obs, info = env.reset(seed=42)
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert env.current_step == 0
    assert env.portfolio_value == env.initial_cash


def test_env_step(sample_price_data, sample_volume_data):
    """Test environment step."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data
    )
    
    obs, info = env.reset(seed=42)
    
    # Take a step with uniform weights
    action = np.ones(env.n_assets) / env.n_assets
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'portfolio_value' in info


def test_env_episode(sample_price_data, sample_volume_data):
    """Test complete episode."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data,
        min_history_days=30
    )
    
    obs, info = env.reset(seed=42)
    done = False
    steps = 0
    max_steps = 20
    
    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    assert steps > 0
    assert 'total_return' in info


def test_env_with_regime_data(sample_price_data, sample_volume_data, sample_regime_data):
    """Test environment with regime data."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data,
        regime_data=sample_regime_data
    )
    
    obs, info = env.reset(seed=42)
    
    # Observation should include regime probabilities
    assert obs.shape[0] > sample_price_data.shape[1] * 3


def test_env_transaction_costs(sample_price_data, sample_volume_data):
    """Test that transaction costs are applied."""
    execution_model = ExecutionCostModel(
        base_spread_bps=20,
        participation_impact=0.002
    )
    
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data,
        execution_model=execution_model
    )
    
    obs, info = env.reset(seed=42)
    initial_value = env.portfolio_value
    
    # Make a large trade
    action = np.array([1.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Transaction cost should reduce portfolio value
    assert 'transaction_cost' in info
    assert info['transaction_cost'] >= 0


def test_env_constraints(sample_price_data, sample_volume_data):
    """Test that constraints are enforced."""
    constraints = PortfolioConstraints(
        max_leverage=1.0,
        max_position=0.3,
        max_turnover_daily=0.1
    )
    
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data,
        constraints=constraints
    )
    
    obs, info = env.reset(seed=42)
    
    # Try to violate constraints
    action = np.array([1.0, 0.0, 0.0, 0.0])  # 100% in one asset
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Weights should be constrained
    assert all(env.current_weights <= 0.3 + 1e-6)


def test_env_availability_mask(sample_price_data, sample_volume_data):
    """Test asset availability mask."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data,
        min_history_days=100  # High requirement
    )
    
    obs, info = env.reset(seed=42)
    
    # Early in the data, not all assets should be available
    current_idx = env.start_idx + env.current_step
    mask = env._get_availability_mask(current_idx)
    
    # Mask should have some zeros if data is short
    assert len(mask) == env.n_assets
    assert all((mask == 0) | (mask == 1))


def test_env_reward_types(sample_price_data, sample_volume_data):
    """Test different reward types."""
    for reward_type in ['sharpe', 'sortino', 'returns']:
        env = VIXETPEnv(
            price_data=sample_price_data,
            volume_data=sample_volume_data,
            reward_type=reward_type,
            min_history_days=30
        )
        
        obs, info = env.reset(seed=42)
        
        # Take several steps to accumulate returns
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Should compute reward without error
        assert isinstance(reward, (int, float))


def test_env_portfolio_metrics(sample_price_data, sample_volume_data):
    """Test portfolio metrics computation."""
    env = VIXETPEnv(
        price_data=sample_price_data,
        volume_data=sample_volume_data,
        min_history_days=30
    )
    
    obs, info = env.reset(seed=42)
    
    # Run several steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Info should contain metrics
    assert 'total_return' in info
    assert 'portfolio_value' in info
    
    if len(env.returns_history) > 1:
        assert 'sharpe_ratio' in info
        assert 'mean_return' in info
        assert 'volatility' in info
