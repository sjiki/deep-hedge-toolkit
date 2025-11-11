"""Tests for execution cost models."""

import pytest
import numpy as np
import pandas as pd

from env.execution_models import ExecutionCostModel, DynamicSlippageModel, create_execution_model


def test_execution_cost_model_init():
    """Test initialization of execution cost model."""
    model = ExecutionCostModel(
        base_spread_bps=10,
        participation_impact=0.001,
        borrow_costs_annual={'TICKER': 0.05}
    )
    
    assert model.base_spread_bps == 10
    assert model.participation_impact == 0.001
    assert 'TICKER' in model.borrow_costs_annual


def test_compute_slippage():
    """Test slippage computation."""
    model = ExecutionCostModel(base_spread_bps=20, participation_impact=0.002)
    
    # Create test data
    trade_values = pd.Series([10000, 20000, 30000], index=['A', 'B', 'C'])
    prices = pd.Series([100, 200, 150], index=['A', 'B', 'C'])
    volumes = pd.Series([1e6, 2e6, 1.5e6], index=['A', 'B', 'C'])
    
    slippage = model.compute_slippage(trade_values, prices, volumes)
    
    assert len(slippage) == 3
    assert all(slippage >= 0)
    # Base spread should be at least half of 20 bps = 0.001
    assert all(slippage >= 0.001)


def test_compute_slippage_no_volumes():
    """Test slippage without volume data."""
    model = ExecutionCostModel(base_spread_bps=10, participation_impact=0.001)
    
    trade_values = pd.Series([10000, 20000], index=['A', 'B'])
    prices = pd.Series([100, 200], index=['A', 'B'])
    
    slippage = model.compute_slippage(trade_values, prices, volumes=None)
    
    assert len(slippage) == 2
    # Should only have base spread cost
    expected = 10 / 2 / 10000  # half spread in decimal
    assert np.allclose(slippage, expected)


def test_compute_borrow_cost():
    """Test borrow cost computation."""
    borrow_costs = {
        'A': 0.05,  # 5% annual
        'B': 0.03   # 3% annual
    }
    model = ExecutionCostModel(borrow_costs_annual=borrow_costs)
    
    # Create position data (weights)
    positions = pd.DataFrame([[0.5, 0.3]], columns=['A', 'B'])
    prices = pd.DataFrame([[100, 200]], columns=['A', 'B'])
    
    borrow_cost = model.compute_borrow_cost(positions, prices, days_held=1.0)
    
    assert len(borrow_cost) == 1
    assert borrow_cost.iloc[0] > 0
    
    # Check approximate value: (0.5 * 0.05 + 0.3 * 0.03) / 252
    expected = (0.5 * 0.05 + 0.3 * 0.03) / 252
    assert np.isclose(borrow_cost.iloc[0], expected, rtol=0.01)


def test_compute_total_transaction_cost():
    """Test total transaction cost computation."""
    model = ExecutionCostModel(base_spread_bps=10, participation_impact=0.001)
    
    old_positions = pd.Series([0.3, 0.4, 0.3], index=['A', 'B', 'C'])
    new_positions = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])
    prices = pd.Series([100, 200, 150], index=['A', 'B', 'C'])
    volumes = pd.Series([1e6, 2e6, 1.5e6], index=['A', 'B', 'C'])
    
    cost = model.compute_total_transaction_cost(
        old_positions, new_positions, prices, volumes, portfolio_value=100000
    )
    
    assert cost >= 0
    # Cost should be proportional to turnover
    turnover = np.abs(new_positions - old_positions).sum()
    assert cost < turnover * 0.01  # Should be less than 1% of turnover


def test_dynamic_slippage_model():
    """Test dynamic slippage model with volume threshold."""
    model = DynamicSlippageModel(
        base_spread_bps=10,
        participation_impact=0.001,
        volume_threshold=0.01
    )
    
    # Test with high participation rate
    trade_values = pd.Series([100000], index=['A'])  # Large trade
    prices = pd.Series([100], index=['A'])
    volumes = pd.Series([50000], index=['A'])  # Low volume
    
    slippage = model.compute_slippage(trade_values, prices, volumes)
    
    # Should have higher impact due to large participation
    assert slippage.iloc[0] > 0.0005  # More than just base spread


def test_create_execution_model():
    """Test factory function."""
    config = {
        'execution': {
            'base_spread_bps': 15,
            'participation_impact': 0.002,
            'borrow_cost_annual': {'TEST': 0.04}
        }
    }
    
    # Create basic model
    model = create_execution_model(config, model_type='basic')
    assert isinstance(model, ExecutionCostModel)
    assert model.base_spread_bps == 15
    
    # Create dynamic model
    model = create_execution_model(config, model_type='dynamic')
    assert isinstance(model, DynamicSlippageModel)
    assert model.base_spread_bps == 15


def test_zero_turnover():
    """Test that zero turnover results in zero cost."""
    model = ExecutionCostModel(base_spread_bps=10, participation_impact=0.001)
    
    # Same positions
    positions = pd.Series([0.3, 0.4, 0.3], index=['A', 'B', 'C'])
    prices = pd.Series([100, 200, 150], index=['A', 'B', 'C'])
    volumes = pd.Series([1e6, 2e6, 1.5e6], index=['A', 'B', 'C'])
    
    cost = model.compute_total_transaction_cost(
        positions, positions, prices, volumes, portfolio_value=100000
    )
    
    assert cost == 0.0
