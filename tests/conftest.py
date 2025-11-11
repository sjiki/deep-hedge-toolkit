"""Pytest fixtures for VIX ETP RL tests."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import yaml


@pytest.fixture
def sample_dates():
    """Generate sample date range."""
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=200, freq='D')
    return dates


@pytest.fixture
def sample_price_data(sample_dates):
    """Generate synthetic price data."""
    n_dates = len(sample_dates)
    n_assets = 4
    
    # Generate random walk prices
    np.random.seed(42)
    returns = np.random.randn(n_dates, n_assets) * 0.02
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    tickers = ['UVXY', 'SVXY', 'VXX', 'VIXY']
    df = pd.DataFrame(prices, index=sample_dates, columns=tickers)
    
    return df


@pytest.fixture
def sample_volume_data(sample_dates):
    """Generate synthetic volume data."""
    n_dates = len(sample_dates)
    n_assets = 4
    
    # Generate random volumes
    np.random.seed(42)
    volumes = np.random.uniform(1e6, 5e6, size=(n_dates, n_assets))
    
    tickers = ['UVXY', 'SVXY', 'VXX', 'VIXY']
    df = pd.DataFrame(volumes, index=sample_dates, columns=tickers)
    
    return df


@pytest.fixture
def sample_vix_data(sample_dates):
    """Generate synthetic VIX data with diverse regimes."""
    n_dates = len(sample_dates)
    
    # Generate VIX with more variation and spikes
    np.random.seed(42)
    
    # Create different regime periods
    calm_period = np.full(40, 15.0) + np.random.randn(40) * 1.0
    buildup_period = np.linspace(18, 28, 40) + np.random.randn(40) * 1.5
    spike_period = np.full(40, 35.0) + np.random.randn(40) * 3.0
    revert_period = np.linspace(35, 20, 40) + np.random.randn(40) * 2.0
    calm2_period = np.full(40, 17.0) + np.random.randn(40) * 1.0
    
    vix = np.concatenate([calm_period, buildup_period, spike_period, revert_period, calm2_period])
    vix = vix[:n_dates]
    vix = np.clip(vix, 10, 50)
    
    df = pd.DataFrame({'close': vix}, index=sample_dates)
    return df


@pytest.fixture
def sample_regime_data(sample_dates):
    """Generate synthetic regime probability data."""
    n_dates = len(sample_dates)
    
    # Generate regime probabilities
    np.random.seed(42)
    probs = np.random.dirichlet(alpha=[1, 1, 1, 1], size=n_dates)
    
    regimes = ['Calm', 'BuildUp', 'Spike', 'MeanRevert']
    columns = [f'regime_proba_{r}' for r in regimes]
    
    df = pd.DataFrame(probs, index=sample_dates, columns=columns)
    return df


@pytest.fixture
def temp_csv_file(sample_dates):
    """Create temporary CSV file with price data."""
    n_dates = len(sample_dates)
    
    # Generate data
    np.random.seed(42)
    prices = 100 + np.random.randn(n_dates).cumsum()
    volumes = np.random.uniform(1e6, 5e6, size=n_dates)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': sample_dates,
        'close': prices,
        'volume': volumes
    })
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_config():
    """Generate sample configuration dictionary."""
    config = {
        'data': {
            'csv_sources': {
                'vix': None,
                'spy': None
            },
            'polygon': {
                'api_key': 'test_key',
                'tickers': ['UVXY', 'SVXY', 'VXX', 'VIXY'],
                'date_from': '2020-01-01',
                'date_to': None
            },
            'features': {
                'lookback_days': 60,
                'vol_window': 20,
                'trend_window': 10,
                'regime_window': 30
            }
        },
        'regime': {
            'model_type': 'lightgbm',
            'regimes': ['Calm', 'BuildUp', 'Spike', 'MeanRevert'],
            'labeling': {
                'calm_vix_threshold': 20,
                'buildup_vix_change': 2.0,
                'spike_vix_level': 30,
                'spike_vix_change': 5.0,
                'meanrevert_from_spike': 25
            },
            'lightgbm': {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.05,
                'num_leaves': 15
            }
        },
        'environment': {
            'execution': {
                'base_spread_bps': 10,
                'participation_impact': 0.001,
                'borrow_cost_annual': {
                    'UVXY': 0.05,
                    'SVXY': 0.03,
                    'VXX': 0.04,
                    'VIXY': 0.04
                }
            },
            'constraints': {
                'max_leverage': 1.0,
                'max_position': 0.4,
                'min_position': 0.0,
                'max_turnover_daily': 0.2,
                'position_bands': {
                    'min': 0.05,
                    'max': 0.35
                }
            },
            'risk': {
                'use_risk_budgeting': True,
                'covariance_window': 60,
                'risk_budget_equal_weight': True,
                'spike_cooldown_days': 5,
                'cooldown_turnover_scale': 0.5
            },
            'rebalancing': {
                'frequency_days': 1,
                'min_history_days': 60
            },
            'initial_cash': 100000.0
        },
        'agents': {
            'ppo': {
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'total_timesteps': 10000
            }
        },
        'training': {
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'reward_type': 'sharpe',
            'observation_window': 30,
            'seed': 42
        }
    }
    
    return config


@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary config YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
