"""Builder utility for constructing VIX ETP environment from configuration."""

import pandas as pd
from typing import Optional, Dict, Tuple
import yaml
from pathlib import Path

from data.vix_etp_data_loader import load_vix_etp_data
from models.regime_classifier import RegimeClassifier
from env.execution_models import create_execution_model
from env.portfolio_constraints import create_constraints_from_config
from env.vix_etp_env import VIXETPEnv


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_vix_etp_env(
    config_path: Optional[str] = None,
    config: Optional[Dict] = None,
    vix_csv_path: Optional[str] = None,
    spy_csv_path: Optional[str] = None
) -> Tuple[VIXETPEnv, pd.DataFrame, pd.DataFrame]:
    """
    Build VIX ETP environment from configuration.
    
    Args:
        config_path: Path to config YAML file (optional if config is provided)
        config: Configuration dictionary (optional if config_path is provided)
        vix_csv_path: Path to VIX CSV file (overrides config)
        spy_csv_path: Path to SPY CSV file (overrides config)
        
    Returns:
        Tuple of (environment, price_data, volume_data)
    """
    # Load config
    if config is None:
        if config_path is None:
            raise ValueError("Either config_path or config must be provided")
        config = load_config(config_path)
    
    # Data configuration
    data_config = config.get('data', {})
    
    # CSV sources
    csv_sources = {}
    if vix_csv_path:
        csv_sources['VIX'] = vix_csv_path
    elif data_config.get('csv_sources', {}).get('vix'):
        csv_sources['VIX'] = data_config['csv_sources']['vix']
    
    if spy_csv_path:
        csv_sources['SPY'] = spy_csv_path
    elif data_config.get('csv_sources', {}).get('spy'):
        csv_sources['SPY'] = data_config['csv_sources']['spy']
    
    # Polygon configuration
    polygon_config = data_config.get('polygon', {})
    tickers = polygon_config.get('tickers', ['UVXY', 'SVXY', 'VXX', 'VIXY'])
    date_from = polygon_config.get('date_from', '2020-01-01')
    date_to = polygon_config.get('date_to')
    
    # Load data
    print("Loading price and volume data...")
    price_data, volume_data = load_vix_etp_data(
        tickers=tickers,
        date_from=date_from,
        date_to=date_to,
        csv_sources=csv_sources if csv_sources else None
    )
    
    if price_data.empty:
        raise ValueError("Failed to load any price data")
    
    print(f"Loaded {len(price_data)} days of data for {len(price_data.columns)} assets")
    
    # Build regime classifier if VIX data available
    regime_data = None
    regime_config = config.get('regime', {})
    
    if 'VIX' in csv_sources or 'VIX' in price_data.columns:
        print("Building regime classifier...")
        
        # Get VIX data
        if 'VIX' in price_data.columns:
            vix_data = price_data[['VIX']].rename(columns={'VIX': 'close'})
        else:
            # Load VIX separately if not in main data
            from data.csv_loader import CSVLoader
            vix_loader = CSVLoader(csv_sources['VIX'])
            vix_data = vix_loader.load()
        
        # Create classifier
        classifier = RegimeClassifier(
            model_type=regime_config.get('model_type', 'lightgbm'),
            labeling_params=regime_config.get('labeling'),
            model_params=regime_config.get(regime_config.get('model_type', 'lightgbm'))
        )
        
        # Create features
        feature_config = data_config.get('features', {})
        features = classifier.create_features(
            vix_data,
            window=feature_config.get('regime_window', 30)
        )
        
        # Fit and predict
        _, regime_proba = classifier.fit_predict(features, vix_data=vix_data)
        regime_data = regime_proba
        
        print(f"Regime classification complete. Shape: {regime_data.shape}")
    
    # Create execution model
    print("Creating execution model...")
    execution_model = create_execution_model(
        config.get('environment', {}),
        model_type='dynamic'
    )
    
    # Create constraints
    print("Creating portfolio constraints...")
    constraints = create_constraints_from_config(config.get('environment', {}))
    
    # Environment parameters
    env_config = config.get('environment', {})
    training_config = config.get('training', {})
    
    initial_cash = env_config.get('initial_cash', 100000.0)
    rebalance_frequency = env_config.get('rebalancing', {}).get('frequency_days', 1)
    min_history_days = env_config.get('rebalancing', {}).get('min_history_days', 60)
    observation_window = training_config.get('observation_window', 30)
    reward_type = training_config.get('reward_type', 'sharpe')
    
    # Create environment
    print("Creating VIX ETP environment...")
    env = VIXETPEnv(
        price_data=price_data,
        volume_data=volume_data,
        regime_data=regime_data,
        execution_model=execution_model,
        constraints=constraints,
        initial_cash=initial_cash,
        rebalance_frequency=rebalance_frequency,
        min_history_days=min_history_days,
        observation_window=observation_window,
        reward_type=reward_type
    )
    
    print("Environment created successfully!")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Max episodes steps: {env.max_steps}")
    
    return env, price_data, volume_data


def build_env_from_default_config() -> VIXETPEnv:
    """
    Build environment using default configuration.
    
    Returns:
        VIXETPEnv instance
    """
    # Find config file
    config_paths = [
        'config/vix_etp_config.yaml',
        '../config/vix_etp_config.yaml',
        '../../config/vix_etp_config.yaml'
    ]
    
    config_path = None
    for path in config_paths:
        if Path(path).exists():
            config_path = path
            break
    
    if config_path is None:
        raise FileNotFoundError("Could not find config/vix_etp_config.yaml")
    
    env, _, _ = build_vix_etp_env(config_path=config_path)
    return env


if __name__ == "__main__":
    # Example usage
    env = build_env_from_default_config()
    
    # Test environment
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nAfter step:")
    print(f"  Reward: {reward:.4f}")
    print(f"  Portfolio value: ${info['portfolio_value']:.2f}")
    print(f"  Terminated: {terminated}")
