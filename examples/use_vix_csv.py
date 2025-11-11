"""Example: Using CSV data for VIX ETP trading.

This example demonstrates how to use custom CSV files for VIX and SPY data
instead of relying on API data sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from data.csv_loader import CSVLoader
from env.builder import build_vix_etp_env


def create_sample_csv_data(output_dir='data_samples'):
    """Create sample CSV files for demonstration."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate sample dates
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n = len(dates)
    
    # Create VIX data (with characteristic spikes)
    np.random.seed(42)
    vix_base = 20 + np.random.randn(n).cumsum() * 0.5
    vix_base = np.clip(vix_base, 12, 45)
    
    # Add some spikes
    spike_dates = [300, 600, 900, 1200]
    for spike_idx in spike_dates:
        if spike_idx < n:
            vix_base[spike_idx:spike_idx+10] += 15
            vix_base = np.clip(vix_base, 12, 60)
    
    vix_df = pd.DataFrame({
        'date': dates,
        'close': vix_base,
        'volume': 0  # VIX doesn't have volume
    })
    
    vix_path = output_path / 'vix.csv'
    vix_df.to_csv(vix_path, index=False)
    print(f"Created VIX data: {vix_path}")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  VIX range: {vix_base.min():.2f} to {vix_base.max():.2f}")
    
    # Create SPY data (as VIX proxy/reference)
    spy_returns = np.random.randn(n) * 0.01
    spy_prices = 300 * np.exp(np.cumsum(spy_returns))
    spy_volume = np.random.uniform(50e6, 150e6, size=n)
    
    spy_df = pd.DataFrame({
        'date': dates,
        'close': spy_prices,
        'volume': spy_volume
    })
    
    spy_path = output_path / 'spy.csv'
    spy_df.to_csv(spy_path, index=False)
    print(f"\nCreated SPY data: {spy_path}")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  SPY range: ${spy_prices.min():.2f} to ${spy_prices.max():.2f}")
    
    return str(vix_path), str(spy_path)


def load_and_validate_csv(csv_path):
    """Load and validate CSV data."""
    print(f"\nValidating: {csv_path}")
    
    loader = CSVLoader(csv_path)
    df = loader.load()
    
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isna().sum().to_dict()}")
    
    return df


def build_environment_with_csv(vix_path, spy_path):
    """Build VIX ETP environment using CSV data."""
    print("\n" + "="*60)
    print("Building VIX ETP Environment with CSV Data")
    print("="*60)
    
    # Create temporary config with CSV paths
    config = {
        'data': {
            'csv_sources': {
                'vix': vix_path,
                'spy': spy_path
            },
            'polygon': {
                'tickers': ['UVXY', 'SVXY', 'VXX', 'VIXY'],
                'date_from': '2020-01-01'
            },
            'features': {
                'lookback_days': 60,
                'regime_window': 30
            }
        },
        'regime': {
            'model_type': 'lightgbm',
            'labeling': {
                'calm_vix_threshold': 20,
                'spike_vix_level': 30,
                'spike_vix_change': 5.0
            },
            'lightgbm': {
                'n_estimators': 50,
                'max_depth': 3
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
                'max_turnover_daily': 0.2,
                'position_bands': {'min': 0.05, 'max': 0.35}
            },
            'risk': {
                'use_risk_budgeting': True,
                'covariance_window': 60,
                'spike_cooldown_days': 5
            },
            'rebalancing': {
                'frequency_days': 1,
                'min_history_days': 60
            },
            'initial_cash': 100000.0
        },
        'agents': {
            'ppo': {
                'total_timesteps': 10000
            }
        },
        'training': {
            'reward_type': 'sharpe',
            'observation_window': 30,
            'seed': 42
        }
    }
    
    # Build environment
    # Note: In practice, you'd use build_vix_etp_env with csv_paths
    # For this example, we'll demonstrate the data loading
    
    print("\nCSV Data Sources:")
    print(f"  VIX: {vix_path}")
    print(f"  SPY: {spy_path}")
    
    # Load VIX data
    vix_df = load_and_validate_csv(vix_path)
    
    # Load SPY data
    spy_df = load_and_validate_csv(spy_path)
    
    print("\n" + "="*60)
    print("CSV Data Loaded Successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update config/vix_etp_config.yaml with these CSV paths")
    print("2. Run: python scripts/cli.py train --agent ppo")
    print("3. The environment will automatically use CSV data as priority source")
    
    return vix_df, spy_df


def main():
    """Main example workflow."""
    print("VIX ETP CSV Data Usage Example")
    print("=" * 60)
    
    # Step 1: Create sample CSV files
    print("\nStep 1: Creating sample CSV files...")
    vix_path, spy_path = create_sample_csv_data()
    
    # Step 2: Load and validate
    print("\nStep 2: Loading and validating CSV data...")
    vix_df, spy_df = build_environment_with_csv(vix_path, spy_path)
    
    # Step 3: Show some statistics
    print("\nVIX Statistics:")
    print(vix_df['close'].describe())
    
    print("\nSPY Statistics:")
    print(spy_df['close'].describe())
    
    # Step 4: Usage instructions
    print("\n" + "="*60)
    print("Usage with Training Pipeline")
    print("="*60)
    print("""
To use these CSV files for training:

1. Edit config/vix_etp_config.yaml:
   
   data:
     csv_sources:
       vix: data_samples/vix.csv
       spy: data_samples/spy.csv

2. Train agent:
   
   python scripts/cli.py train --agent ppo --timesteps 100000

3. The data loader will:
   - Try CSV files first (highest priority)
   - Fall back to Polygon API if CSV not found
   - Fall back to yfinance as last resort

4. For regime classification, VIX CSV is required

Advantages of CSV approach:
- Full control over data quality
- Reproducible experiments
- No API dependencies
- Works offline
- Can include proprietary indicators
    """)


if __name__ == "__main__":
    main()
