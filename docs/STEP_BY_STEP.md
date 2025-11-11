# VIX ETP Reinforcement Learning Module - Step-by-Step Guide

This guide walks you through the complete process of using the VIX ETP RL module for Sharpe Ratio optimization.

## Table of Contents

1. [Installation](#installation)
2. [Data Acquisition](#data-acquisition)
3. [Training Agents](#training-agents)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Walk-Forward Validation](#walk-forward-validation)
6. [Performance Reporting](#performance-reporting)
7. [CSV Data Usage](#csv-data-usage)
8. [Advanced Configuration](#advanced-configuration)

## Installation

### Prerequisites

- Python 3.10 or 3.11
- pip package manager

### Install Dependencies

```bash
cd deep-hedge-toolkit
pip install -r requirements.txt
```

### Environment Setup

For Polygon.io API access (optional but recommended):

```bash
export POLYGON_API_KEY=your_api_key_here
```

## Data Acquisition

The module supports three data sources with automatic fallback:

1. **CSV files** (highest priority)
2. **Polygon.io API** (if API key provided)
3. **yfinance** (free fallback)

### Using CSV Data

Prepare CSV files with columns: `date`, `close`, `volume`

```bash
# Example CSV structure:
# date,close,volume
# 2020-01-01,100.5,1500000
# 2020-01-02,101.2,1600000
```

Update `config/vix_etp_config.yaml`:

```yaml
data:
  csv_sources:
    vix: path/to/vix.csv
    spy: path/to/spy.csv  # Optional
```

### Using API Data

With Polygon.io API key set:

```yaml
data:
  polygon:
    api_key: ${POLYGON_API_KEY}
    tickers:
      - UVXY
      - SVXY
      - VXX
      - VIXY
    date_from: "2020-01-01"
```

## Training Agents

### Quick Start - Train PPO Agent

```bash
python scripts/cli.py train --agent ppo --timesteps 100000
```

### Train SAC Agent

```bash
python scripts/cli.py train --agent sac --timesteps 100000
```

### Custom Configuration

```bash
python scripts/cli.py train \
  --agent ppo \
  --timesteps 200000 \
  --config config/vix_etp_config.yaml \
  --output outputs/my_training \
  --seed 42
```

### Output

Training produces:
- `outputs/vix_etp_ppo_model.zip` - Trained model
- `outputs/training_metrics_ppo.json` - Training metrics

## Hyperparameter Tuning

Use Optuna to optimize hyperparameters:

```bash
python scripts/cli.py tune --trials 50 --timesteps 50000
```

### Tuning Configuration

Edit `config/vix_etp_config.yaml` to specify parameters to tune:

```yaml
optuna:
  n_trials: 50
  timeout_seconds: 3600
  n_jobs: 1
  
  tune_params:
    learning_rate: [0.0001, 0.001]
    clip_range: [0.1, 0.3]
    ent_coef: [0.0, 0.1]
    gamma: [0.95, 0.99]
    max_turnover_daily: [0.1, 0.3]
    spike_cooldown_days: [3, 10]
```

### Output

- `outputs/optuna_best_params.json` - Best parameters and trial results

### Apply Best Parameters

After tuning, update your config with the best parameters found in `optuna_best_params.json`.

## Walk-Forward Validation

Perform rolling walk-forward validation:

```bash
python scripts/cli.py walkforward \
  --agent ppo \
  --train-window 365 \
  --test-window 90 \
  --step 30 \
  --timesteps 50000
```

### Parameters

- `--train-window`: Training window size in days (default: 365)
- `--test-window`: Test window size in days (default: 90)
- `--step`: Step size between folds in days (default: 30)
- `--timesteps`: Training timesteps per fold (default: 50000)

### Output

- `outputs/rolling_walkforward.json` - Complete validation results with metrics:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio (proxy via max drawdown)
  - Expected Shortfall (ES95)
  - Hit Ratio
  - Max Drawdown

## Performance Reporting

Generate visual and text reports:

```bash
python scripts/cli.py report
```

### Text Summary Only

```bash
python scripts/cli.py report --text-only
```

### Custom Output

```bash
python scripts/cli.py report \
  --results-dir outputs \
  --output my_report.png
```

### Report Contents

- **Sharpe Ratio by Fold**: Bar chart showing performance across folds
- **Returns Distribution**: Histogram of returns across all folds
- **Risk-Return Profile**: Scatter plot with Sharpe ratio coloring
- **Metrics Summary**: Text summary of key performance metrics

## CSV Data Usage

### Example: Using Custom VIX Data

1. **Prepare CSV file** (`data/my_vix.csv`):
   ```csv
   date,close,volume
   2020-01-01,15.2,0
   2020-01-02,15.5,0
   ...
   ```

2. **Update config**:
   ```yaml
   data:
     csv_sources:
       vix: data/my_vix.csv
   ```

3. **Train with CSV data**:
   ```bash
   python scripts/cli.py train --agent ppo --timesteps 100000
   ```

### Flexible Column Mapping

For CSVs with different column names, use the CSV loader directly:

```python
from data import CSVLoader

loader = CSVLoader(
    'my_data.csv',
    date_column='Date',
    price_column='Close',
    volume_column='Vol'
)
data = loader.load()
```

## Advanced Configuration

### Environment Parameters

Edit `config/vix_etp_config.yaml`:

```yaml
environment:
  execution:
    base_spread_bps: 10  # Spread in basis points
    participation_impact: 0.001  # Market impact coefficient
    borrow_cost_annual:
      UVXY: 0.05  # 5% annual borrow cost
  
  constraints:
    max_leverage: 1.0  # Long-only
    max_position: 0.4  # Max 40% per asset
    max_turnover_daily: 0.2  # Max 20% daily turnover
    position_bands:
      min: 0.05  # Min position if held
      max: 0.35  # Max position limit
  
  risk:
    use_risk_budgeting: true
    covariance_window: 60
    spike_cooldown_days: 5
    cooldown_turnover_scale: 0.5  # Reduce turnover during cooldown
```

### Regime Classification

Configure regime detection:

```yaml
regime:
  model_type: lightgbm  # or 'logistic'
  labeling:
    calm_vix_threshold: 20
    spike_vix_level: 30
    spike_vix_change: 5.0
```

### Agent Hyperparameters

PPO configuration:

```yaml
agents:
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    clip_range: 0.2
    ent_coef: 0.01
```

SAC configuration:

```yaml
agents:
  sac:
    learning_rate: 0.0003
    buffer_size: 100000
    batch_size: 256
    tau: 0.005
    gamma: 0.99
```

## Evaluation Workflow

Complete workflow example:

```bash
# 1. Train initial model
python scripts/cli.py train --agent ppo --timesteps 100000

# 2. Evaluate trained model
python scripts/cli.py eval \
  --model outputs/vix_etp_ppo_model.zip \
  --agent ppo \
  --episodes 20

# 3. Tune hyperparameters
python scripts/cli.py tune --trials 50 --timesteps 50000

# 4. Retrain with best parameters (update config first)
python scripts/cli.py train --agent ppo --timesteps 200000

# 5. Run walk-forward validation
python scripts/cli.py walkforward --agent ppo --timesteps 50000

# 6. Generate report
python scripts/cli.py report
```

## Environment Constraints in Action

The environment enforces:

1. **Long-only constraint**: No short positions
2. **Leverage cap**: Total position <= 100% (or configured max)
3. **Position limits**: Individual positions between min and max bands
4. **Turnover cap**: Daily rebalancing limited to configured percentage
5. **Spike cooldown**: Reduced trading activity after VIX spikes
6. **Risk budgeting**: Covariance-based position sizing (optional)
7. **Availability mask**: Assets require minimum history before trading

## Metrics Explanation

### Sharpe Ratio
Risk-adjusted return: `mean_return / std_return * sqrt(252)`

Higher is better. Typical target: > 1.0

### Sortino Ratio
Downside risk-adjusted return using only negative returns

Better measure for asymmetric return distributions

### Calmar Ratio
Return / Max Drawdown

Measures return per unit of worst-case drawdown

### Expected Shortfall (ES95)
Average loss in worst 5% of cases

More conservative than Value-at-Risk

### Hit Ratio
Percentage of positive return periods

Target: > 0.5 for profitable strategy

## Troubleshooting

### Issue: "Failed to load any price data"

**Solution**: 
- Check internet connection for API access
- Verify API key: `echo $POLYGON_API_KEY`
- Use CSV files as alternative
- Check CSV file paths in config

### Issue: Low Sharpe Ratio

**Solutions**:
- Run hyperparameter tuning
- Increase training timesteps
- Adjust reward function (try 'sortino')
- Review constraint parameters (turnover, position limits)

### Issue: Tests Failing

```bash
# Run tests with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_csv_loader.py -v
```

## Best Practices

1. **Start with CSV data** for reproducibility
2. **Tune hyperparameters** before production training
3. **Use walk-forward validation** to assess generalization
4. **Monitor spike cooldown** effectiveness
5. **Review transaction costs** impact on performance
6. **Validate regime classification** quality
7. **Test on out-of-sample data** before live trading

## Next Steps

- Review generated reports
- Experiment with different reward functions
- Adjust execution cost parameters
- Implement custom risk constraints
- Extend to additional VIX ETPs
- Add custom features to observation space

## Support

For issues or questions:
- Open a GitHub issue
- Review test files for usage examples
- Check configuration schema in `config/vix_etp_config.yaml`
