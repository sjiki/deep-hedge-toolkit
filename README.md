# Deep Hedge Strategy Toolkit

A comprehensive suite of tools for implementing and managing multi-layered portfolio hedging strategies, including advanced VIX ETP trading using Reinforcement Learning.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

---

## 📦 What's Included

### 1. **VIX ETP Reinforcement Learning Module** (NEW!)
Advanced machine learning system for VIX ETP portfolio optimization:
- PPO and SAC reinforcement learning agents
- Regime-aware trading with VIX spike detection
- Dynamic execution cost modeling
- Risk budgeting and portfolio constraints
- Hyperparameter tuning with Optuna
- Rolling walk-forward validation
- Comprehensive performance reporting

**Key Features:**
- ✓ Multi-asset VIX ETP trading (UVXY, SVXY, VXX, VIXY)
- ✓ Regime classification (Calm, BuildUp, Spike, MeanRevert)
- ✓ Transaction cost modeling (spread + market impact)
- ✓ Portfolio constraints (leverage, turnover, position limits)
- ✓ CSV data support with flexible column mapping
- ✓ Automated testing (43+ unit tests)
- ✓ CLI interface for all operations

**Quick Start:**
```bash
# Train a PPO agent
python -m scripts.cli train --agent ppo --timesteps 100000

# Run hyperparameter tuning
python -m scripts.cli tune --trials 50

# Perform walk-forward validation
python -m scripts.cli walkforward --agent ppo

# Generate performance report
python -m scripts.cli report

# Or use Makefile
make train
make tune
make walkforward
make report
```

See [docs/STEP_BY_STEP.md](docs/STEP_BY_STEP.md) for detailed guide.

### 2. **Hedge Calculator** (`hedge_calculator.py`)
Python-based calculator for designing optimal hedge strategies:
- Black-Scholes pricing for puts and calls
- Multi-layer cost calculations
- Scenario analysis (market declines from -2% to -40%)
- Optimal hedge ratio finder
- Excel report generation

**Features:**
- ✓ Calculates costs for collar, put spread, and far OTM strategies
- ✓ Determines optimal coverage ratio to meet risk/cost targets
- ✓ Generates comprehensive Excel reports

### 2. **Backtesting Tool** (`backtest_hedge_strategy.py`)
Historical performance analysis:
- Tests against 8 major market crashes (1987-2022)
- Monte Carlo simulation (10,000+ paths)
- Sensitivity analysis (volatility × market decline)
- Performance metrics and reporting

**Historical Scenarios Included:**
- 1987 Black Monday (-22%)
- 2000 Dot-Com Crash (-49%)
- 2008 Financial Crisis (-57%)
- 2020 COVID Crash (-34%)
- And more...

### 3. **Implementation Workflow** (`deep_hedge_workflow.md`)
Step-by-step guide with:
- 6 phases from assessment to crisis management
- Weekly/monthly/quarterly checklists
- Rebalancing triggers and rules
- Crisis response protocols
- Tax and regulatory considerations

### 4. **Spreadsheet Templates** (CSV format)
Ready-to-use tracking templates:
- `hedge_tracking_template.csv` - Position tracking with Greeks
- `monthly_performance_template.csv` - Monthly P&L analysis
- `rebalancing_checklist.csv` - Action tracking
- `scenario_analysis_template.csv` - Pre-built stress tests

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Microsoft Excel or LibreOffice (for viewing generated reports)

### Installation

```bash
# Clone the repository
git clone https://github.com/sjiki/deep-hedge-toolkit.git
cd deep-hedge-toolkit

# Install required packages
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from hedge_calculator import DeepHedgeCalculator; print('✓ Installation successful')"
```

### Basic Usage

**Option 1: Direct Calculation**
```python
from hedge_calculator import DeepHedgeCalculator

# Initialize with your portfolio
calculator = DeepHedgeCalculator(
    portfolio_value=10_000_000,
    risk_free_rate=0.045,
    annual_volatility=0.18
)

# Get layer costs
costs = calculator.calculate_layer_costs()
print(costs)

# Run scenario analysis
scenarios = calculator.scenario_analysis()
print(scenarios)

# Find optimal hedge ratio
optimal = calculator.optimal_hedge_ratio(
    target_max_loss_pct=15,
    max_hedge_cost_pct=2.5
)
print(optimal)

# Generate Excel report
calculator.generate_report('my_hedge_analysis.xlsx')
```

**Option 2: Backtesting**
```bash
python backtest_hedge_strategy.py
```
This will run the full backtesting suite and generate a comprehensive Excel report.

---

## 📊 Example Output

### Hedge Layer Costs (Example: $10M Portfolio)

| Layer | Strategy | Duration | Annual Cost | Protection Level |
|-------|----------|----------|-------------|------------------|
| Layer 1 | Collar (98 Put / 102 Call) | 90 days | $20,000 | 0-2% decline |
| Layer 2 | Put Spread (95/85) | 180 days | $60,000 | 5-15% decline |
| Layer 3 | Far OTM Puts (80 strike) | 365 days | $40,000 | 20%+ decline |
| **TOTAL** | | | **$120,000** | **Multi-layer** |

**Total Cost:** 1.2% of portfolio annually

### Scenario Analysis Results

| Market Decline | Unhedged Loss | With Hedges | Protection % |
|----------------|---------------|-------------|--------------|
| -5% | -$500,000 | -$312,500 | 62.5% |
| -10% | -$1,000,000 | -$437,500 | 71.3% |
| -20% | -$2,000,000 | -$943,750 | 52.8% |
| -30% | -$3,000,000 | -$1,443,750 | 51.9% |

---

## 🛠️ Advanced Features

### Optimal Hedge Ratio Finder
Automatically determines the optimal coverage percentage to meet your:
- Maximum acceptable loss threshold (e.g., 15% in a -30% crash)
- Maximum hedge cost budget (e.g., 2.5% annually)

### Monte Carlo Simulation
- Runs 10,000+ random market scenarios
- Tests hedge performance across various conditions
- Calculates downside protection statistics
- Compares hedged vs. unhedged distributions

### Sensitivity Analysis
Tests how hedge performance varies with:
- Different volatility levels (12% to 30%)
- Various market declines (-5% to -40%)
- Generates heat maps for visualization

---

## 📈 Workflow Overview

### Phase 1: Portfolio Assessment (Week 1)
- Analyze portfolio composition and beta
- Define risk tolerance
- Set up trading accounts

### Phase 2: Strategy Design (Week 2)
- Calculate hedge ratios
- Select specific instruments and strikes
- Run cost-benefit analysis
- Create execution plan

### Phase 3: Execution (Week 3)
- Execute layer by layer
- Set up monitoring dashboard
- Implement rebalancing triggers

### Phase 4: Ongoing Management
- **Daily:** Monitor key metrics
- **Weekly:** Check expiries and decay
- **Monthly:** Performance review, position rolling
- **Quarterly:** Strategy optimization
- **Annually:** Full performance analysis

---

## 📋 Tracking Templates

### Position Tracking
Includes all critical metrics:
- Strike, expiry, contracts
- Entry price, current price
- Greeks (Delta, Gamma, Theta, Vega)
- Unrealized P&L
- Protection level

### Performance Tracking
Monthly analysis:
- Portfolio return vs. market
- Hedge costs and payoffs
- Net return after hedging
- Sharpe ratio, alpha, beta
- VIX levels

### Rebalancing Checklist
Action tracking for:
- Monthly reviews
- Position rolls
- Trigger events (VIX spikes, portfolio growth)
- Cost impacts

---

## 🎯 Key Metrics Explained

### Protection Effectiveness
```
Protection % = (Avoided Loss / Total Loss) × 100

Example:
Market down 20% = -$2M unhedged loss
With hedges = -$944K actual loss
Protection = ($2M - $944K) / $2M = 52.8%
```

### Return on Hedge Spend
```
Hedge ROI = Protection Amount / Hedge Cost

Example:
Protection provided: $600K
Hedge cost: $150K
ROI = $600K / $150K = 4.0x
```

### Sharpe Ratio (Hedged vs. Unhedged)
Measures risk-adjusted returns. Higher is better.
Hedged portfolios typically show improved Sharpe ratios due to reduced downside volatility.

---

## ⚠️ Important Considerations

### Tax Implications
- Section 1256 contracts (index options) get 60/40 tax treatment
- Consult tax advisor for your situation
- Track wash sales carefully

### Regulatory
- Ensure proper options approval level (typically Level 3+)
- Understand margin requirements
- Be aware of position limits

### Risk Warnings
- Hedges cost money (reduce returns in up markets)
- No hedge is perfect (basis risk exists)
- Requires active management and monitoring
- Options can expire worthless

---

## 📚 Additional Resources

### Package Structure

**Core Hedging Tools:**
1. `hedge_calculator.py` - Hedge strategy calculator
2. `backtest_hedge_strategy.py` - Historical backtesting
3. `deep_hedge_workflow.md` - Implementation workflow
4. CSV templates for tracking and analysis

**VIX ETP RL Module:**
- `data/` - Data loading (CSV, Polygon, yfinance)
- `models/` - Regime classification
- `env/` - Trading environment & constraints
- `agents/` - PPO and SAC agent wrappers
- `scripts/` - Training pipelines and CLI
- `tests/` - Comprehensive test suite (43+ tests)
- `config/` - Configuration files
- `docs/` - Step-by-step guides
- `examples/` - Usage examples

**Key Files:**
- `scripts/cli.py` - Unified command-line interface
- `config/vix_etp_config.yaml` - Configuration
- `docs/STEP_BY_STEP.md` - Detailed guide
- `Makefile` - Convenience commands
- `.github/workflows/tests.yml` - CI/CD pipeline

### Learning Resources
- [CBOE Education Portal](https://www.cboe.com/education)
- [Options Industry Council](https://www.optionseducation.org)
- CFA Institute hedging frameworks

---

## 🤝 Support

For questions or issues:
1. Review the workflow guide ([deep_hedge_workflow.md](deep_hedge_workflow.md))
2. Check example outputs in generated Excel reports
3. [Open an issue](https://github.com/sjiki/deep-hedge-toolkit/issues) on GitHub
4. Consult with a financial advisor for personalized advice

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License & Disclaimer

**LICENSE:** MIT License - See [LICENSE](LICENSE) file for details.

**DISCLAIMER:** This toolkit is for educational and informational purposes only.
It is NOT financial advice. Always consult with qualified financial professionals
before implementing any hedging strategy. Past performance does not guarantee
future results. Options trading involves substantial risk of loss.

---

## 🔄 Version History

**Version 1.0** (2025-02-01)
- Initial release
- Multi-layer hedge calculator
- Historical backtesting (8 scenarios)
- Monte Carlo simulation
- Complete workflow documentation
- Excel report generation

---

**Built for portfolio managers, traders, and institutional investors seeking
professional-grade hedging tools and frameworks.**
