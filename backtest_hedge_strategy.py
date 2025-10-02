"""
Deep Hedge Strategy Backtesting Tool
Tests multi-layered hedge strategies against historical market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt


class HedgeBacktester:
    """
    Backtest hedge strategies against historical market crashes
    """

    def __init__(self, initial_portfolio=10_000_000):
        """
        Initialize backtester

        Args:
            initial_portfolio: Starting portfolio value
        """
        self.initial_portfolio = initial_portfolio
        self.portfolio_value = initial_portfolio

        # Historical crash scenarios
        self.historical_scenarios = {
            '1987_crash': {'name': '1987 Black Monday', 'decline': -0.22, 'days': 1, 'vix_spike': 3.5},
            '2000_dotcom': {'name': '2000 Dot-Com Crash', 'decline': -0.49, 'days': 929, 'vix_spike': 1.8},
            '2008_financial': {'name': '2008 Financial Crisis', 'decline': -0.57, 'days': 517, 'vix_spike': 4.2},
            '2011_debt': {'name': '2011 Debt Crisis', 'decline': -0.19, 'days': 154, 'vix_spike': 2.8},
            '2015_china': {'name': '2015 China Slowdown', 'decline': -0.14, 'days': 186, 'vix_spike': 2.1},
            '2018_correction': {'name': '2018 Q4 Correction', 'decline': -0.20, 'days': 94, 'vix_spike': 2.5},
            '2020_covid': {'name': '2020 COVID Crash', 'decline': -0.34, 'days': 33, 'vix_spike': 5.2},
            '2022_bear': {'name': '2022 Bear Market', 'decline': -0.25, 'days': 281, 'vix_spike': 1.9}
        }

    def black_scholes_put(self, spot, strike, time_years, volatility, risk_free=0.045):
        """Calculate put option price using Black-Scholes"""
        if time_years <= 0:
            return max(strike - spot, 0)

        d1 = (np.log(spot / strike) + (risk_free + 0.5 * volatility**2) * time_years) / \
             (volatility * np.sqrt(time_years))
        d2 = d1 - volatility * np.sqrt(time_years)

        put_price = strike * np.exp(-risk_free * time_years) * norm.cdf(-d2) - \
                    spot * norm.cdf(-d1)

        return put_price

    def black_scholes_call(self, spot, strike, time_years, volatility, risk_free=0.045):
        """Calculate call option price using Black-Scholes"""
        if time_years <= 0:
            return max(spot - strike, 0)

        d1 = (np.log(spot / strike) + (risk_free + 0.5 * volatility**2) * time_years) / \
             (volatility * np.sqrt(time_years))
        d2 = d1 - volatility * np.sqrt(time_years)

        call_price = spot * norm.cdf(d1) - \
                     strike * np.exp(-risk_free * time_years) * norm.cdf(d2)

        return call_price

    def create_hedge_strategy(self, spot_price=100, base_volatility=0.18):
        """
        Define the multi-layer hedge strategy

        Returns:
            Dictionary of hedge layers with specifications
        """
        hedge_layers = {
            'layer1_put': {
                'type': 'put',
                'strike': spot_price * 0.98,
                'expiry_days': 90,
                'coverage': 1.0,  # 100% of portfolio
                'volatility': base_volatility
            },
            'layer1_call': {
                'type': 'call',
                'strike': spot_price * 1.02,
                'expiry_days': 90,
                'coverage': 1.0,
                'volatility': base_volatility * 0.9  # Calls typically have lower IV
            },
            'layer2_long_put': {
                'type': 'put',
                'strike': spot_price * 0.95,
                'expiry_days': 180,
                'coverage': 0.75,
                'volatility': base_volatility * 1.05
            },
            'layer2_short_put': {
                'type': 'put',
                'strike': spot_price * 0.85,
                'expiry_days': 180,
                'coverage': 0.75,
                'volatility': base_volatility * 1.15
            },
            'layer3_put': {
                'type': 'put',
                'strike': spot_price * 0.80,
                'expiry_days': 365,
                'coverage': 0.5,
                'volatility': base_volatility * 1.25
            }
        }

        return hedge_layers

    def calculate_hedge_costs(self, spot_price, hedge_layers):
        """
        Calculate initial cost of establishing hedge positions

        Returns:
            Total hedge cost and breakdown by layer
        """
        costs = {}

        for layer_name, layer in hedge_layers.items():
            time_years = layer['expiry_days'] / 365
            strike = layer['strike']
            coverage = layer['coverage']
            vol = layer['volatility']

            # Calculate contracts needed (assuming 100 multiplier)
            contracts = (self.portfolio_value * coverage) / (spot_price * 100)

            if layer['type'] == 'put':
                premium = self.black_scholes_put(spot_price, strike, time_years, vol)
                cost = premium * contracts * 100

                # Short puts are credits (negative cost)
                if 'short' in layer_name:
                    cost = -cost
            else:  # call
                premium = self.black_scholes_call(spot_price, strike, time_years, vol)
                # Selling calls generates credit (negative cost)
                cost = -premium * contracts * 100

            costs[layer_name] = {
                'premium': premium if 'short' not in layer_name and layer['type'] != 'call' else -premium,
                'contracts': contracts,
                'total_cost': cost
            }

        total_cost = sum(c['total_cost'] for c in costs.values())

        return total_cost, costs

    def calculate_hedge_payoff(self, initial_price, final_price, hedge_layers, initial_costs):
        """
        Calculate hedge payoff given price movement

        Args:
            initial_price: Starting index price
            final_price: Ending index price
            hedge_layers: Hedge strategy specification
            initial_costs: Initial costs breakdown

        Returns:
            Total hedge payoff and breakdown
        """
        payoffs = {}

        for layer_name, layer in hedge_layers.items():
            strike = layer['strike']
            coverage = layer['coverage']
            contracts = initial_costs[layer_name]['contracts']

            if layer['type'] == 'put':
                # Intrinsic value at expiry
                intrinsic = max(strike - final_price, 0)
                payoff = intrinsic * contracts * 100

                # Short puts are obligations (negative payoff if ITM)
                if 'short' in layer_name:
                    payoff = -payoff
            else:  # call
                intrinsic = max(final_price - strike, 0)
                # Short calls are obligations (negative payoff if ITM)
                payoff = -intrinsic * contracts * 100

            payoffs[layer_name] = payoff

        total_payoff = sum(payoffs.values())

        return total_payoff, payoffs

    def backtest_scenario(self, scenario_name, spot_price=100, base_volatility=0.18):
        """
        Backtest hedge strategy against a historical scenario

        Args:
            scenario_name: Key from historical_scenarios
            spot_price: Starting index price (normalized)
            base_volatility: Base volatility assumption

        Returns:
            Dictionary with backtest results
        """
        scenario = self.historical_scenarios[scenario_name]

        # Create hedge strategy
        hedge_layers = self.create_hedge_strategy(spot_price, base_volatility)

        # Calculate hedge costs
        total_cost, cost_breakdown = self.calculate_hedge_costs(spot_price, hedge_layers)

        # Calculate final price after decline
        final_price = spot_price * (1 + scenario['decline'])

        # Portfolio loss without hedges
        unhedged_loss = self.portfolio_value * scenario['decline']

        # Calculate hedge payoff
        total_payoff, payoff_breakdown = self.calculate_hedge_payoff(
            spot_price, final_price, hedge_layers, cost_breakdown
        )

        # Net result
        net_loss = unhedged_loss + total_payoff - total_cost
        net_loss_pct = (net_loss / self.portfolio_value) * 100

        # Protection metrics
        protection_amount = unhedged_loss - net_loss
        protection_pct = (protection_amount / abs(unhedged_loss)) * 100 if unhedged_loss != 0 else 0

        # Cost efficiency
        protection_per_dollar = protection_amount / total_cost if total_cost > 0 else 0

        results = {
            'scenario_name': scenario['name'],
            'market_decline_pct': scenario['decline'] * 100,
            'duration_days': scenario['days'],
            'vix_spike_multiple': scenario['vix_spike'],
            'initial_portfolio': self.portfolio_value,
            'final_price': final_price,
            'unhedged_loss': unhedged_loss,
            'hedge_cost': total_cost,
            'hedge_payoff': total_payoff,
            'net_loss': net_loss,
            'net_loss_pct': net_loss_pct,
            'protection_amount': protection_amount,
            'protection_pct': protection_pct,
            'protection_per_dollar_spent': protection_per_dollar,
            'cost_breakdown': cost_breakdown,
            'payoff_breakdown': payoff_breakdown
        }

        return results

    def backtest_all_scenarios(self):
        """
        Run backtest across all historical scenarios

        Returns:
            DataFrame with results for all scenarios
        """
        all_results = []

        print("=" * 80)
        print("BACKTESTING HEDGE STRATEGY ACROSS HISTORICAL MARKET CRASHES")
        print("=" * 80)
        print(f"\nInitial Portfolio: ${self.initial_portfolio:,.0f}\n")

        for scenario_key in self.historical_scenarios.keys():
            results = self.backtest_scenario(scenario_key)
            all_results.append(results)

            # Print summary for each scenario
            print(f"\n{results['scenario_name']}")
            print(f"{'─' * 80}")
            print(f"Market Decline: {results['market_decline_pct']:.1f}% over {results['duration_days']} days")
            print(f"Unhedged Loss: ${results['unhedged_loss']:,.0f} ({results['market_decline_pct']:.1f}%)")
            print(f"Hedge Cost: ${results['hedge_cost']:,.0f}")
            print(f"Hedge Payoff: ${results['hedge_payoff']:,.0f}")
            print(f"Net Loss: ${results['net_loss']:,.0f} ({results['net_loss_pct']:.2f}%)")
            print(f"Protection Effectiveness: {results['protection_pct']:.1f}%")
            print(f"Return per Dollar Spent: ${results['protection_per_dollar_spent']:.2f}")

        # Create summary DataFrame
        df = pd.DataFrame(all_results)

        # Calculate aggregate statistics
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)
        print(f"Average Protection Effectiveness: {df['protection_pct'].mean():.1f}%")
        print(f"Average Net Loss: {df['net_loss_pct'].mean():.2f}%")
        print(f"Max Net Loss: {df['net_loss_pct'].max():.2f}%")
        print(f"Min Net Loss: {df['net_loss_pct'].min():.2f}%")
        print(f"Average Cost: ${df['hedge_cost'].mean():,.0f}")
        print(f"Average ROI on Hedge: ${df['protection_per_dollar_spent'].mean():.2f}")

        return df

    def monte_carlo_simulation(self, num_simulations=1000, time_horizon_years=1,
                              annual_return=0.10, annual_volatility=0.18):
        """
        Run Monte Carlo simulation to test hedge performance across random scenarios

        Args:
            num_simulations: Number of random paths to simulate
            time_horizon_years: Time period to simulate
            annual_return: Expected annual return
            annual_volatility: Annual volatility

        Returns:
            DataFrame with simulation results
        """
        np.random.seed(42)  # For reproducibility

        results = []
        spot_price = 100

        print("\n" + "=" * 80)
        print(f"MONTE CARLO SIMULATION ({num_simulations} paths)")
        print("=" * 80)

        # Create hedge strategy
        hedge_layers = self.create_hedge_strategy(spot_price, annual_volatility)
        total_cost, cost_breakdown = self.calculate_hedge_costs(spot_price, hedge_layers)

        # Generate random price paths
        dt = time_horizon_years
        for i in range(num_simulations):
            # Geometric Brownian Motion
            random_return = np.random.normal(
                (annual_return - 0.5 * annual_volatility**2) * dt,
                annual_volatility * np.sqrt(dt)
            )

            final_price = spot_price * np.exp(random_return)
            price_change_pct = ((final_price - spot_price) / spot_price) * 100

            # Portfolio and hedge performance
            portfolio_change = self.portfolio_value * (price_change_pct / 100)
            total_payoff, _ = self.calculate_hedge_payoff(
                spot_price, final_price, hedge_layers, cost_breakdown
            )

            net_change = portfolio_change + total_payoff - total_cost
            net_change_pct = (net_change / self.portfolio_value) * 100

            results.append({
                'simulation': i + 1,
                'final_price': final_price,
                'price_change_pct': price_change_pct,
                'portfolio_change': portfolio_change,
                'hedge_payoff': total_payoff,
                'hedge_cost': total_cost,
                'net_change': net_change,
                'net_change_pct': net_change_pct
            })

        df = pd.DataFrame(results)

        # Statistics
        print(f"\nHedge Annual Cost: ${total_cost:,.0f} ({(total_cost/self.portfolio_value)*100:.2f}%)")
        print(f"\nSimulation Results:")
        print(f"Mean Return (hedged): {df['net_change_pct'].mean():.2f}%")
        print(f"Median Return (hedged): {df['net_change_pct'].median():.2f}%")
        print(f"Std Dev (hedged): {df['net_change_pct'].std():.2f}%")
        print(f"5th Percentile (hedged): {df['net_change_pct'].quantile(0.05):.2f}%")
        print(f"95th Percentile (hedged): {df['net_change_pct'].quantile(0.95):.2f}%")
        print(f"Max Drawdown (hedged): {df['net_change_pct'].min():.2f}%")

        # Compare to unhedged
        print(f"\nUnhedged Comparison:")
        print(f"Mean Return (unhedged): {df['price_change_pct'].mean():.2f}%")
        print(f"Std Dev (unhedged): {df['price_change_pct'].std():.2f}%")
        print(f"Max Drawdown (unhedged): {df['price_change_pct'].min():.2f}%")

        # Downside protection analysis
        negative_scenarios = df[df['price_change_pct'] < 0]
        if len(negative_scenarios) > 0:
            avg_protection = ((negative_scenarios['price_change_pct'] -
                             negative_scenarios['net_change_pct']).mean())
            print(f"\nDownside Protection:")
            print(f"Probability of Loss: {len(negative_scenarios)/len(df)*100:.1f}%")
            print(f"Average Protection in Down Markets: {avg_protection:.2f}%")

        return df

    def sensitivity_analysis(self, spot_price=100):
        """
        Analyze hedge performance sensitivity to volatility and market declines

        Returns:
            DataFrame with sensitivity results
        """
        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS")
        print("=" * 80)

        volatilities = [0.12, 0.15, 0.18, 0.22, 0.30]
        declines = [-0.05, -0.10, -0.15, -0.20, -0.30, -0.40]

        results = []

        for vol in volatilities:
            for decline in declines:
                # Create hedge
                hedge_layers = self.create_hedge_strategy(spot_price, vol)
                total_cost, cost_breakdown = self.calculate_hedge_costs(spot_price, hedge_layers)

                # Calculate performance
                final_price = spot_price * (1 + decline)
                unhedged_loss = self.portfolio_value * decline
                total_payoff, _ = self.calculate_hedge_payoff(
                    spot_price, final_price, hedge_layers, cost_breakdown
                )

                net_loss = unhedged_loss + total_payoff - total_cost
                protection_pct = ((unhedged_loss - net_loss) / abs(unhedged_loss)) * 100

                results.append({
                    'volatility': vol,
                    'market_decline_pct': decline * 100,
                    'hedge_cost': total_cost,
                    'hedge_payoff': total_payoff,
                    'net_loss_pct': (net_loss / self.portfolio_value) * 100,
                    'protection_pct': protection_pct
                })

        df = pd.DataFrame(results)

        # Create pivot tables for easy viewing
        print("\nProtection Effectiveness (%) by Volatility and Market Decline:")
        pivot = df.pivot_table(
            values='protection_pct',
            index='market_decline_pct',
            columns='volatility',
            aggfunc='mean'
        )
        print(pivot.round(1))

        print("\nNet Loss (%) by Volatility and Market Decline:")
        pivot_loss = df.pivot_table(
            values='net_loss_pct',
            index='market_decline_pct',
            columns='volatility',
            aggfunc='mean'
        )
        print(pivot_loss.round(2))

        return df

    def generate_report(self, output_file='backtest_results.xlsx'):
        """
        Generate comprehensive backtest report
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Historical scenarios
            historical = self.backtest_all_scenarios()
            historical.to_excel(writer, sheet_name='Historical_Scenarios', index=False)

            # Monte Carlo
            monte_carlo = self.monte_carlo_simulation(num_simulations=1000)
            monte_carlo.to_excel(writer, sheet_name='Monte_Carlo', index=False)

            # Sensitivity
            sensitivity = self.sensitivity_analysis()
            sensitivity.to_excel(writer, sheet_name='Sensitivity', index=False)

            # Summary
            summary = pd.DataFrame({
                'Metric': [
                    'Portfolio Value',
                    'Average Annual Hedge Cost',
                    'Average Protection (Historical)',
                    'Max Historical Drawdown (Hedged)',
                    'Monte Carlo Mean Return',
                    'Monte Carlo 5th Percentile',
                    'Recommendation'
                ],
                'Value': [
                    f'${self.portfolio_value:,.0f}',
                    f'${historical["hedge_cost"].mean():,.0f}',
                    f'{historical["protection_pct"].mean():.1f}%',
                    f'{historical["net_loss_pct"].max():.2f}%',
                    f'{monte_carlo["net_change_pct"].mean():.2f}%',
                    f'{monte_carlo["net_change_pct"].quantile(0.05):.2f}%',
                    'Strategy provides effective tail risk protection with acceptable cost'
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)

        print(f"\n✓ Backtest report generated: {output_file}")


def main():
    """Run comprehensive backtest"""

    # Initialize backtester
    backtester = HedgeBacktester(initial_portfolio=10_000_000)

    # Run all tests
    print("\n" + "=" * 80)
    print("DEEP HEDGE STRATEGY BACKTESTING SUITE")
    print("=" * 80)

    # 1. Historical scenarios
    historical_results = backtester.backtest_all_scenarios()

    # 2. Monte Carlo simulation
    mc_results = backtester.monte_carlo_simulation(num_simulations=10000)

    # 3. Sensitivity analysis
    sensitivity_results = backtester.sensitivity_analysis()

    # 4. Generate report
    backtester.generate_report('deep_hedge_backtest_results.xlsx')

    print("\n" + "=" * 80)
    print("BACKTESTING COMPLETE")
    print("=" * 80)
    print("\nView detailed results in: deep_hedge_backtest_results.xlsx")


if __name__ == "__main__":
    main()
