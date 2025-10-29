"""
Deep Hedge Multi-Layer Protection Calculator
Calculates optimal hedge ratios, costs, and protection levels for portfolio hedging
Supports optional Databento integration for real market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import Optional

# Optional Databento integration
try:
    from databento_provider import DatabentoProvider
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False

class DeepHedgeCalculator:
    """
    Multi-layered portfolio hedging calculator with Black-Scholes pricing
    
    Supports optional integration with Databento for real market data:
    - Use actual volatility from VIX
    - Fetch real option prices
    - Validate calculations against market data
    """

    def __init__(self, portfolio_value, risk_free_rate=0.045, annual_volatility=0.18, 
                 databento_provider: Optional['DatabentoProvider'] = None):
        """
        Initialize the hedge calculator

        Args:
            portfolio_value: Total portfolio value in dollars
            risk_free_rate: Annual risk-free rate (default 4.5%)
            annual_volatility: Annual volatility (default 18%)
            databento_provider: Optional DatabentoProvider instance for real market data
        """
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        self.annual_volatility = annual_volatility
        self.current_price = 100  # Normalized index price
        self.databento_provider = databento_provider

    def black_scholes_put(self, strike, time_to_expiry, spot_price=None):
        """
        Calculate put option price using Black-Scholes formula

        Args:
            strike: Strike price
            time_to_expiry: Time to expiration in years
            spot_price: Current spot price (default: self.current_price)

        Returns:
            Put option price
        """
        if spot_price is None:
            spot_price = self.current_price

        if time_to_expiry <= 0:
            return max(strike - spot_price, 0)

        d1 = (np.log(spot_price / strike) + (self.risk_free_rate + 0.5 * self.annual_volatility**2) * time_to_expiry) / \
             (self.annual_volatility * np.sqrt(time_to_expiry))
        d2 = d1 - self.annual_volatility * np.sqrt(time_to_expiry)

        put_price = strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - \
                    spot_price * norm.cdf(-d1)

        return put_price

    def black_scholes_call(self, strike, time_to_expiry, spot_price=None):
        """
        Calculate call option price using Black-Scholes formula
        """
        if spot_price is None:
            spot_price = self.current_price

        if time_to_expiry <= 0:
            return max(spot_price - strike, 0)

        d1 = (np.log(spot_price / strike) + (self.risk_free_rate + 0.5 * self.annual_volatility**2) * time_to_expiry) / \
             (self.annual_volatility * np.sqrt(time_to_expiry))
        d2 = d1 - self.annual_volatility * np.sqrt(time_to_expiry)

        call_price = spot_price * norm.cdf(d1) - \
                     strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)

        return call_price
    
    def update_volatility_from_vix(self, start_date: str, end_date: str) -> float:
        """
        Update volatility using real VIX data from Databento
        
        Args:
            start_date: Start date for VIX data (YYYY-MM-DD)
            end_date: End date for VIX data (YYYY-MM-DD)
            
        Returns:
            Updated annual volatility
        """
        if not self.databento_provider:
            print("Warning: Databento provider not configured. Using default volatility.")
            return self.annual_volatility
        
        try:
            vix_data = self.databento_provider.get_vix_data(start_date, end_date)
            
            if not vix_data.empty and 'close' in vix_data.columns:
                # VIX is in percentage points, convert to decimal
                avg_vix = vix_data['close'].mean()
                self.annual_volatility = avg_vix / 100.0
                
                print(f"✓ Updated volatility from VIX data: {self.annual_volatility*100:.2f}%")
                return self.annual_volatility
            else:
                print("Warning: No VIX data retrieved. Using default volatility.")
                return self.annual_volatility
                
        except Exception as e:
            print(f"Error fetching VIX data: {e}. Using default volatility.")
            return self.annual_volatility

    def calculate_put_spread_cost(self, long_strike, short_strike, time_to_expiry):
        """Calculate cost of a put spread (buy higher strike, sell lower strike)"""
        long_put = self.black_scholes_put(long_strike, time_to_expiry)
        short_put = self.black_scholes_put(short_strike, time_to_expiry)
        return long_put - short_put

    def calculate_collar_cost(self, put_strike, call_strike, time_to_expiry):
        """Calculate net cost of collar (buy put, sell call)"""
        put_cost = self.black_scholes_put(put_strike, time_to_expiry)
        call_credit = self.black_scholes_call(call_strike, time_to_expiry)
        return put_cost - call_credit

    def calculate_layer_costs(self):
        """
        Calculate costs for all hedge layers

        Returns:
            DataFrame with layer details and costs
        """
        layers = []

        # Layer 1: Near-term protection (3 months)
        layer1_collar = {
            'Layer': 'Layer 1: Daily Protection',
            'Strategy': 'Collar (Buy 98 Put / Sell 102 Call)',
            'Time_to_Expiry_Days': 90,
            'Time_to_Expiry_Years': 90/365,
            'Put_Strike': 98,
            'Call_Strike': 102,
            'Cost_Per_Unit': self.calculate_collar_cost(98, 102, 90/365),
            'Contracts_Needed': self.portfolio_value / (self.current_price * 100),  # Assuming 100 multiplier
            'Protection_Level': '0-2% decline',
            'Allocation_Percent': 100  # Full portfolio
        }
        layers.append(layer1_collar)

        # Layer 2: Intermediate protection (6 months)
        put_spread_cost = self.calculate_put_spread_cost(95, 85, 180/365)
        layer2 = {
            'Layer': 'Layer 2: Moderate Correction',
            'Strategy': 'Put Spread (Buy 95 / Sell 85)',
            'Time_to_Expiry_Days': 180,
            'Time_to_Expiry_Years': 180/365,
            'Put_Strike': 95,
            'Call_Strike': None,
            'Cost_Per_Unit': put_spread_cost,
            'Contracts_Needed': self.portfolio_value / (self.current_price * 100) * 0.75,  # 75% of portfolio
            'Protection_Level': '5-15% decline',
            'Allocation_Percent': 75
        }
        layers.append(layer2)

        # Layer 3a: Crisis protection - Far OTM puts (12 months)
        far_otm_put = self.black_scholes_put(80, 365/365)
        layer3a = {
            'Layer': 'Layer 3a: Crisis Protection',
            'Strategy': 'Far OTM Puts (80 Strike)',
            'Time_to_Expiry_Days': 365,
            'Time_to_Expiry_Years': 1.0,
            'Put_Strike': 80,
            'Call_Strike': None,
            'Cost_Per_Unit': far_otm_put,
            'Contracts_Needed': self.portfolio_value / (self.current_price * 100) * 0.5,  # 50% of portfolio
            'Protection_Level': '20%+ decline',
            'Allocation_Percent': 50
        }
        layers.append(layer3a)

        df = pd.DataFrame(layers)
        df['Total_Cost'] = df['Cost_Per_Unit'] * df['Contracts_Needed'] * 100  # 100 multiplier
        df['Cost_Percent_of_Portfolio'] = (df['Total_Cost'] / self.portfolio_value) * 100

        return df

    def scenario_analysis(self, market_declines=None):
        """
        Analyze portfolio protection under different market decline scenarios

        Args:
            market_declines: List of market decline percentages (e.g., [-5, -10, -20, -30])

        Returns:
            DataFrame with scenario analysis results
        """
        if market_declines is None:
            market_declines = [-2, -5, -10, -15, -20, -30, -40]

        scenarios = []

        for decline_pct in market_declines:
            new_price = self.current_price * (1 + decline_pct/100)

            # Calculate portfolio loss without hedges
            unhedged_loss = self.portfolio_value * (decline_pct/100)

            # Calculate hedge payoffs
            # Layer 1: Collar (98 put, 102 call) - 3 months
            layer1_put_payoff = max(98 - new_price, 0) * (self.portfolio_value / self.current_price)
            layer1_call_loss = max(new_price - 102, 0) * (self.portfolio_value / self.current_price)
            layer1_net = layer1_put_payoff - layer1_call_loss

            # Layer 2: Put spread (95/85) - 75% coverage
            layer2_long_payoff = max(95 - new_price, 0) * (self.portfolio_value * 0.75 / self.current_price)
            layer2_short_loss = max(85 - new_price, 0) * (self.portfolio_value * 0.75 / self.current_price)
            layer2_net = layer2_long_payoff - layer2_short_loss

            # Layer 3: Far OTM put (80 strike) - 50% coverage
            layer3_payoff = max(80 - new_price, 0) * (self.portfolio_value * 0.5 / self.current_price)

            total_hedge_payoff = layer1_net + layer2_net + layer3_payoff

            # Get hedge costs
            layer_costs = self.calculate_layer_costs()
            total_hedge_cost = layer_costs['Total_Cost'].sum()

            # Net P&L (portfolio loss + hedge payoff - hedge cost)
            net_loss = unhedged_loss + total_hedge_payoff - total_hedge_cost
            net_loss_pct = (net_loss / self.portfolio_value) * 100

            # Protection effectiveness
            protection_pct = ((unhedged_loss - net_loss) / abs(unhedged_loss)) * 100 if unhedged_loss != 0 else 0

            scenarios.append({
                'Market_Decline_Pct': decline_pct,
                'New_Index_Price': new_price,
                'Unhedged_Loss': unhedged_loss,
                'Layer1_Payoff': layer1_net,
                'Layer2_Payoff': layer2_net,
                'Layer3_Payoff': layer3_payoff,
                'Total_Hedge_Payoff': total_hedge_payoff,
                'Hedge_Cost': total_hedge_cost,
                'Net_Loss': net_loss,
                'Net_Loss_Pct': net_loss_pct,
                'Protection_Effectiveness_Pct': protection_pct
            })

        return pd.DataFrame(scenarios)

    def optimal_hedge_ratio(self, target_max_loss_pct=15, max_hedge_cost_pct=2):
        """
        Calculate optimal hedge ratio to achieve target maximum loss

        Args:
            target_max_loss_pct: Maximum acceptable loss percentage (e.g., 15 for 15%)
            max_hedge_cost_pct: Maximum acceptable annual hedge cost (e.g., 2 for 2%)

        Returns:
            Dictionary with optimal hedge parameters
        """
        # Test different hedge coverage levels
        results = []

        for coverage in np.arange(0.25, 1.05, 0.05):  # 25% to 100% coverage
            # Adjust allocation percentages
            temp_calculator = DeepHedgeCalculator(
                self.portfolio_value * coverage,
                self.risk_free_rate,
                self.annual_volatility
            )

            # Run scenario for -30% market decline (stress test)
            scenario = temp_calculator.scenario_analysis([-30])

            # Calculate effective loss on full portfolio
            hedge_payoff = scenario.iloc[0]['Total_Hedge_Payoff']
            hedge_cost = scenario.iloc[0]['Hedge_Cost']
            unhedged_portion_loss = self.portfolio_value * (1 - coverage) * -0.30
            hedged_portion_loss = scenario.iloc[0]['Net_Loss']

            total_loss = unhedged_portion_loss + hedged_portion_loss
            total_loss_pct = (total_loss / self.portfolio_value) * 100
            total_cost_pct = (hedge_cost / self.portfolio_value) * 100

            results.append({
                'Coverage_Ratio': coverage,
                'Total_Loss_Pct_at_30pct_Decline': total_loss_pct,
                'Annual_Hedge_Cost_Pct': total_cost_pct,
                'Meets_Loss_Target': abs(total_loss_pct) <= target_max_loss_pct,
                'Meets_Cost_Target': total_cost_pct <= max_hedge_cost_pct
            })

        df = pd.DataFrame(results)

        # Find optimal ratio that meets both criteria
        optimal = df[df['Meets_Loss_Target'] & df['Meets_Cost_Target']]

        if len(optimal) > 0:
            # Choose lowest cost option that meets targets
            best = optimal.loc[optimal['Annual_Hedge_Cost_Pct'].idxmin()]
            return {
                'optimal_coverage_ratio': best['Coverage_Ratio'],
                'expected_max_loss_pct': best['Total_Loss_Pct_at_30pct_Decline'],
                'annual_cost_pct': best['Annual_Hedge_Cost_Pct'],
                'recommendation': 'OPTIMAL FOUND',
                'all_results': df
            }
        else:
            # Find closest to target
            df['distance_from_target'] = abs(df['Total_Loss_Pct_at_30pct_Decline'] + target_max_loss_pct)
            best = df.loc[df['distance_from_target'].idxmin()]
            return {
                'optimal_coverage_ratio': best['Coverage_Ratio'],
                'expected_max_loss_pct': best['Total_Loss_Pct_at_30pct_Decline'],
                'annual_cost_pct': best['Annual_Hedge_Cost_Pct'],
                'recommendation': 'NO PERFECT SOLUTION - CLOSEST MATCH',
                'all_results': df
            }

    def generate_report(self, output_file='hedge_analysis_report.xlsx'):
        """
        Generate comprehensive Excel report with all analyses

        Args:
            output_file: Output Excel filename
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Input Parameters
            params = pd.DataFrame({
                'Parameter': ['Portfolio Value', 'Risk-Free Rate', 'Annual Volatility',
                             'Current Index Price', 'Analysis Date'],
                'Value': [f'${self.portfolio_value:,.0f}', f'{self.risk_free_rate*100:.2f}%',
                         f'{self.annual_volatility*100:.1f}%', self.current_price,
                         datetime.now().strftime('%Y-%m-%d')]
            })
            params.to_excel(writer, sheet_name='Parameters', index=False)

            # Sheet 2: Layer Costs
            layer_costs = self.calculate_layer_costs()
            layer_costs.to_excel(writer, sheet_name='Layer_Costs', index=False)

            # Sheet 3: Scenario Analysis
            scenarios = self.scenario_analysis()
            scenarios.to_excel(writer, sheet_name='Scenario_Analysis', index=False)

            # Sheet 4: Optimal Hedge Ratio
            optimal = self.optimal_hedge_ratio()
            optimal_df = optimal['all_results']
            optimal_df.to_excel(writer, sheet_name='Optimal_Hedge_Ratio', index=False)

            # Sheet 5: Summary
            total_cost = layer_costs['Total_Cost'].sum()
            summary = pd.DataFrame({
                'Metric': [
                    'Total Annual Hedge Cost',
                    'Hedge Cost % of Portfolio',
                    'Recommended Coverage Ratio',
                    'Expected Max Loss (30% decline)',
                    'Protection at -10% market',
                    'Protection at -20% market',
                    'Protection at -30% market'
                ],
                'Value': [
                    f'${total_cost:,.0f}',
                    f'{(total_cost/self.portfolio_value)*100:.2f}%',
                    f'{optimal["optimal_coverage_ratio"]*100:.0f}%',
                    f'{optimal["expected_max_loss_pct"]:.2f}%',
                    f'{scenarios[scenarios["Market_Decline_Pct"]==-10]["Net_Loss_Pct"].values[0]:.2f}%',
                    f'{scenarios[scenarios["Market_Decline_Pct"]==-20]["Net_Loss_Pct"].values[0]:.2f}%',
                    f'{scenarios[scenarios["Market_Decline_Pct"]==-30]["Net_Loss_Pct"].values[0]:.2f}%'
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)

        print(f"\n✓ Report generated: {output_file}")
        return output_file


def main():
    """
    Example usage of the Deep Hedge Calculator
    Demonstrates both standard and Databento-enhanced usage
    """
    print("=" * 70)
    print("DEEP HEDGE MULTI-LAYER PROTECTION CALCULATOR")
    print("=" * 70)

    # Check for Databento integration
    import os
    databento_available = DATABENTO_AVAILABLE and os.environ.get('DATABENTO_API_KEY')
    
    if databento_available:
        print("\n✓ Databento integration available")
        print("  Will use real market data where applicable")
    else:
        print("\n⚠️  Databento not configured - using Black-Scholes models")
        if not DATABENTO_AVAILABLE:
            print("  Install with: pip install databento")
        if not os.environ.get('DATABENTO_API_KEY'):
            print("  Set API key: export DATABENTO_API_KEY='your_key'")

    # Initialize calculator with $10M portfolio
    portfolio_value = 10_000_000
    
    # Optional: Initialize with Databento provider
    databento_provider = None
    if databento_available:
        try:
            databento_provider = DatabentoProvider(api_key=os.environ.get('DATABENTO_API_KEY'))
            print("  ✓ Databento provider initialized")
        except Exception as e:
            print(f"  Warning: Could not initialize Databento: {e}")
    
    calculator = DeepHedgeCalculator(
        portfolio_value=portfolio_value,
        risk_free_rate=0.045,  # 4.5%
        annual_volatility=0.18,  # 18%
        databento_provider=databento_provider
    )

    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Risk-Free Rate: {calculator.risk_free_rate*100:.1f}%")
    print(f"Annual Volatility: {calculator.annual_volatility*100:.1f}%")
    
    # Optional: Update volatility from real VIX data
    if databento_provider:
        print("\nAttempting to update volatility from VIX data...")
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        calculator.update_volatility_from_vix(start_date, end_date)

    # Calculate layer costs
    print("\n" + "=" * 70)
    print("HEDGE LAYER COSTS")
    print("=" * 70)
    layer_costs = calculator.calculate_layer_costs()
    print(layer_costs[['Layer', 'Strategy', 'Cost_Per_Unit', 'Total_Cost', 'Cost_Percent_of_Portfolio']].to_string(index=False))

    total_cost = layer_costs['Total_Cost'].sum()
    print(f"\nTotal Annual Hedge Cost: ${total_cost:,.0f} ({(total_cost/portfolio_value)*100:.2f}% of portfolio)")

    # Scenario analysis
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS")
    print("=" * 70)
    scenarios = calculator.scenario_analysis()
    print(scenarios[['Market_Decline_Pct', 'Unhedged_Loss', 'Total_Hedge_Payoff',
                     'Net_Loss', 'Net_Loss_Pct', 'Protection_Effectiveness_Pct']].to_string(index=False))

    # Optimal hedge ratio
    print("\n" + "=" * 70)
    print("OPTIMAL HEDGE RATIO ANALYSIS")
    print("=" * 70)
    optimal = calculator.optimal_hedge_ratio(target_max_loss_pct=15, max_hedge_cost_pct=2.5)
    print(f"Recommendation: {optimal['recommendation']}")
    print(f"Optimal Coverage Ratio: {optimal['optimal_coverage_ratio']*100:.1f}%")
    print(f"Expected Max Loss (30% decline): {optimal['expected_max_loss_pct']:.2f}%")
    print(f"Annual Hedge Cost: {optimal['annual_cost_pct']:.2f}%")

    # Generate Excel report
    print("\n" + "=" * 70)
    print("GENERATING EXCEL REPORT")
    print("=" * 70)
    report_file = calculator.generate_report('deep_hedge_analysis.xlsx')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nView detailed results in: {report_file}")


if __name__ == "__main__":
    main()
