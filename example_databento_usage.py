"""
Example: Using Databento Integration with Deep Hedge Toolkit
Demonstrates how to fetch real market data and enhance backtesting
"""

import os
import sys
from datetime import datetime, timedelta

# Import core hedge toolkit
from hedge_calculator import DeepHedgeCalculator

# Try to import Databento provider
try:
    from databento_provider import DatabentoProvider, DatabentoBacktestEnhancer, DATABENTO_AVAILABLE
except ImportError:
    print("Error: databento_provider.py not found")
    sys.exit(1)


def example_historical_prices():
    """Example 1: Fetch historical price data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Fetching Historical Price Data")
    print("=" * 80)
    
    if not DATABENTO_AVAILABLE:
        print("⚠️  Databento package not installed. Showing example code only.")
        print("\nTo use this feature, install databento:")
        print("  pip install databento")
        print("\nThen set your API key:")
        print("  export DATABENTO_API_KEY='your_key_here'")
        return
    
    api_key = os.environ.get('DATABENTO_API_KEY')
    if not api_key:
        print("⚠️  DATABENTO_API_KEY environment variable not set")
        print("\nSet your API key:")
        print("  export DATABENTO_API_KEY='your_key_here'")
        print("\nGet your API key from: https://databento.com")
        return
    
    try:
        provider = DatabentoProvider(api_key=api_key)
        
        # Fetch SPY data for 2020 (includes COVID crash)
        print("\nFetching SPY data for 2020...")
        spy_data = provider.get_historical_prices(
            symbol='SPY',
            start_date='2020-01-01',
            end_date='2020-12-31',
            dataset='XNAS.ITCH'
        )
        
        if not spy_data.empty:
            print(f"\n✓ Successfully fetched {len(spy_data)} trading days")
            print("\nFirst 5 rows:")
            print(spy_data.head())
            print("\nLast 5 rows:")
            print(spy_data.tail())
        else:
            print("No data returned. Check your dataset and symbol.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Valid Databento API key")
        print("2. Appropriate dataset subscription")


def example_vix_data():
    """Example 2: Fetch VIX volatility data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Fetching VIX Volatility Data")
    print("=" * 80)
    
    if not DATABENTO_AVAILABLE:
        print("⚠️  Databento package not installed.")
        return
    
    api_key = os.environ.get('DATABENTO_API_KEY')
    if not api_key:
        print("⚠️  DATABENTO_API_KEY not set")
        return
    
    try:
        provider = DatabentoProvider(api_key=api_key)
        
        print("\nFetching VIX data for 2020...")
        vix_data = provider.get_vix_data(
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        if not vix_data.empty:
            print(f"\n✓ Successfully fetched {len(vix_data)} VIX data points")
            print("\nVIX Statistics:")
            print(f"  Mean VIX: {vix_data['close'].mean():.2f}")
            print(f"  Max VIX: {vix_data['close'].max():.2f}")
            print(f"  Min VIX: {vix_data['close'].min():.2f}")
            print(f"  Std Dev: {vix_data['close'].std():.2f}")
        else:
            print("No VIX data returned.")
            
    except Exception as e:
        print(f"Error: {e}")


def example_real_crash_analysis():
    """Example 3: Analyze real market crashes"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Real Market Crash Analysis")
    print("=" * 80)
    
    if not DATABENTO_AVAILABLE:
        print("⚠️  Databento package not installed.")
        return
    
    api_key = os.environ.get('DATABENTO_API_KEY')
    if not api_key:
        print("⚠️  DATABENTO_API_KEY not set")
        return
    
    try:
        provider = DatabentoProvider(api_key=api_key)
        
        print("\nFetching historical market crash data...")
        crashes = provider.get_historical_market_crashes(
            symbol='SPY',
            dataset='XNAS.ITCH'
        )
        
        if crashes:
            print(f"\n✓ Successfully analyzed {len(crashes)} crash periods\n")
            
            for crash in crashes:
                print(f"\n{crash['name']}:")
                print(f"  Period: {crash['start_date']} to {crash['end_date']}")
                print(f"  Duration: {crash['duration_days']} days")
                print(f"  Decline: {crash['decline_pct']:.2f}%")
                print(f"  Initial Price: ${crash['initial_price']:.2f}")
                print(f"  Final Price: ${crash['final_price']:.2f}")
        else:
            print("No crash data retrieved.")
            
    except Exception as e:
        print(f"Error: {e}")


def example_enhanced_backtesting():
    """Example 4: Enhanced backtesting with real data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Enhanced Backtesting with Real Market Data")
    print("=" * 80)
    
    if not DATABENTO_AVAILABLE:
        print("⚠️  Databento package not installed.")
        print("\nThis example would:")
        print("1. Fetch real market crash data from Databento")
        print("2. Apply hedge strategy to actual price movements")
        print("3. Calculate real protection effectiveness")
        print("4. Compare to simulated scenarios")
        return
    
    api_key = os.environ.get('DATABENTO_API_KEY')
    if not api_key:
        print("⚠️  DATABENTO_API_KEY not set")
        return
    
    try:
        # Initialize provider and calculator
        provider = DatabentoProvider(api_key=api_key)
        calculator = DeepHedgeCalculator(
            portfolio_value=10_000_000,
            risk_free_rate=0.045,
            annual_volatility=0.18
        )
        
        # Initialize enhancer
        enhancer = DatabentoBacktestEnhancer(provider)
        
        # Define crash periods to test
        crash_periods = [
            {'name': '2020 COVID Crash', 'start': '2020-02-19', 'end': '2020-03-23'},
            {'name': '2022 Bear Market', 'start': '2022-01-03', 'end': '2022-10-13'},
        ]
        
        print("\nRunning enhanced backtest with real data...")
        results = enhancer.backtest_with_real_data(
            symbol='SPY',
            crash_periods=crash_periods,
            hedge_calculator=calculator,
            dataset='XNAS.ITCH'
        )
        
        if not results.empty:
            print("\n✓ Backtest complete!\n")
            print(results.to_string(index=False))
        else:
            print("No backtest results generated.")
            
    except Exception as e:
        print(f"Error: {e}")


def example_without_databento():
    """Example 5: Using toolkit without Databento (fallback)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Standard Toolkit Usage (No Databento Required)")
    print("=" * 80)
    
    print("\nInitializing hedge calculator...")
    calculator = DeepHedgeCalculator(
        portfolio_value=10_000_000,
        risk_free_rate=0.045,
        annual_volatility=0.18
    )
    
    print("\nCalculating hedge layer costs...")
    costs = calculator.calculate_layer_costs()
    print("\nHedge Layer Costs:")
    print(costs[['Layer', 'Strategy', 'Total_Cost', 'Cost_Percent_of_Portfolio']].to_string(index=False))
    
    total_cost = costs['Total_Cost'].sum()
    print(f"\nTotal Annual Cost: ${total_cost:,.0f}")
    print(f"As % of Portfolio: {(total_cost / 10_000_000) * 100:.2f}%")
    
    print("\n✓ Standard toolkit functionality works without Databento!")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("DATABENTO INTEGRATION EXAMPLES")
    print("Deep Hedge Toolkit with Real Market Data")
    print("=" * 80)
    
    # Check Databento availability
    if DATABENTO_AVAILABLE:
        print("\n✓ Databento package is installed")
    else:
        print("\n⚠️  Databento package is NOT installed")
        print("   Install with: pip install databento")
    
    # Check API key
    api_key = os.environ.get('DATABENTO_API_KEY')
    if api_key:
        print(f"✓ DATABENTO_API_KEY is set (length: {len(api_key)})")
    else:
        print("⚠️  DATABENTO_API_KEY environment variable is not set")
        print("   Set with: export DATABENTO_API_KEY='your_key'")
    
    print("\nRunning examples...\n")
    
    # Run examples
    example_without_databento()
    example_historical_prices()
    example_vix_data()
    example_real_crash_analysis()
    example_enhanced_backtesting()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Get Databento API key: https://databento.com")
    print("2. Set environment variable: export DATABENTO_API_KEY='your_key'")
    print("3. Install databento: pip install databento")
    print("4. Run this script again to see real data examples")
    print("\nFor more information, see databento_provider.py")


if __name__ == "__main__":
    main()
