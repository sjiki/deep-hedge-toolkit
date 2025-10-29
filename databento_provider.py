"""
Databento Market Data Provider
Integrates with Databento API to fetch real market data for hedge calculations and backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    print("Warning: databento package not installed. Install with: pip install databento")


class DatabentoProvider:
    """
    Provider for fetching market data from Databento API
    
    Features:
    - Historical equity and index prices for backtesting
    - Option prices and implied volatility
    - VIX data for volatility calculations
    - Support for multiple datasets (OPRA, DBEQ, etc.)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Databento provider
        
        Args:
            api_key: Databento API key (if None, will try to read from environment)
        """
        if not DATABENTO_AVAILABLE:
            raise ImportError(
                "databento package is required. Install with: pip install databento"
            )
        
        self.client = db.Historical(api_key)
        self.api_key = api_key
        
    def get_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        dataset: str = 'XNAS.ITCH',
        schema: str = 'ohlcv-1d'
    ) -> pd.DataFrame:
        """
        Fetch historical price data
        
        Args:
            symbol: Ticker symbol (e.g., 'SPY', 'QQQ')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            dataset: Databento dataset (default: XNAS.ITCH for NASDAQ)
            schema: Data schema (default: ohlcv-1d for daily OHLCV)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            data = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                start=start_date,
                end=end_date,
                schema=schema
            )
            
            df = data.to_df()
            
            # Standardize column names
            df = df.rename(columns={
                'ts_event': 'date',
            })
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_option_chain(
        self,
        symbol: str,
        date: str,
        dataset: str = 'OPRA'
    ) -> pd.DataFrame:
        """
        Fetch option chain data for a specific date
        
        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            date: Date in 'YYYY-MM-DD' format
            dataset: Databento dataset (default: OPRA for options)
            
        Returns:
            DataFrame with option chain data including strikes, premiums, greeks
        """
        try:
            # Fetch options data
            data = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                start=date,
                end=date,
                schema='definition'
            )
            
            df = data.to_df()
            
            # Parse option chain
            # Note: Actual implementation would parse option symbols and organize by strike/expiry
            return df
            
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_vix_data(
        self,
        start_date: str,
        end_date: str,
        dataset: str = 'XNAS.ITCH'
    ) -> pd.DataFrame:
        """
        Fetch VIX (volatility index) data
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            dataset: Databento dataset
            
        Returns:
            DataFrame with VIX values
        """
        try:
            # Fetch VIX data (symbol might vary by dataset)
            vix_data = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=['VIX'],
                start=start_date,
                end=end_date,
                schema='ohlcv-1d'
            )
            
            df = vix_data.to_df()
            
            if not df.empty and 'ts_event' in df.columns:
                df['date'] = pd.to_datetime(df['ts_event'])
                df = df.set_index('date')
            
            return df
            
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def calculate_implied_volatility_from_chain(
        self,
        option_chain: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.045
    ) -> Dict[str, float]:
        """
        Calculate implied volatility from option chain
        
        Args:
            option_chain: DataFrame with option prices
            spot_price: Current spot price
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with ATM, OTM put, OTM call implied volatilities
        """
        # This would implement IV calculation from option prices
        # For now, return placeholder structure
        return {
            'atm_iv': 0.18,
            'otm_put_iv': 0.22,
            'otm_call_iv': 0.15
        }
    
    def get_historical_market_crashes(
        self,
        symbol: str = 'SPY',
        dataset: str = 'XNAS.ITCH'
    ) -> List[Dict]:
        """
        Identify historical market crash periods from actual data
        
        Args:
            symbol: Index symbol to analyze
            dataset: Databento dataset
            
        Returns:
            List of dictionaries with crash information
        """
        crashes = []
        
        # Define crash periods to fetch
        crash_periods = [
            {'name': '2020 COVID Crash', 'start': '2020-02-19', 'end': '2020-03-23'},
            {'name': '2022 Bear Market', 'start': '2022-01-03', 'end': '2022-10-13'},
            {'name': '2018 Q4 Correction', 'start': '2018-09-20', 'end': '2018-12-24'},
        ]
        
        for period in crash_periods:
            try:
                df = self.get_historical_prices(
                    symbol=symbol,
                    start_date=period['start'],
                    end_date=period['end'],
                    dataset=dataset
                )
                
                if not df.empty and 'close' in df.columns:
                    initial_price = df['close'].iloc[0]
                    final_price = df['close'].iloc[-1]
                    decline_pct = ((final_price - initial_price) / initial_price) * 100
                    duration_days = len(df)
                    
                    crashes.append({
                        'name': period['name'],
                        'start_date': period['start'],
                        'end_date': period['end'],
                        'initial_price': initial_price,
                        'final_price': final_price,
                        'decline_pct': decline_pct,
                        'duration_days': duration_days,
                        'data': df
                    })
            except Exception as e:
                print(f"Error fetching crash data for {period['name']}: {e}")
        
        return crashes
    
    def get_greeks_from_options(
        self,
        option_chain: pd.DataFrame,
        spot_price: float
    ) -> pd.DataFrame:
        """
        Extract or calculate Greeks from option chain data
        
        Args:
            option_chain: DataFrame with option data
            spot_price: Current spot price
            
        Returns:
            DataFrame with Delta, Gamma, Theta, Vega for each option
        """
        # This would extract Greeks if available in the data
        # or calculate them from prices
        return pd.DataFrame()


class DatabentoBacktestEnhancer:
    """
    Enhances backtesting with real market data from Databento
    """
    
    def __init__(self, provider: DatabentoProvider):
        """
        Initialize with a Databento provider
        
        Args:
            provider: Configured DatabentoProvider instance
        """
        self.provider = provider
    
    def backtest_with_real_data(
        self,
        symbol: str,
        crash_periods: List[Dict],
        hedge_calculator,
        dataset: str = 'XNAS.ITCH'
    ) -> pd.DataFrame:
        """
        Backtest hedge strategy using real historical market data
        
        Args:
            symbol: Ticker symbol
            crash_periods: List of crash period definitions
            hedge_calculator: Instance of hedge calculator
            dataset: Databento dataset
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        
        for period in crash_periods:
            # Get actual market data
            market_data = self.provider.get_historical_prices(
                symbol=symbol,
                start_date=period['start'],
                end_date=period['end'],
                dataset=dataset
            )
            
            if market_data.empty:
                continue
            
            # Calculate actual decline
            initial_price = market_data['close'].iloc[0]
            final_price = market_data['close'].iloc[-1]
            actual_decline = ((final_price - initial_price) / initial_price)
            
            # Run hedge calculation with actual data
            # This would integrate with the existing hedge calculator
            result = {
                'period': period['name'],
                'start_date': period['start'],
                'end_date': period['end'],
                'initial_price': initial_price,
                'final_price': final_price,
                'actual_decline_pct': actual_decline * 100,
                'duration_days': len(market_data)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)


def get_sample_usage():
    """
    Return sample usage code as a string
    """
    return """
# Sample Usage:

import os
from databento_provider import DatabentoProvider, DatabentoBacktestEnhancer

# Initialize provider with API key
api_key = os.environ.get('DATABENTO_API_KEY')  # Set your API key
provider = DatabentoProvider(api_key=api_key)

# 1. Fetch historical prices
spy_data = provider.get_historical_prices(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2020-12-31',
    dataset='XNAS.ITCH'
)
print(spy_data.head())

# 2. Fetch VIX data for volatility
vix_data = provider.get_vix_data(
    start_date='2020-01-01',
    end_date='2020-12-31'
)
print(vix_data.head())

# 3. Get option chain
options = provider.get_option_chain(
    symbol='SPY',
    date='2024-01-15',
    dataset='OPRA'
)
print(options.head())

# 4. Backtest with real data
from hedge_calculator import DeepHedgeCalculator

calculator = DeepHedgeCalculator(
    portfolio_value=10_000_000,
    risk_free_rate=0.045,
    annual_volatility=0.18
)

enhancer = DatabentoBacktestEnhancer(provider)

crash_periods = [
    {'name': '2020 COVID', 'start': '2020-02-19', 'end': '2020-03-23'},
    {'name': '2022 Bear', 'start': '2022-01-03', 'end': '2022-10-13'}
]

results = enhancer.backtest_with_real_data(
    symbol='SPY',
    crash_periods=crash_periods,
    hedge_calculator=calculator,
    dataset='XNAS.ITCH'
)

print(results)
"""


if __name__ == "__main__":
    if DATABENTO_AVAILABLE:
        print("Databento provider module loaded successfully!")
        print("\nSample usage:")
        print(get_sample_usage())
    else:
        print("Databento package not installed.")
        print("Install with: pip install databento")
        print("\nAfter installation, you can use this module to fetch real market data.")
