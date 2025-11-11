"""Enhanced Polygon.io data loader for VIX ETPs with wide price and volume data."""

import pandas as pd
import requests
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import time


class PolygonEnhancedLoader:
    """Load daily aggregate price and volume data from Polygon.io API."""
    
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize Polygon loader.
        
        Args:
            api_key: Polygon.io API key
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        
    def load_ticker(
        self,
        ticker: str,
        date_from: str,
        date_to: Optional[str] = None,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Load daily data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD), None for today
            adjusted: Whether to use adjusted prices
            
        Returns:
            DataFrame with columns: date (index), open, high, low, close, volume
        """
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/{ticker}/range/1/day/{date_from}/{date_to}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "OK":
                    raise ValueError(f"API returned status: {data.get('status')}")
                
                results = data.get("results", [])
                if not results:
                    print(f"Warning: No data returned for {ticker}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Rename columns to standard format
                df = df.rename(columns={
                    't': 'timestamp',
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    'vw': 'vwap',
                    'n': 'transactions'
                })
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')
                
                # Select and order columns
                columns = ['open', 'high', 'low', 'close', 'volume']
                df = df[[col for col in columns if col in df.columns]]
                
                return df.sort_index()
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {ticker}: {e}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"Failed to load {ticker} after {self.max_retries} attempts: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def load_multiple(
        self,
        tickers: List[str],
        date_from: str,
        date_to: Optional[str] = None,
        adjusted: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD), None for today
            adjusted: Whether to use adjusted prices
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker in tickers:
            print(f"Loading {ticker}...")
            df = self.load_ticker(ticker, date_from, date_to, adjusted)
            if not df.empty:
                results[ticker] = df
            time.sleep(0.2)  # Rate limiting
        
        return results
    
    def create_wide_dataframe(
        self,
        data_dict: Dict[str, pd.DataFrame],
        price_col: str = 'close',
        volume_col: str = 'volume'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create wide-format DataFrames for prices and volumes.
        
        Args:
            data_dict: Dictionary mapping ticker to DataFrame
            price_col: Column to use for prices
            volume_col: Column to use for volumes
            
        Returns:
            Tuple of (price_df, volume_df) with tickers as columns
        """
        if not data_dict:
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract prices
        prices = {}
        volumes = {}
        
        for ticker, df in data_dict.items():
            if price_col in df.columns:
                prices[ticker] = df[price_col]
            if volume_col in df.columns:
                volumes[ticker] = df[volume_col]
        
        # Create wide DataFrames
        price_df = pd.DataFrame(prices)
        volume_df = pd.DataFrame(volumes)
        
        # Fill forward then backward for missing values
        price_df = price_df.ffill().bfill()
        volume_df = volume_df.fillna(0)
        
        return price_df, volume_df


def load_with_polygon(
    tickers: List[str],
    api_key: str,
    date_from: str = "2020-01-01",
    date_to: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load wide price and volume data using Polygon.io.
    
    Args:
        tickers: List of ticker symbols
        api_key: Polygon.io API key
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD), None for today
        
    Returns:
        Tuple of (price_df, volume_df) with tickers as columns
    """
    loader = PolygonEnhancedLoader(api_key)
    data_dict = loader.load_multiple(tickers, date_from, date_to)
    return loader.create_wide_dataframe(data_dict)
