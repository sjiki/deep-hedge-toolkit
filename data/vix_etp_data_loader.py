"""VIX ETP data loader with fallback logic: CSV → Polygon → yfinance."""

import pandas as pd
import os
from typing import Optional, Dict, List, Tuple
import yfinance as yf

from data.csv_loader import CSVLoader, validate_csv_data
from data.polygon_enhanced_loader import PolygonEnhancedLoader


class VIXETPDataLoader:
    """Main data loader with prioritized fallback: CSV → Polygon → yfinance."""
    
    def __init__(
        self,
        tickers: List[str],
        date_from: str = "2020-01-01",
        date_to: Optional[str] = None,
        polygon_api_key: Optional[str] = None,
        csv_sources: Optional[Dict[str, str]] = None
    ):
        """
        Initialize VIX ETP data loader.
        
        Args:
            tickers: List of ticker symbols to load
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD), None for today
            polygon_api_key: Polygon.io API key (optional)
            csv_sources: Dictionary mapping ticker to CSV file path (optional)
        """
        self.tickers = tickers
        self.date_from = date_from
        self.date_to = date_to
        self.polygon_api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY")
        self.csv_sources = csv_sources or {}
        
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load price and volume data with fallback logic.
        
        Returns:
            Tuple of (price_df, volume_df) with tickers as columns
        """
        prices = {}
        volumes = {}
        
        # Try CSV sources first
        csv_loaded = self._load_from_csv(prices, volumes)
        
        # For tickers not loaded from CSV, try Polygon
        remaining_tickers = [t for t in self.tickers if t not in prices]
        if remaining_tickers and self.polygon_api_key:
            polygon_loaded = self._load_from_polygon(remaining_tickers, prices, volumes)
        else:
            polygon_loaded = set()
        
        # For tickers still not loaded, try yfinance
        remaining_tickers = [t for t in self.tickers if t not in prices]
        if remaining_tickers:
            self._load_from_yfinance(remaining_tickers, prices, volumes)
        
        # Create wide DataFrames
        price_df = pd.DataFrame(prices)
        volume_df = pd.DataFrame(volumes)
        
        # Align indices and fill missing values
        if not price_df.empty:
            price_df = price_df.sort_index()
            volume_df = volume_df.sort_index()
            
            # Forward fill then backward fill for prices
            price_df = price_df.ffill().bfill()
            
            # Fill volumes with 0
            volume_df = volume_df.fillna(0)
        
        return price_df, volume_df
    
    def _load_from_csv(
        self,
        prices: Dict[str, pd.Series],
        volumes: Dict[str, pd.Series]
    ) -> set:
        """Load data from CSV sources."""
        loaded = set()
        
        for ticker, csv_path in self.csv_sources.items():
            if ticker not in self.tickers:
                continue
                
            try:
                print(f"Loading {ticker} from CSV: {csv_path}")
                loader = CSVLoader(csv_path)
                df = loader.load()
                df = validate_csv_data(df, ticker)
                
                if not df.empty:
                    # Filter date range
                    df = df[df.index >= self.date_from]
                    if self.date_to:
                        df = df[df.index <= self.date_to]
                    
                    prices[ticker] = df['close']
                    volumes[ticker] = df.get('volume', pd.Series(0, index=df.index))
                    loaded.add(ticker)
                    print(f"  ✓ Loaded {len(df)} rows from CSV")
            except Exception as e:
                print(f"  ✗ Failed to load {ticker} from CSV: {e}")
        
        return loaded
    
    def _load_from_polygon(
        self,
        tickers: List[str],
        prices: Dict[str, pd.Series],
        volumes: Dict[str, pd.Series]
    ) -> set:
        """Load data from Polygon.io."""
        if not self.polygon_api_key:
            return set()
        
        loaded = set()
        
        try:
            print(f"Loading {len(tickers)} tickers from Polygon.io...")
            loader = PolygonEnhancedLoader(self.polygon_api_key)
            data_dict = loader.load_multiple(tickers, self.date_from, self.date_to)
            
            for ticker, df in data_dict.items():
                if not df.empty and 'close' in df.columns:
                    prices[ticker] = df['close']
                    volumes[ticker] = df.get('volume', pd.Series(0, index=df.index))
                    loaded.add(ticker)
                    print(f"  ✓ Loaded {ticker}: {len(df)} rows from Polygon")
        except Exception as e:
            print(f"  ✗ Failed to load from Polygon: {e}")
        
        return loaded
    
    def _load_from_yfinance(
        self,
        tickers: List[str],
        prices: Dict[str, pd.Series],
        volumes: Dict[str, pd.Series]
    ):
        """Load data from yfinance as final fallback."""
        print(f"Loading {len(tickers)} tickers from yfinance...")
        
        for ticker in tickers:
            try:
                print(f"  Loading {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.date_from, end=self.date_to)
                
                if not df.empty:
                    prices[ticker] = df['Close']
                    volumes[ticker] = df.get('Volume', pd.Series(0, index=df.index))
                    print(f"    ✓ Loaded {len(df)} rows from yfinance")
                else:
                    print(f"    ✗ No data returned for {ticker}")
            except Exception as e:
                print(f"    ✗ Failed to load {ticker}: {e}")


def load_vix_etp_data(
    tickers: Optional[List[str]] = None,
    date_from: str = "2020-01-01",
    date_to: Optional[str] = None,
    polygon_api_key: Optional[str] = None,
    csv_sources: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load VIX ETP data.
    
    Args:
        tickers: List of ticker symbols (default: UVXY, SVXY, VXX, VIXY)
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD), None for today
        polygon_api_key: Polygon.io API key (optional)
        csv_sources: Dictionary mapping ticker to CSV file path (optional)
        
    Returns:
        Tuple of (price_df, volume_df) with tickers as columns
    """
    if tickers is None:
        tickers = ["UVXY", "SVXY", "VXX", "VIXY"]
    
    loader = VIXETPDataLoader(
        tickers=tickers,
        date_from=date_from,
        date_to=date_to,
        polygon_api_key=polygon_api_key,
        csv_sources=csv_sources
    )
    
    return loader.load()
