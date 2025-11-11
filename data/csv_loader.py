"""CSV data loader with flexible column mapping and date parsing."""

import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path


class CSVLoader:
    """Load price and volume data from CSV files with flexible column mapping."""
    
    def __init__(
        self,
        file_path: str,
        date_column: str = "date",
        price_column: str = "close",
        volume_column: Optional[str] = "volume",
        date_format: Optional[str] = None
    ):
        """
        Initialize CSV loader.
        
        Args:
            file_path: Path to CSV file
            date_column: Name of date column
            price_column: Name of price column
            volume_column: Name of volume column (optional)
            date_format: Date format string (optional, will auto-detect if None)
        """
        self.file_path = Path(file_path)
        self.date_column = date_column
        self.price_column = price_column
        self.volume_column = volume_column
        self.date_format = date_format
        
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with columns: date (index), close, volume (if available)
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        # Read CSV
        df = pd.read_csv(self.file_path)
        
        # Check required columns exist
        if self.date_column not in df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in CSV")
        if self.price_column not in df.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in CSV")
        
        # Parse dates
        if self.date_format:
            df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format)
        else:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Create output dataframe
        result = pd.DataFrame({
            'date': df[self.date_column],
            'close': df[self.price_column].astype(float)
        })
        
        # Add volume if available
        if self.volume_column and self.volume_column in df.columns:
            result['volume'] = df[self.volume_column].astype(float)
        else:
            result['volume'] = 0.0
        
        # Set date as index
        result = result.set_index('date').sort_index()
        
        # Remove duplicates, keep last
        result = result[~result.index.duplicated(keep='last')]
        
        return result
    
    @staticmethod
    def load_multiple(
        file_paths: Dict[str, str],
        date_column: str = "date",
        price_column: str = "close",
        volume_column: Optional[str] = "volume",
        date_format: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple CSV files.
        
        Args:
            file_paths: Dictionary mapping ticker to file path
            date_column: Name of date column
            price_column: Name of price column
            volume_column: Name of volume column (optional)
            date_format: Date format string (optional)
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker, file_path in file_paths.items():
            try:
                loader = CSVLoader(
                    file_path=file_path,
                    date_column=date_column,
                    price_column=price_column,
                    volume_column=volume_column,
                    date_format=date_format
                )
                results[ticker] = loader.load()
            except Exception as e:
                print(f"Warning: Failed to load {ticker} from {file_path}: {e}")
        
        return results


def validate_csv_data(df: pd.DataFrame, ticker: str = "Unknown") -> pd.DataFrame:
    """
    Validate and clean CSV data.
    
    Args:
        df: Input DataFrame
        ticker: Ticker name for logging
        
    Returns:
        Cleaned DataFrame
    """
    # Remove NaN prices
    n_before = len(df)
    df = df.dropna(subset=['close'])
    n_after = len(df)
    
    if n_after < n_before:
        print(f"Warning: Removed {n_before - n_after} rows with NaN prices for {ticker}")
    
    # Remove zero or negative prices
    n_before = len(df)
    df = df[df['close'] > 0]
    n_after = len(df)
    
    if n_after < n_before:
        print(f"Warning: Removed {n_before - n_after} rows with invalid prices for {ticker}")
    
    # Fill NaN volumes with 0
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)
    
    return df
