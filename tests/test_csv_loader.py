"""Tests for CSV data loader."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data.csv_loader import CSVLoader, validate_csv_data


def test_csv_loader_basic(temp_csv_file):
    """Test basic CSV loading."""
    loader = CSVLoader(temp_csv_file)
    df = loader.load()
    
    assert isinstance(df, pd.DataFrame)
    assert 'close' in df.columns
    assert 'volume' in df.columns
    assert df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex)
    assert len(df) > 0


def test_csv_loader_missing_file():
    """Test loading non-existent file."""
    loader = CSVLoader('nonexistent.csv')
    
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_csv_loader_custom_columns(temp_csv_file):
    """Test loading with custom column names."""
    loader = CSVLoader(
        temp_csv_file,
        date_column='date',
        price_column='close',
        volume_column='volume'
    )
    df = loader.load()
    
    assert 'close' in df.columns
    assert 'volume' in df.columns


def test_validate_csv_data(sample_dates):
    """Test data validation."""
    # Create data with issues
    n = len(sample_dates)
    data = {
        'close': [100.0] * (n - 3) + [np.nan, -5.0, 0.0],
        'volume': [1e6] * (n - 1) + [np.nan]
    }
    df = pd.DataFrame(data, index=sample_dates)
    
    # Validate
    cleaned = validate_csv_data(df, 'TEST')
    
    # Should remove invalid prices
    assert len(cleaned) < len(df)
    assert all(cleaned['close'] > 0)
    
    # Volume NaNs should be filled with 0
    assert cleaned['volume'].isna().sum() == 0


def test_csv_loader_date_parsing(tmp_path):
    """Test date parsing."""
    # Create CSV with specific date format
    csv_path = tmp_path / 'test.csv'
    
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Price': np.arange(100, 110),
        'Vol': np.arange(1000, 1010)
    })
    data.to_csv(csv_path, index=False)
    
    # Load with custom column names
    loader = CSVLoader(
        csv_path,
        date_column='Date',
        price_column='Price',
        volume_column='Vol'
    )
    df = loader.load()
    
    assert len(df) == 10
    assert isinstance(df.index, pd.DatetimeIndex)


def test_csv_loader_no_volume(tmp_path):
    """Test loading CSV without volume column."""
    csv_path = tmp_path / 'test.csv'
    
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'price': np.arange(100, 110)
    })
    data.to_csv(csv_path, index=False)
    
    loader = CSVLoader(
        csv_path,
        date_column='date',
        price_column='price',
        volume_column='volume'  # Doesn't exist
    )
    df = loader.load()
    
    assert len(df) == 10
    assert 'volume' in df.columns
    assert all(df['volume'] == 0.0)


def test_csv_loader_duplicates(tmp_path):
    """Test handling of duplicate dates."""
    csv_path = tmp_path / 'test.csv'
    
    # Create data with duplicates
    dates = ['2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03']
    data = pd.DataFrame({
        'date': dates,
        'close': [100, 101, 102, 103],
        'volume': [1000, 1001, 1002, 1003]
    })
    data.to_csv(csv_path, index=False)
    
    loader = CSVLoader(csv_path)
    df = loader.load()
    
    # Should keep only unique dates (last occurrence)
    assert len(df) == 3
    assert df.loc['2020-01-02', 'close'] == 102


def test_csv_loader_load_multiple(tmp_path):
    """Test loading multiple CSV files."""
    # Create multiple CSV files
    file_paths = {}
    
    for ticker in ['TICKER1', 'TICKER2']:
        csv_path = tmp_path / f'{ticker}.csv'
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': np.arange(100, 110),
            'volume': np.arange(1000, 1010)
        })
        data.to_csv(csv_path, index=False)
        file_paths[ticker] = str(csv_path)
    
    # Load multiple
    results = CSVLoader.load_multiple(file_paths)
    
    assert len(results) == 2
    assert 'TICKER1' in results
    assert 'TICKER2' in results
    assert len(results['TICKER1']) == 10
    assert len(results['TICKER2']) == 10
