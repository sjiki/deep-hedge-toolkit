"""Tests for environment builder."""

import pytest
from env.builder import load_config, build_vix_etp_env
from env.vix_etp_env import VIXETPEnv


def test_load_config(temp_config_file):
    """Test loading configuration from YAML."""
    config = load_config(temp_config_file)
    
    assert isinstance(config, dict)
    assert 'data' in config
    assert 'environment' in config
    assert 'agents' in config


def test_build_env_from_config(temp_config_file, sample_price_data, sample_volume_data, monkeypatch):
    """Test building environment from config."""
    # Mock data loading to avoid real API calls
    def mock_load_data(*args, **kwargs):
        return sample_price_data, sample_volume_data
    
    import data.vix_etp_data_loader
    monkeypatch.setattr(data.vix_etp_data_loader, 'load_vix_etp_data', mock_load_data)
    
    config = load_config(temp_config_file)
    
    # Build environment
    env, price_data, volume_data = build_vix_etp_env(config=config)
    
    assert isinstance(env, VIXETPEnv)
    assert not price_data.empty
    assert not volume_data.empty


def test_build_env_validation(temp_config_file, monkeypatch):
    """Test environment validation."""
    # Mock data loading to return empty data
    def mock_load_empty(*args, **kwargs):
        import pandas as pd
        return pd.DataFrame(), pd.DataFrame()
    
    import data.vix_etp_data_loader
    monkeypatch.setattr(data.vix_etp_data_loader, 'load_vix_etp_data', mock_load_empty)
    
    config = load_config(temp_config_file)
    
    # Should raise error for empty data
    with pytest.raises(ValueError, match="Failed to load any price data"):
        build_vix_etp_env(config=config)


def test_env_config_parameters(temp_config_file, sample_price_data, sample_volume_data, monkeypatch):
    """Test that config parameters are properly applied."""
    def mock_load_data(*args, **kwargs):
        return sample_price_data, sample_volume_data
    
    import data.vix_etp_data_loader
    monkeypatch.setattr(data.vix_etp_data_loader, 'load_vix_etp_data', mock_load_data)
    
    config = load_config(temp_config_file)
    env, _, _ = build_vix_etp_env(config=config)
    
    # Check that parameters from config are applied
    assert env.initial_cash == config['environment']['initial_cash']
    assert env.reward_type == config['training']['reward_type']
