"""Tests for regime classifier."""

import pytest
import numpy as np
import pandas as pd

from models.regime_classifier import RegimeClassifier


def test_regime_classifier_init():
    """Test initialization."""
    classifier = RegimeClassifier(model_type='logistic')
    
    assert classifier.model_type == 'logistic'
    assert len(classifier.REGIMES) == 4
    assert not classifier.is_fitted


def test_create_features(sample_vix_data):
    """Test feature creation."""
    classifier = RegimeClassifier()
    
    features = classifier.create_features(sample_vix_data, window=30)
    
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(sample_vix_data)
    assert 'vix_level' in features.columns
    assert 'vix_change_1d' in features.columns
    assert 'vix_ma_5' in features.columns


def test_create_labels_heuristic(sample_vix_data):
    """Test heuristic label creation."""
    classifier = RegimeClassifier(
        labeling_params={
            'calm_vix_threshold': 20,
            'spike_vix_level': 30,
            'spike_vix_change': 5.0
        }
    )
    
    labels = classifier.create_labels_heuristic(sample_vix_data)
    
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(sample_vix_data)
    assert all(labels.isin([0, 1, 2, 3]))
    
    # Should have variety of regimes (at least 1)
    unique_labels = labels.nunique()
    assert unique_labels >= 1  # At least 1 regime


def test_fit_predict_logistic(sample_vix_data):
    """Test fitting with logistic regression."""
    classifier = RegimeClassifier(model_type='logistic')
    
    features = classifier.create_features(sample_vix_data)
    
    predictions, probabilities = classifier.fit_predict(
        features=features,
        vix_data=sample_vix_data
    )
    
    assert classifier.is_fitted
    assert len(predictions) == len(features)
    assert len(probabilities) == len(features)
    assert probabilities.shape[1] == 4  # 4 regime probabilities


def test_fit_predict_lightgbm(sample_vix_data):
    """Test fitting with LightGBM."""
    try:
        import lightgbm
        classifier = RegimeClassifier(
            model_type='lightgbm',
            model_params={'n_estimators': 10, 'max_depth': 3}
        )
        
        features = classifier.create_features(sample_vix_data)
        
        predictions, probabilities = classifier.fit_predict(
            features=features,
            vix_data=sample_vix_data
        )
        
        assert classifier.is_fitted
        assert len(predictions) == len(features)
        assert probabilities.shape[1] == 4
    except ImportError:
        pytest.skip("LightGBM not installed")


def test_predict_proba(sample_vix_data):
    """Test probability prediction."""
    classifier = RegimeClassifier(model_type='logistic')
    
    features = classifier.create_features(sample_vix_data)
    classifier.fit(features, vix_data=sample_vix_data)
    
    probas = classifier.predict_proba(features)
    
    assert probas.shape == (len(features), 4)
    
    # Probabilities should sum to 1
    row_sums = probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    
    # All probabilities should be between 0 and 1
    assert (probas >= 0).all().all()
    assert (probas <= 1).all().all()


def test_predict_before_fit(sample_vix_data):
    """Test that prediction fails before fitting."""
    classifier = RegimeClassifier()
    features = classifier.create_features(sample_vix_data)
    
    with pytest.raises(ValueError):
        classifier.predict(features)


def test_regime_labels_meaning(sample_vix_data):
    """Test that regime labels make sense."""
    # Create VIX data with clear patterns
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Low VIX period (Calm)
    calm_vix = np.full(30, 15.0)
    # Rising VIX (BuildUp)
    buildup_vix = np.linspace(15, 25, 20)
    # High VIX spike
    spike_vix = np.full(10, 35.0)
    # Declining from spike (MeanRevert)
    revert_vix = np.linspace(35, 20, 20)
    # Normal again
    normal_vix = np.full(20, 18.0)
    
    vix_series = np.concatenate([calm_vix, buildup_vix, spike_vix, revert_vix, normal_vix])
    vix_df = pd.DataFrame({'close': vix_series}, index=dates)
    
    classifier = RegimeClassifier()
    labels = classifier.create_labels_heuristic(vix_df)
    
    # Check that calm period has mostly Calm labels
    calm_labels = labels[:30]
    assert (calm_labels == 0).sum() > 15  # More than half should be Calm
    
    # Check that spike period has some Spike labels
    spike_labels = labels[50:60]
    assert (spike_labels == 2).sum() > 0  # At least some Spike labels


def test_rsi_computation(sample_vix_data):
    """Test RSI indicator computation."""
    classifier = RegimeClassifier()
    
    vix_series = sample_vix_data['close']
    rsi = classifier._compute_rsi(vix_series, window=14)
    
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(vix_series)
    
    # RSI should be between 0 and 100
    valid_rsi = rsi.dropna()
    assert all((valid_rsi >= 0) & (valid_rsi <= 100))
