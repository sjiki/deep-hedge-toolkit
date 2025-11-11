"""Regime classifier for VIX market states using LightGBM or logistic regression."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Literal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class RegimeClassifier:
    """Classify market regimes: Calm, BuildUp, Spike, MeanRevert."""
    
    REGIMES = ['Calm', 'BuildUp', 'Spike', 'MeanRevert']
    
    def __init__(
        self,
        model_type: Literal['lightgbm', 'logistic'] = 'lightgbm',
        labeling_params: Optional[Dict] = None,
        model_params: Optional[Dict] = None
    ):
        """
        Initialize regime classifier.
        
        Args:
            model_type: Type of model ('lightgbm' or 'logistic')
            labeling_params: Parameters for heuristic labeling
            model_params: Model-specific hyperparameters
        """
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        self.model_type = model_type
        
        # Default labeling parameters
        self.labeling_params = {
            'calm_vix_threshold': 20,
            'buildup_vix_change': 2.0,
            'spike_vix_level': 30,
            'spike_vix_change': 5.0,
            'meanrevert_from_spike': 25,
        }
        if labeling_params:
            self.labeling_params.update(labeling_params)
        
        # Default model parameters
        if model_type == 'lightgbm':
            self.model_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'objective': 'multiclass',
                'num_class': 4,
                'verbosity': -1
            }
        else:  # logistic
            self.model_params = {
                'max_iter': 1000,
                'C': 1.0,
                'multi_class': 'multinomial',
                'solver': 'lbfgs'
            }
        
        if model_params:
            self.model_params.update(model_params)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_features(self, vix_data: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Create features from VIX data.
        
        Args:
            vix_data: DataFrame with VIX close prices
            window: Lookback window for features
            
        Returns:
            DataFrame with features
        """
        df = pd.DataFrame(index=vix_data.index)
        
        # Ensure we have a 'close' column
        if 'close' in vix_data.columns:
            vix = vix_data['close']
        elif isinstance(vix_data, pd.Series):
            vix = vix_data
        else:
            vix = vix_data.iloc[:, 0]
        
        # Level features
        df['vix_level'] = vix
        df['vix_pct_rank'] = vix.rolling(window * 2).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x), raw=False
        )
        
        # Change features
        df['vix_change_1d'] = vix.diff(1)
        df['vix_change_5d'] = vix.diff(5)
        df['vix_change_pct_1d'] = vix.pct_change(1)
        df['vix_change_pct_5d'] = vix.pct_change(5)
        
        # Volatility features
        df['vix_vol_20d'] = vix.pct_change().rolling(20).std()
        df['vix_vol_5d'] = vix.pct_change().rolling(5).std()
        
        # Moving averages
        df['vix_ma_5'] = vix.rolling(5).mean()
        df['vix_ma_20'] = vix.rolling(20).mean()
        df['vix_vs_ma5'] = vix - df['vix_ma_5']
        df['vix_vs_ma20'] = vix - df['vix_ma_20']
        
        # Trend features
        df['vix_trend_5d'] = (df['vix_ma_5'] - df['vix_ma_5'].shift(5)) / df['vix_ma_5'].shift(5)
        df['vix_rsi_14'] = self._compute_rsi(vix, 14)
        
        # Recent spike indicators
        df['recent_spike'] = (vix.rolling(5).max() > self.labeling_params['spike_vix_level']).astype(int)
        df['days_since_spike'] = self._days_since_condition(vix > self.labeling_params['spike_vix_level'])
        
        return df.bfill().fillna(0)
    
    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _days_since_condition(self, condition: pd.Series) -> pd.Series:
        """Count days since condition was True."""
        result = pd.Series(index=condition.index, dtype=float)
        days_counter = 0
        
        for i, val in enumerate(condition):
            if val:
                days_counter = 0
            else:
                days_counter += 1
            result.iloc[i] = days_counter
        
        return result
    
    def create_labels_heuristic(self, vix_data: pd.DataFrame) -> pd.Series:
        """
        Create regime labels using heuristic rules.
        
        Args:
            vix_data: DataFrame with VIX close prices
            
        Returns:
            Series with regime labels (0=Calm, 1=BuildUp, 2=Spike, 3=MeanRevert)
        """
        # Ensure we have a 'close' column
        if 'close' in vix_data.columns:
            vix = vix_data['close']
        elif isinstance(vix_data, pd.Series):
            vix = vix_data
        else:
            vix = vix_data.iloc[:, 0]
        
        labels = pd.Series(index=vix.index, dtype=int)
        
        vix_change_1d = vix.diff(1)
        vix_5d_max = vix.rolling(5).max()
        
        for i in range(len(vix)):
            current_vix = vix.iloc[i]
            change_1d = vix_change_1d.iloc[i] if i > 0 else 0
            max_5d = vix_5d_max.iloc[i] if i >= 4 else current_vix
            
            # Spike: VIX > threshold AND large 1-day change
            if (current_vix > self.labeling_params['spike_vix_level'] and 
                change_1d > self.labeling_params['spike_vix_change']):
                labels.iloc[i] = 2  # Spike
            
            # MeanRevert: Coming down from recent spike
            elif (max_5d > self.labeling_params['spike_vix_level'] and 
                  current_vix < self.labeling_params['meanrevert_from_spike'] and
                  change_1d < 0):
                labels.iloc[i] = 3  # MeanRevert
            
            # BuildUp: Moderate VIX rising
            elif (current_vix > self.labeling_params['calm_vix_threshold'] and
                  change_1d > self.labeling_params['buildup_vix_change']):
                labels.iloc[i] = 1  # BuildUp
            
            # Calm: Low VIX, stable
            else:
                labels.iloc[i] = 0  # Calm
        
        return labels
    
    def fit(self, features: pd.DataFrame, labels: Optional[pd.Series] = None, vix_data: Optional[pd.DataFrame] = None):
        """
        Fit the regime classifier.
        
        Args:
            features: Feature DataFrame
            labels: Target labels (optional, will use heuristic if None)
            vix_data: VIX data for heuristic labeling (required if labels is None)
        """
        if labels is None:
            if vix_data is None:
                raise ValueError("Either labels or vix_data must be provided")
            labels = self.create_labels_heuristic(vix_data)
        
        # Align features and labels
        valid_idx = features.index.intersection(labels.index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(**self.model_params)
        else:
            self.model = LogisticRegression(**self.model_params)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Series with regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions, index=features.index, name='regime')
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            DataFrame with probability columns (regime_proba_*)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(features)
        probas = self.model.predict_proba(X_scaled)
        
        # Create DataFrame with regime probability columns
        proba_df = pd.DataFrame(
            probas,
            index=features.index,
            columns=[f'regime_proba_{regime}' for regime in self.REGIMES]
        )
        
        return proba_df
    
    def fit_predict(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        vix_data: Optional[pd.DataFrame] = None
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Fit and predict in one call.
        
        Args:
            features: Feature DataFrame
            labels: Target labels (optional, will use heuristic if None)
            vix_data: VIX data for heuristic labeling (required if labels is None)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.fit(features, labels, vix_data)
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        
        return predictions, probabilities
