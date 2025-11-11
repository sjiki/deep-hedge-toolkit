"""Execution cost models for VIX ETP trading including slippage and borrow costs."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class ExecutionCostModel:
    """Model dynamic slippage (spread + participation impact) and borrow costs."""
    
    def __init__(
        self,
        base_spread_bps: float = 10.0,
        participation_impact: float = 0.001,
        borrow_costs_annual: Optional[Dict[str, float]] = None
    ):
        """
        Initialize execution cost model.
        
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            participation_impact: Market impact coefficient (impact = coef * |trade| / volume)
            borrow_costs_annual: Dictionary mapping ticker to annual borrow cost rate
        """
        self.base_spread_bps = base_spread_bps
        self.participation_impact = participation_impact
        self.borrow_costs_annual = borrow_costs_annual or {}
        
    def compute_slippage(
        self,
        trade_values: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Compute total slippage cost.
        
        Args:
            trade_values: Dollar value of trades
            prices: Current prices
            volumes: Trading volumes (optional, for participation impact)
            
        Returns:
            Series of slippage costs (as fraction of trade value)
        """
        # Base spread cost (half-spread)
        spread_cost = self.base_spread_bps / 2 / 10000
        
        # Participation impact
        if volumes is not None and self.participation_impact > 0:
            # Estimate participation rate
            shares_traded = np.abs(trade_values) / prices
            participation_rate = shares_traded / (volumes + 1)  # +1 to avoid division by zero
            impact_cost = self.participation_impact * participation_rate
        else:
            impact_cost = 0.0
        
        total_slippage = spread_cost + impact_cost
        
        return pd.Series(total_slippage, index=trade_values.index)
    
    def compute_borrow_cost(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        days_held: float = 1.0
    ) -> pd.Series:
        """
        Compute borrow costs for long positions.
        
        Args:
            positions: DataFrame of position weights by ticker
            prices: DataFrame of prices by ticker
            days_held: Number of days positions are held
            
        Returns:
            Series of borrow costs (as fraction of portfolio value)
        """
        total_cost = pd.Series(0.0, index=positions.index)
        
        for ticker in positions.columns:
            if ticker in self.borrow_costs_annual:
                annual_rate = self.borrow_costs_annual[ticker]
                daily_rate = annual_rate / 252
                
                # Cost = position_value * daily_rate * days_held
                position_value = positions[ticker]
                cost = position_value * daily_rate * days_held
                total_cost += cost
        
        return total_cost
    
    def compute_total_transaction_cost(
        self,
        old_positions: pd.Series,
        new_positions: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        portfolio_value: float = 1.0
    ) -> float:
        """
        Compute total transaction cost for rebalancing.
        
        Args:
            old_positions: Old position weights
            new_positions: New position weights
            prices: Current prices
            volumes: Trading volumes (optional)
            portfolio_value: Total portfolio value
            
        Returns:
            Total transaction cost as fraction of portfolio value
        """
        # Align positions
        all_tickers = old_positions.index.union(new_positions.index)
        old_pos = old_positions.reindex(all_tickers, fill_value=0.0)
        new_pos = new_positions.reindex(all_tickers, fill_value=0.0)
        
        # Compute trades
        trades = new_pos - old_pos
        trade_values = np.abs(trades) * portfolio_value
        
        # Compute slippage
        slippage_rates = self.compute_slippage(trade_values, prices, volumes)
        slippage_cost = (slippage_rates * np.abs(trades)).sum()
        
        return slippage_cost


class DynamicSlippageModel(ExecutionCostModel):
    """Enhanced slippage model with volume-dependent impact."""
    
    def __init__(
        self,
        base_spread_bps: float = 10.0,
        participation_impact: float = 0.001,
        borrow_costs_annual: Optional[Dict[str, float]] = None,
        volume_threshold: float = 0.01
    ):
        """
        Initialize dynamic slippage model.
        
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            participation_impact: Market impact coefficient
            borrow_costs_annual: Dictionary mapping ticker to annual borrow cost rate
            volume_threshold: Participation rate threshold for increased impact
        """
        super().__init__(base_spread_bps, participation_impact, borrow_costs_annual)
        self.volume_threshold = volume_threshold
    
    def compute_slippage(
        self,
        trade_values: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Compute slippage with increased impact for large trades.
        
        Args:
            trade_values: Dollar value of trades
            prices: Current prices
            volumes: Trading volumes (optional)
            
        Returns:
            Series of slippage costs (as fraction of trade value)
        """
        # Base spread cost
        spread_cost = self.base_spread_bps / 2 / 10000
        
        if volumes is not None and self.participation_impact > 0:
            # Calculate participation rate
            shares_traded = np.abs(trade_values) / prices
            participation_rate = shares_traded / (volumes + 1)
            
            # Increased impact for large trades
            impact_multiplier = np.where(
                participation_rate > self.volume_threshold,
                1.0 + (participation_rate - self.volume_threshold) * 10,
                1.0
            )
            
            impact_cost = self.participation_impact * participation_rate * impact_multiplier
        else:
            impact_cost = 0.0
        
        total_slippage = spread_cost + impact_cost
        
        return pd.Series(total_slippage, index=trade_values.index)


def create_execution_model(
    config: Dict,
    model_type: str = 'dynamic'
) -> ExecutionCostModel:
    """
    Factory function to create execution cost model from config.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model ('basic' or 'dynamic')
        
    Returns:
        ExecutionCostModel instance
    """
    exec_config = config.get('execution', {})
    
    base_spread_bps = exec_config.get('base_spread_bps', 10.0)
    participation_impact = exec_config.get('participation_impact', 0.001)
    borrow_costs = exec_config.get('borrow_cost_annual', {})
    
    if model_type == 'dynamic':
        return DynamicSlippageModel(
            base_spread_bps=base_spread_bps,
            participation_impact=participation_impact,
            borrow_costs_annual=borrow_costs,
            volume_threshold=exec_config.get('volume_threshold', 0.01)
        )
    else:
        return ExecutionCostModel(
            base_spread_bps=base_spread_bps,
            participation_impact=participation_impact,
            borrow_costs_annual=borrow_costs
        )
