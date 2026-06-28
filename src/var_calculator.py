"""VaR and CVaR calculation"""

import numpy as np
import pandas as pd


class VaRCalculator:
    """Calculate Value-at-Risk and Conditional Value-at-Risk"""
    
    def __init__(self, confidence_level=0.99, horizon_days=1, num_simulations=10000):
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
        self.num_simulations = num_simulations
        self.alpha = 1 - confidence_level
    
    def calculate_pnl(self, simulated_returns, spot_prices, positions):
        """
        Calculate P&L from simulated returns
        
        Args:
            simulated_returns: (N, m) array of simulated returns
            spot_prices: (m,) Series of current prices
            positions: dict {ticker: quantity}
        
        Returns:
            (N,) array of portfolio P&L
        """
        # Portfolio value at each position
        portfolio_values = np.array([
            spot_prices[ticker] * positions[ticker]
            for ticker in positions.keys()
        ])
        
        # P&L by scenario
        pnl_by_asset = simulated_returns * portfolio_values
        
        # Total portfolio P&L
        total_pnl = pnl_by_asset.sum(axis=1)
        
        return total_pnl
    
    def calculate_var_cvar(self, pnl_array):
        """
        Calculate VaR and CVaR
        
        Args:
            pnl_array: (N,) array of P&L values
        
        Returns:
            (var_value, cvar_value)
        """
        # VaR is the negative loss at alpha percentile
        var_loss = np.percentile(pnl_array, self.alpha * 100)
        var_value = -var_loss
        
        # CVaR is the mean of losses worse than VaR
        tail_losses = pnl_array[pnl_array <= var_loss]
        cvar_value = -tail_losses.mean() if len(tail_losses) > 0 else var_value
        
        return var_value, cvar_value
    
    def calculate_by_asset(self, simulated_returns, spot_prices, positions):
        """Calculate VaR/CVaR by individual asset"""
        results = {}
        
        for i, (ticker, quantity) in enumerate(positions.items()):
            asset_pnl = simulated_returns[:, i] * spot_prices[ticker] * quantity
            var, cvar = self.calculate_var_cvar(asset_pnl)
            results[ticker] = {"VaR": var, "CVaR": cvar}
        
        return results
