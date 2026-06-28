"""Configuration file for Monte Carlo analysis"""

from datetime import datetime, timedelta

CONFIG = {
    # Data parameters
    "tickers": ["AAPL", "MSFT", "GOOGL"],  # Replace with your assets
    "valuation_date": datetime(2025, 9, 8),
    "lookback_years": 4,
    
    # Portfolio positions
    "positions": {
        "AAPL": 1000,      # shares
        "MSFT": 500,
        "GOOGL": 750,
    },
    
    # VaR/CVaR parameters
    "confidence_level": 0.99,
    "horizon_days": 1,
    "num_simulations": 10000,
    
    # Simulation parameters
    "random_seed": 42,
    "variance_explained": 0.999,  # For PCA
}

# Calculated dates
CONFIG["start_date"] = CONFIG["valuation_date"] - timedelta(days=365 * CONFIG["lookback_years"])
