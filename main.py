"""
Monte Carlo VaR/CVaR Analysis
Multi-method comparison: Cholesky (Normal/Empirical), PCA (Normal/Empirical)
Data sourced from yfinance
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from src.config import CONFIG
from src.data_loader import DataLoader
from src.simulators import (
    CholeskyNormal,
    CholeskyEmpirical,
    PCANormal,
    PCAEmpirical,
)
from src.var_calculator import VaRCalculator
from src.reporting import Reporter


def main():
    """Main pipeline for Monte Carlo VaR/CVaR analysis"""
    
    print("=" * 80)
    print("MONTE CARLO VaR/CVaR ANALYSIS - Multi-Method Comparison")
    print("=" * 80)
    
    # Load data
    loader = DataLoader(
        tickers=CONFIG["tickers"],
        start_date=CONFIG["start_date"],
        end_date=CONFIG["valuation_date"]
    )
    
    prices_df, returns_df = loader.fetch_and_process()
    
    print(f"\nData loaded: {len(returns_df)} observations")
    print(f"Tickers: {CONFIG['tickers']}")
    
    # Instantiate simulators
    simulators = {
        "Cholesky - Normal": CholeskyNormal(),
        "Cholesky - Empirical": CholeskyEmpirical(),
        "PCA - Normal": PCANormal(),
        "PCA - Empirical": PCAEmpirical(),
    }
    
    # VaR calculator
    var_calc = VaRCalculator(
        confidence_level=CONFIG["confidence_level"],
        horizon_days=CONFIG["horizon_days"],
        num_simulations=CONFIG["num_simulations"]
    )
    
    # Run simulations and calculate VaR/CVaR
    results = {}
    for method_name, simulator in simulators.items():
        print(f"\nRunning {method_name}...")
        
        simulated_returns = simulator.simulate(returns_df, CONFIG["num_simulations"])
        pnl = var_calc.calculate_pnl(
            simulated_returns,
            prices_df.iloc[-1],
            CONFIG["positions"]
        )
        
        var_values, cvar_values = var_calc.calculate_var_cvar(pnl)
        results[method_name] = {
            "VaR": var_values,
            "CVaR": cvar_values,
            "PnL": pnl
        }
    
    # Generate reports
    reporter = Reporter(results, CONFIG)
    reporter.print_summary()
    reporter.export_to_csv("results.csv")
    reporter.plot_distributions()
    
    print("\n" + "=" * 80)
    print("Analysis complete. Results saved.")


if __name__ == "__main__":
    main()
