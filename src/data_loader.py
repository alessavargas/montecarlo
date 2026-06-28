"""Data loading and preprocessing"""

import numpy as np
import pandas as pd
import yfinance as yf


class DataLoader:
    """Fetch and process financial data from yfinance"""
    
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
    
    def fetch_and_process(self):
        """
        Fetch price data and calculate log returns
        Returns: (prices_df, returns_df)
        """
        print(f"Fetching data for {len(self.tickers)} tickers...")
        
        try:
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            prices_df = data["Close"].copy()
            prices_df.dropna(inplace=True)
            
            if prices_df.empty:
                raise ValueError("No data retrieved from yfinance")
            
            # Calculate log returns
            log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
            
            print(f"Data shape: {log_returns.shape}")
            print(f"Date range: {log_returns.index[0].date()} to {log_returns.index[-1].date()}")
            
            return prices_df, log_returns
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise


class StatisticalSummary:
    """Calculate statistical properties of returns"""
    
    @staticmethod
    def describe(returns_df):
        """Print statistical summary"""
        print("\nReturn Statistics:")
        print(returns_df.describe())
        print("\nCorrelation Matrix:")
        print(returns_df.corr())
