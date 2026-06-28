# Monte Carlo VaR/CVaR Analysis

A modular Python library for calculating Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) using multiple Monte Carlo simulation methods.

## Features

- **4 Simulation Methods**:
  - Cholesky decomposition (Normal distribution)
  - Cholesky decomposition (Empirical/Bootstrap)
  - Principal Component Analysis (Normal)
  - Principal Component Analysis (Empirical)

- **Data Integration**: Automatic data fetching from Yahoo Finance via `yfinance`
- **Flexible Configuration**: Easy asset and position customization
- **Comprehensive Reporting**: Summary tables, CSV exports, and visualizations

## Installation

```bash
git clone https://github.com/alessavargas/montecarlo.git
cd montecarlo
pip install -r requirements.txt
```

## Quick Start

Edit `src/config.py` with your assets and positions:

```python
CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "positions": {
        "AAPL": 1000,
        "MSFT": 500,
        "GOOGL": 750,
    },
    "confidence_level": 0.99,
    "num_simulations": 10000,
}
```

Run the analysis:

```bash
python main.py
```

## Structure

```
montecarlo/
├── main.py                 # Main pipeline
├── requirements.txt        # Dependencies
├── README.md              # This file
└── src/
    ├── __init__.py        # Package initialization
    ├── config.py          # Configuration
    ├── data_loader.py     # Data fetching & processing
    ├── simulators.py      # MC simulation methods
    ├── var_calculator.py  # VaR/CVaR calculations
    └── reporting.py       # Results reporting
```

## Simulation Methods

### Cholesky - Normal
- Uses historical covariance structure
- Assumes normally distributed returns
- Fastest method

### Cholesky - Empirical
- Preserves historical distribution properties
- Uses bootstrap sampling
- Captures fat tails and skewness

### PCA - Normal
- Decomposes correlation into principal components
- Reduces dimensionality
- Captures common risk factors

### PCA - Empirical
- Combines PCA with bootstrap sampling
- Best for capturing complex dependencies
- Computationally intensive

## License

MIT

## Author

Alessa Vargas (@alessavargas)
