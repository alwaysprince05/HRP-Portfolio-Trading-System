"""
backtest.py
Module for portfolio backtesting and performance metrics.
"""
import numpy as np
import pandas as pd

def backtest_portfolio(returns, weights):
    """Compute portfolio returns and cumulative returns."""
    port_rets = returns @ weights
    cum_rets = (1 + port_rets).cumprod()
    return port_rets, cum_rets

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    """Compute annualized Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()

def max_drawdown(cum_returns):
    """Compute maximum drawdown from cumulative returns."""
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    return drawdown.min(), drawdown
