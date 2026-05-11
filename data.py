"""
data.py
Module for downloading and preprocessing financial data.
"""
import yfinance as yf
import numpy as np
import pandas as pd

def download_data(tickers, start, end):
    """Download adjusted close prices for given tickers from Yahoo Finance."""
    data = yf.download(tickers, start=start, end=end)
    # yfinance returns MultiIndex columns: (Price, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data['Close'].to_frame()
    close = close.ffill()
    close = close[tickers]  # Ensure order matches input tickers
    return close

def compute_log_returns(prices):
    """Compute daily log returns from price data."""
    return np.log(prices / prices.shift(1)).dropna()

def compute_cov_corr(returns):
    """Compute covariance and correlation matrices from returns."""
    cov = returns.cov()
    corr = returns.corr()
    return cov, corr
