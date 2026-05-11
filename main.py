"""
main.py
Main script to run the HRP portfolio system and backtest.
"""
import numpy as np
import pandas as pd
from data import download_data, compute_log_returns, compute_cov_corr
from hrp import hrp_allocation
from backtest import backtest_portfolio, sharpe_ratio, max_drawdown
from visualize import plot_all_together

# 1. Data
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'GLD', 'TLT', 'SPY']
START = '2018-01-01'
END = '2024-01-01'

prices = download_data(TICKERS, START, END)
returns = compute_log_returns(prices)
cov, corr = compute_cov_corr(returns)

# 2. Portfolio Weights
# HRP
hrp_weights = hrp_allocation(cov, corr)
# Equal Weight
ew_weights = pd.Series(1/len(TICKERS), index=TICKERS)
# Mean-Variance (Markowitz)
def mean_variance_weights(cov):
    inv = np.linalg.pinv(cov.values)
    ones = np.ones(len(cov))
    w = inv @ ones
    w /= w.sum()
    return pd.Series(w, index=cov.index)
mv_weights = mean_variance_weights(cov)

# 3. Backtest
hrp_rets, hrp_cum = backtest_portfolio(returns, hrp_weights)
ew_rets, ew_cum = backtest_portfolio(returns, ew_weights)
mv_rets, mv_cum = backtest_portfolio(returns, mv_weights)

# 4. Metrics
hrp_sharpe = sharpe_ratio(hrp_rets)
ew_sharpe = sharpe_ratio(ew_rets)
mv_sharpe = sharpe_ratio(mv_rets)
hrp_mdd, hrp_dd = max_drawdown(hrp_cum)
ew_mdd, ew_dd = max_drawdown(ew_cum)
mv_mdd, mv_dd = max_drawdown(mv_cum)

print("Sharpe Ratios:")
print(f"HRP: {hrp_sharpe:.2f}, Equal Weight: {ew_sharpe:.2f}, Mean-Variance: {mv_sharpe:.2f}")
print("Max Drawdowns:")
print(f"HRP: {hrp_mdd:.2%}, Equal Weight: {ew_mdd:.2%}, Mean-Variance: {mv_mdd:.2%}")

# 5. Visualization
plot_all_together(
    {'HRP': hrp_cum, 'Equal Weight': ew_cum, 'Mean-Variance': mv_cum},
    {'HRP': hrp_dd, 'Equal Weight': ew_dd, 'Mean-Variance': mv_dd},
    corr
)
