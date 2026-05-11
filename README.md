# HRP Portfolio Trading System

## Maintainer
alwaysprince05

## What is this project?
This project is a Python-based quantitative trading system that implements Hierarchical Risk Parity (HRP) portfolio allocation. It allows you to:
- Download historical price data for a mix of stocks and ETFs using yfinance
- Compute daily log returns, covariance, and correlation matrices
- Allocate portfolio weights using HRP, Equal Weight, and Mean-Variance (Markowitz) methods
- Backtest the strategies and compute performance metrics (Sharpe ratio, max drawdown)
- Visualize cumulative returns, drawdowns, and the correlation heatmap

## How to fork and run
1. **Fork this repository** using the GitHub interface (click the "Fork" button at the top right of the repo page).
2. **Clone your fork** to your local machine:
   ```sh
   git clone https://github.com/YOUR_USERNAME/HRP-Portfolio-Trading-System.git
   cd HRP-Portfolio-Trading-System
   ```
3. **Install dependencies** (preferably in a virtual environment):
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the main script:**
   ```sh
   python main.py
   ```

## Relevant Wikipedia links
- [Hierarchical Risk Parity](https://en.wikipedia.org/wiki/Hierarchical_risk_parity)
- [Mean-variance optimization](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)
- [Drawdown (economics)](https://en.wikipedia.org/wiki/Drawdown_(economics))
- [Backtesting](https://en.wikipedia.org/wiki/Backtesting)
- [yfinance](https://github.com/ranaroussi/yfinance)

---

This project is maintained by alwaysprince05 and is shared for educational and research purposes.
