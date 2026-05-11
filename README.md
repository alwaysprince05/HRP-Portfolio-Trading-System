# HRP Portfolio Trading System

An interactive portfolio analytics project built with Python and Streamlit using Hierarchical Risk Parity (HRP), Equal Weight, and Mean-Variance allocation methods.

Maintainer: `alwaysprince05`

## Features

- Download historical asset prices with `yfinance`
- Compute log returns, covariance, and correlation matrices
- Build portfolio allocations with:
  - Hierarchical Risk Parity (HRP)
  - Equal Weight
  - Mean-Variance (Markowitz-style inverse covariance)
- Backtest each strategy and compute:
  - Sharpe ratio
  - Maximum drawdown
  - Cumulative growth
- Explore a clean Streamlit dashboard with:
  - Custom ticker/date inputs
  - Performance metrics cards
  - Drawdown and cumulative return charts
  - Weights table and correlation analysis

## Project Structure

```text
HRP-Portfolio-Trading-System/
├── backtest.py          # Backtesting and metrics
├── data.py              # Data download + preprocessing
├── hrp.py               # HRP allocation logic
├── main.py              # Script-based run
├── streamlit_app.py     # Interactive dashboard
├── visualize.py         # Matplotlib visualization helpers
├── requirements.txt
└── README.md
```

## Setup

### 1) Clone repository

```bash
git clone git@github.com:alwaysprince05/HRP-Portfolio-Trading-System.git
cd HRP-Portfolio-Trading-System
```

### 2) Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Run Options

### Option A: Streamlit dashboard (recommended)

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

### Option B: Python script mode

```bash
python main.py
```

This prints metrics in terminal and renders charts.

## Dashboard Usage

1. Enter comma-separated tickers in the sidebar  
2. Select start/end dates  
3. Choose allocation method  
4. Click **Run**

If Yahoo data is blocked due to network/proxy limits, enable **Use simulated data** and run again.

## Methods Used

- **HRP**: Clusters assets by correlation distance and allocates risk recursively
- **Equal Weight**: Same weight to each selected asset
- **Mean-Variance**: Inverse covariance approximation for efficient weighting

## Notes

- This project is for education and research purposes only.
- It is not financial advice.
- Real trading requires risk controls, transaction costs, slippage, and robust validation.
