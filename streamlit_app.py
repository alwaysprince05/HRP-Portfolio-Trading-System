import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from backtest import backtest_portfolio, max_drawdown, sharpe_ratio
from data import compute_cov_corr, compute_log_returns, download_data
from hrp import hrp_allocation


def mean_variance_weights(cov: pd.DataFrame) -> pd.Series:
    inv = np.linalg.pinv(cov.values)
    ones = np.ones(len(cov))
    w = inv @ ones
    w /= w.sum()
    return pd.Series(w, index=cov.index)


def simulate_prices(tickers: list[str], start: str, end: str, seed: int = 7) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="B")
    if len(idx) < 30:
        idx = pd.date_range(end=end, periods=252, freq="B")

    rng = np.random.default_rng(seed)
    n = len(idx)
    m = len(tickers)
    daily_mu = 0.0003
    daily_sigma = 0.012
    rets = rng.normal(daily_mu, daily_sigma, size=(n, m))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def style_app() -> None:
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 1.2rem;
        }
        .block-container {
            max-width: 1200px;
        }
        .hero-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid #374151;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }
        .hero-title {
            color: #f9fafb;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin-bottom: 0;
        }
        .chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #334155;
            background: #0f172a;
            color: #93c5fd;
            font-size: 0.8rem;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="HRP Portfolio Trading System", layout="wide")
style_app()

# Ensure matplotlib cache is writable (avoids warnings/slowness on some setups)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

with st.sidebar:
    st.header("Control Panel")
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,JPM,GLD,TLT,SPY"
    tickers_raw = st.text_input("Tickers (comma-separated)", value=default_tickers)
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    c1, c2 = st.columns(2)
    start_dt = c1.date_input("Start", value=date(2018, 1, 1))
    end_dt = c2.date_input("End", value=date(2024, 1, 1))

    method = st.selectbox(
        "Allocation method",
        ["HRP", "Equal Weight", "Mean-Variance"],
        help="HRP clusters assets by correlation; Mean-Variance uses inverse covariance approximation.",
    )
    use_sim = st.toggle("Use simulated data (no Yahoo download)", value=False)
    run = st.button("Run", type="primary")
    st.caption("Tip: if market data fails due to network/proxy limits, enable simulated data.")

if not run:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">HRP Portfolio Dashboard</div>
            <p class="hero-subtitle">Choose your assets and settings from the left panel, then click <b>Run</b>.</p>
            <span class="chip">HRP</span>
            <span class="chip">Backtest</span>
            <span class="chip">Risk Metrics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if len(tickers) < 2:
    st.error("Please provide at least 2 tickers.")
    st.stop()

start = start_dt.isoformat()
end = end_dt.isoformat()

with st.spinner("Loading prices..."):
    if use_sim:
        prices = simulate_prices(tickers, start, end)
        data_source = "Simulated"
    else:
        try:
            prices = download_data(tickers, start, end)
            data_source = "Yahoo Finance (yfinance)"
        except Exception as e:
            st.error(
                "Yahoo download failed. This can happen due to rate limits / proxy blocks.\n\n"
                "Enable **Use simulated data** in the sidebar and rerun, or try again later."
            )
            st.exception(e)
            st.stop()

if prices.empty or prices.dropna(how="all").empty:
    st.error("No price data returned for the selected tickers/date range.")
    st.stop()

returns = compute_log_returns(prices)
if returns.empty:
    st.error("Not enough data to compute returns. Try a wider date range.")
    st.stop()

cov, corr = compute_cov_corr(returns)

try:
    if method == "HRP":
        weights = hrp_allocation(cov, corr)
    elif method == "Equal Weight":
        weights = pd.Series(1 / len(tickers), index=tickers)
    else:
        weights = mean_variance_weights(cov)
except Exception as e:
    st.error("Failed to compute weights (likely due to missing/invalid data).")
    st.exception(e)
    st.stop()

port_rets, port_cum = backtest_portfolio(returns, weights)
sr = float(sharpe_ratio(port_rets))
mdd, dd_series = max_drawdown(port_cum)

ann_return = (1 + port_rets.mean()) ** 252 - 1
ann_vol = port_rets.std() * np.sqrt(252)
final_growth = float(port_cum.iloc[-1] - 1)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">Portfolio Performance Overview</div>
        <p class="hero-subtitle">
            Data source: <b>{data_source}</b> &nbsp;|&nbsp; Samples: <b>{len(prices):,}</b> &nbsp;|&nbsp; Method: <b>{method}</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sharpe Ratio", f"{sr:.2f}")
m2.metric("Max Drawdown", f"{mdd:.2%}")
m3.metric("Annualized Return", f"{ann_return:.2%}")
m4.metric("Annualized Volatility", f"{ann_vol:.2%}")
st.progress(min(max((final_growth + 0.5), 0.0), 1.0), text=f"Net Growth: {final_growth:.2%}")

tab1, tab2, tab3 = st.tabs(["Performance", "Portfolio Composition", "Data & Correlations"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cumulative Return")
        st.area_chart(port_cum.rename("Cumulative Return"), use_container_width=True)
    with c2:
        st.subheader("Drawdown")
        st.line_chart(dd_series.rename("Drawdown"), use_container_width=True)
    with st.expander("Daily Return Distribution"):
        st.bar_chart(port_rets.rename("Daily Return"), use_container_width=True)

with tab2:
    st.subheader("Allocation Weights")
    wdf = (
        weights.sort_values(ascending=False)
        .to_frame("weight")
        .assign(weight_pct=lambda x: (x["weight"] * 100).round(2))
    )
    st.dataframe(wdf, use_container_width=True)
    st.bar_chart(wdf["weight"], use_container_width=True)

with tab3:
    st.subheader("Price Snapshot")
    st.dataframe(prices.tail(15), use_container_width=True)
    st.subheader("Correlation Matrix")
    st.dataframe(corr.style.format("{:.2f}").background_gradient(cmap="Blues"), use_container_width=True)

