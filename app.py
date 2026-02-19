import os
import math
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =============================
# Configuration
# =============================
APP_TITLE = "Portfolio Optimizer & Risk Intelligence (Saudi + Global)"
DATA_DIR = "data"
CACHE_TTL_SECONDS = 60 * 60
TRADING_DAYS = 252  # standard equities convention

DEFAULT_UNIVERSE: Dict[str, str] = {
    # Saudi (Tadawul)
    "Saudi Aramco (2222.SR)": "2222.SR",
    "Al Rajhi Bank (1120.SR)": "1120.SR",
    "SABIC (2010.SR)": "2010.SR",
    "STC (7010.SR)": "7010.SR",
    "TASI Index (^TASI)": "^TASI",
    # Global
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Gold (GLD)": "GLD",
    "US 10Y Treasury (IEF)": "IEF",
    "Crude Oil (CL=F)": "CL=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
}


# =============================
# Utility functions
# =============================
def pct(x: float) -> str:
    return f"{x*100:,.2f}%"


def annualize_mean(daily: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(daily.mean() * periods)


def annualize_vol(daily: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(daily.std(ddof=0) * math.sqrt(periods))


def max_drawdown(cum_index: pd.Series) -> float:
    peak = cum_index.cummax()
    dd = (cum_index / peak) - 1.0
    return float(dd.min())


def normalize_prices_to_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert daily returns into a cumulative index starting at 1.0."""
    return (1.0 + returns).cumprod()


def export_csv(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path)
    except Exception:
        # fail silently: export is "nice to have", not critical for app
        pass


def validate_asset_count(selected: List[str]) -> None:
    if len(selected) < 3:
        st.warning("Select at least 3 assets for meaningful diversification analysis.")
        st.stop()
    if len(selected) > 8:
        st.error("Select 8 assets or fewer to keep the dashboard fast and readable.")
        st.stop()


# =============================
# Data layer
# =============================
def _download_prices_raw(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # prefer Adj Close, fall back to Close
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            out = df["Adj Close"].copy()
        else:
            out = df["Close"].copy()
    else:
        # single ticker sometimes returns flat columns
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            out = df[["Close"]].copy()
        else:
            out = df.copy()

    out = out.dropna(how="all")
    return out


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    return _download_prices_raw(tickers, start, end)


def prepare_prices(
    tickers: List[str],
    ticker_to_label: Dict[str, str],
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (prices, data_quality_table). Handles missing tickers gracefully."""
    raw = load_prices(tickers, start, end)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Rename tickers → friendly labels
    raw = raw.rename(columns=ticker_to_label)

    # Drop columns that are entirely missing (common yfinance issue for some tickers/date ranges)
    non_empty_cols = raw.columns[raw.notna().any()].tolist()
    dropped = sorted(set(raw.columns) - set(non_empty_cols))
    raw = raw[non_empty_cols]

    # Basic cleaning: forward fill then drop remaining missing
    raw = raw.dropna(axis=0, how="all")
    raw = raw.ffill().dropna()

    # Data quality
    quality = pd.DataFrame({
        "Asset": raw.columns,
        "Start": [raw[c].first_valid_index() for c in raw.columns],
        "End": [raw[c].last_valid_index() for c in raw.columns],
        "Coverage %": [
            float(raw[c].notna().mean() * 100.0) for c in raw.columns
        ],
    }).sort_values("Coverage %")

    if dropped:
        st.warning(
            "Some assets returned no data for this date range and were removed: "
            + ", ".join(dropped)
        )

    return raw, quality


# =============================
# Portfolio math
# =============================
def portfolio_series(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Daily portfolio returns series."""
    w = np.asarray(weights, dtype=float)
    return pd.Series(returns.values @ w, index=returns.index, name="Portfolio")


def portfolio_stats(mu: pd.Series, cov: pd.DataFrame, w: np.ndarray, rf: float) -> Tuple[float, float, float]:
    """(annual_return, annual_vol, sharpe)"""
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (ret - rf) / (vol + 1e-12)
    return ret, vol, sharpe


def optimize_weights(mu: pd.Series, cov: pd.DataFrame, rf: float, goal: str) -> Optional[np.ndarray]:
    """goal: 'max_sharpe' or 'min_vol'"""
    if not SCIPY_OK:
        return None

    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.full(n, 1.0 / n)

    if goal == "min_vol":
        fun = lambda w: portfolio_stats(mu, cov, w, rf)[1]
    else:
        fun = lambda w: -portfolio_stats(mu, cov, w, rf)[2]

    res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return None
    return res.x


def monte_carlo_frontier(mu: pd.Series, cov: pd.DataFrame, rf: float, n_sims: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(mu)
    W = rng.random((n_sims, n))
    W = W / W.sum(axis=1, keepdims=True)

    rets, vols, sharpes = [], [], []
    for w in W:
        r, v, s = portfolio_stats(mu, cov, w, rf)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)

    return pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sharpes})


# =============================
# UI functions
# =============================
def chart_cumulative(perf: pd.DataFrame) -> None:
    fig = px.line(perf, title="Cumulative Performance (Index starts at 1.0)")
    fig.update_layout(legend_title_text="Asset")
    st.plotly_chart(fig, use_container_width=True)


def chart_volatility(asset_vol: pd.Series) -> None:
    vol_df = pd.DataFrame({
        "Asset": asset_vol.index,
        "Annualized Volatility": asset_vol.values,
    }).sort_values("Annualized Volatility", ascending=False)

    fig = px.bar(vol_df, x="Asset", y="Annualized Volatility", title="Annualized Volatility by Asset")
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)


def chart_corr(returns: pd.DataFrame) -> None:
    corr = returns.corr()
    fig = px.imshow(corr, title="Correlation Heatmap (Lower = Better Diversification)", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)


def chart_drawdown(perf: pd.DataFrame) -> None:
    dd = (perf / perf.cummax()) - 1.0
    fig = px.line(dd, title="Drawdown (%) — Distance from the Previous Peak")
    st.plotly_chart(fig, use_container_width=True)


# =============================
# App
# =============================
def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "This dashboard uses Yahoo Finance (via yfinance) to compare assets, evaluate risk, and explore optimized allocations."
    )

    # ---- Sidebar
    st.sidebar.header("⚙️ Controls")
    asset_labels = list(DEFAULT_UNIVERSE.keys())

    default_selected = [
        "Saudi Aramco (2222.SR)",
        "Al Rajhi Bank (1120.SR)",
        "STC (7010.SR)",
        "S&P 500 (^GSPC)",
        "Gold (GLD)",
    ]

    selected = st.sidebar.multiselect(
        "Select 3–8 assets",
        options=asset_labels,
        default=default_selected,
        help="Mix Saudi and global assets to explore diversification.",
    )
    validate_asset_count(selected)

    today = date.today()
    start_default = date(today.year - 5, 1, 1)
    start_date = st.sidebar.date_input("Start date", start_default)
    end_date = st.sidebar.date_input("End date", today)

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    rf = st.sidebar.number_input(
        "Risk-free rate (annual, %)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.25,
        help="Used in Sharpe ratio. Example: 3%.",
    ) / 100.0

    opt_mode_label = st.sidebar.selectbox(
        "Optimization goal",
        options=["Max Sharpe (recommended)", "Min Volatility"],
        help="Max Sharpe focuses on risk-adjusted return; Min Volatility focuses on stability.",
    )
    goal = "min_vol" if "Min" in opt_mode_label else "max_sharpe"

    n_sims = st.sidebar.slider(
        "Monte Carlo portfolios",
        min_value=300,
        max_value=5000,
        value=1500,
        step=100,
        help="More simulations gives a smoother frontier, but takes longer.",
    )

    st.sidebar.divider()
    st.sidebar.caption("If Yahoo Finance is slow, reduce the date range or number of assets.")

    tickers = [DEFAULT_UNIVERSE[a] for a in selected]
    ticker_to_label = {DEFAULT_UNIVERSE[a]: a for a in selected}

    # ---- Load & prepare
    with st.spinner("Loading market data from Yahoo Finance..."):
        prices, quality = prepare_prices(tickers, ticker_to_label, str(start_date), str(end_date))

    if prices.empty or prices.shape[1] < 3:
        st.error("Not enough usable data returned. Try a different date range or different assets.")
        st.stop()

    # Export for reproducibility
    export_path = os.path.join(DATA_DIR, "cached_prices.csv")
    export_csv(prices, export_path)
    st.caption(f"Cached data exported to `{export_path}` for reproducibility.")

    # ---- Data quality (real analyst touch)
    with st.expander("Data Quality (coverage & availability)"):
        st.dataframe(
            quality.style.format({"Coverage %": "{:.1f}"}),
            use_container_width=True,
            height=240,
        )

    # ---- Returns & cumulative
    returns = prices.pct_change().dropna()
    cum = normalize_prices_to_index(returns)

    # Baseline equal-weight KPIs
    w_eq = np.full(prices.shape[1], 1.0 / prices.shape[1])
    port_daily = portfolio_series(returns, w_eq)
    port_cum = (1.0 + port_daily).cumprod()

    port_ann_ret = annualize_mean(port_daily)
    port_ann_vol = annualize_vol(port_daily)
    port_sharpe = (port_ann_ret - rf) / (port_ann_vol + 1e-12)
    port_mdd = max_drawdown(port_cum)

    asset_ann_ret = returns.apply(annualize_mean)
    asset_ann_vol = returns.apply(annualize_vol)
    best_asset = asset_ann_ret.idxmax()
    worst_asset = asset_ann_ret.idxmin()

    # ---- KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Portfolio (Eq) Exp. Return", pct(port_ann_ret))
    c2.metric("Portfolio (Eq) Volatility", pct(port_ann_vol))
    c3.metric("Portfolio (Eq) Sharpe", f"{port_sharpe:,.2f}")
    c4.metric("Portfolio (Eq) Max Drawdown", pct(port_mdd))
    c5.metric("Best Asset (Return)", best_asset)
    c6.metric("Worst Asset (Return)", worst_asset)

    # ---- Performance section
    st.subheader("1) Performance")
    perf = cum.copy()
    perf["Equal-weight Portfolio"] = port_cum
    chart_cumulative(perf)

    st.info(
        "Interpretation: higher lines indicate stronger cumulative growth. "
        "Use the risk section to confirm whether returns came with higher drawdowns."
    )

    # ---- Risk section
    st.subheader("2) Risk & Diversification")
    r1, r2 = st.columns([1.0, 1.0])
    with r1:
        chart_volatility(asset_ann_vol)
    with r2:
        chart_corr(returns)

    chart_drawdown(perf)
    st.success(
        "Takeaway: diversification improves when correlations are low. "
        "Drawdown highlights downside pain that volatility alone may not capture."
    )

    # ---- Optimization
    st.subheader("3) Portfolio Optimization")
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS

    if not SCIPY_OK:
        st.warning("SciPy is missing, so true optimization is disabled. Install SciPy to enable optimized weights.")
    w_opt = optimize_weights(mu, cov, rf, goal) if SCIPY_OK else None

    sim_df = monte_carlo_frontier(mu, cov, rf, n_sims=n_sims)
    fig_frontier = px.scatter(
        sim_df,
        x="Volatility",
        y="Return",
        hover_data=["Sharpe"],
        title="Monte Carlo Efficient Frontier (Simulated Portfolios)",
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    wcol, tcol = st.columns([0.9, 1.1])
    assets = list(returns.columns)

    with wcol:
        st.markdown("#### Weights")
        if w_opt is None:
            st.caption("Showing equal-weight allocation (optimization unavailable).")
            w_show = pd.Series(w_eq, index=assets, name="Equal-weight")
        else:
            w_show = pd.Series(w_opt, index=assets, name="Optimized")

        weights_df = (w_show * 100).round(2).sort_values(ascending=False).to_frame("Weight (%)")
        st.dataframe(weights_df, use_container_width=True, height=320)

        fig_w = px.bar(
            weights_df.reset_index().rename(columns={"index": "Asset"}),
            x="Asset",
            y="Weight (%)",
            title="Portfolio Allocation",
        )
        fig_w.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig_w, use_container_width=True)

    with tcol:
        st.markdown("#### Portfolio Comparison")
        eq_ret, eq_vol, eq_sh = portfolio_stats(mu, cov, w_eq, rf)
        if w_opt is None:
            opt_ret, opt_vol, opt_sh = (np.nan, np.nan, np.nan)
        else:
            opt_ret, opt_vol, opt_sh = portfolio_stats(mu, cov, w_opt, rf)

        comp = pd.DataFrame({
            "Portfolio": ["Equal-weight", "Optimized"],
            "Expected Return": [eq_ret, opt_ret],
            "Volatility": [eq_vol, opt_vol],
            "Sharpe": [eq_sh, opt_sh],
        })

        st.dataframe(
            comp.style.format({"Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}"}),
            use_container_width=True,
            height=180,
        )

        # More specific insight tied to numbers (less “generic template”)
        if w_opt is None:
            st.write(
                "The frontier shows feasible risk-return tradeoffs. Enable SciPy to compute a true optimized allocation."
            )
        else:
            uplift = opt_sh - eq_sh
            st.write(
                f"Compared to equal-weight, the optimized portfolio improves Sharpe by **{uplift:.2f}** "
                f"(risk-adjusted performance), subject to historical estimates and the assumptions noted below."
            )

    # ---- Insights & assumptions
    st.subheader("4) Key Insights & Recommendations")
    st.markdown(
        f"- Over this period, **{best_asset}** had the highest annualized return, while **{worst_asset}** lagged."
    )
    st.markdown("- Diversification improves when correlations are low; use the heatmap to identify redundant exposure.")
    st.markdown("- Drawdown reveals downside risk that may matter more than volatility for conservative investors.")
    st.markdown("- Use Max Sharpe for risk-adjusted growth, and Min Volatility when stability is the priority.")

    st.subheader("5) Assumptions & Limitations")
    st.markdown(
        """
- Yahoo Finance data quality can vary by ticker and date range.
- Returns are based on price changes (typically Adj Close). Dividends, taxes, and transaction costs are excluded.
- Annualization assumes ~252 trading days; crypto trades 24/7 and may be less comparable under this convention.
- Optimization is based on historical mean/covariance and may not generalize to the future.
"""
    )

    with st.expander("Raw price data (last 30 rows)"):
        st.dataframe(prices.tail(30), use_container_width=True)


if __name__ == "__main__":
    main()

