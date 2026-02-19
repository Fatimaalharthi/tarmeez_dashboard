#   Portfolio Optimizer & Risk Intelligence Dashboard

### Saudi & Global Market Allocation Tool

------------------------------------------------------------------------

## Executive Summary

This project delivers a live, interactive financial dashboard designed
to support data-driven portfolio allocation decisions. It enables
investors to analyze asset performance, assess portfolio risk, evaluate
diversification benefits, and construct optimized allocations using
publicly available market data.

The dashboard integrates financial analytics with executive-level
visualization to answer a central investment question:

> How can capital be optimally allocated across Saudi and global assets
> to maximize return while managing risk?

Through performance metrics, risk diagnostics, correlation analysis,
Monte Carlo simulations, and constrained portfolio optimization, the
application transforms historical data into actionable allocation
insights.

This tool is suitable for investors, analysts, and decision-makers
seeking structured portfolio intelligence rather than static reporting.

------------------------------------------------------------------------

## Project Overview

This dashboard provides an interactive environment for evaluating
financial assets across Saudi and global markets. It goes beyond simple
trend visualization by integrating portfolio theory, risk-adjusted
performance metrics, and optimization modeling into a single analytical
framework.

The application is fully interactive, live-deployable, and built to
support real-world investment decisions.

Target audience includes:

-   Individual investors
-   Portfolio managers
-   Financial analysts
-   Investment decision stakeholders

The system emphasizes clarity, usability, and analytical transparency.

------------------------------------------------------------------------

## Data Source

All financial data is sourced from **Yahoo Finance**, accessed using the
Python library `yfinance`.

The dataset includes daily adjusted closing prices for:

-   Saudi equities (Tadawul)
-   Global indices (S&P 500, Nasdaq)
-   Commodities (Gold, Oil)
-   Fixed income proxies (US Treasury ETF)
-   Cryptocurrency (Bitcoin)

Data characteristics:

-   Publicly available
-   Daily frequency
-   Historical time series
-   No proprietary or confidential sources used

For reproducibility and compliance with submission requirements, the
dashboard exports a cached CSV file into the `/data` folder during
execution.

------------------------------------------------------------------------

## Methodology & Analytical Approach

### 1. Data Collection

Market data is dynamically retrieved based on user-selected assets and
time range.

### 2. Data Cleaning & Preparation

-   Missing values are forward-filled
-   Incomplete records are removed
-   Data integrity is validated before analysis

### 3. Return & Risk Calculations

Daily returns are calculated and annualized using standard financial
formulas:

-   Annualized Return
-   Annualized Volatility
-   Sharpe Ratio
-   Maximum Drawdown

### 4. Diversification Analysis

A correlation matrix is computed to evaluate cross-asset relationships.
Lower correlations indicate stronger diversification benefits.

### 5. Portfolio Optimization

Two optimization strategies are implemented:

-   **Maximum Sharpe Ratio**
-   **Minimum Volatility**

Optimization is solved using constrained numerical optimization (SciPy
SLSQP).

### 6. Monte Carlo Simulation

Thousands of random portfolio allocations are simulated to generate an
Efficient Frontier, illustrating the trade-off between expected return
and volatility.

------------------------------------------------------------------------

## Dashboard Features

-   Interactive asset selection (3-8 assets)
-   Adjustable date range
-   Customizable risk-free rate
-   Optimization objective selection
-   Monte Carlo simulation control
-   KPI summary metrics
-   Cumulative performance comparison
-   Volatility ranking visualization
-   Correlation heatmap
-   Drawdown analysis
-   Efficient frontier visualization
-   Optimized portfolio weights display
-   Stakeholder-ready insights & recommendations
-   Assumptions and limitations disclosure

------------------------------------------------------------------------

## Key Insights

-   High-return assets often exhibit higher volatility and deeper
    drawdowns.
-   Diversification improves when assets demonstrate low correlation.
-   Equal-weight portfolios are simple but rarely optimal.
-   Optimized portfolios can significantly improve risk-adjusted
    performance.
-   Maximum Sharpe allocations distribute capital across uncorrelated
    assets.

------------------------------------------------------------------------

## Assumptions & Limitations

-   Historical performance does not guarantee future results.
-   Annualization assumes approximately 252 trading days.
-   Dividends, taxes, and transaction costs are excluded.
-   Optimization assumes stable mean and covariance estimates.
-   Cryptocurrency operates 24/7, which may slightly affect
    annualization comparisons.

------------------------------------------------------------------------

## Technical Stack

-   Python
-   Streamlit
-   yfinance
-   pandas
-   numpy
-   plotly
-   scipy

------------------------------------------------------------------------

## Repository Structure

portfolio-optimizer-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
├── /data
│   └── cached_prices.csv
│
├── /images
│   ├── dashboard_overview.png
│   ├── correlation_heatmap.png
│   └── optimization_section.png

------------------------------------------------------------------------

## Live Dashboard

(Add your deployed Streamlit link here)

------------------------------------------------------------------------

## Conclusion

This project demonstrates applied financial analytics, portfolio theory,
optimization modeling, and executive-level data visualization.

It provides an interactive investment decision-support system that
integrates performance analysis, diversification modeling, and
optimization into a unified framework.
