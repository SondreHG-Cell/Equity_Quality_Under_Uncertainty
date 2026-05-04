from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from step5_evaluation import (
    raw_performance,
    risk_adjusted_performance,
    alpha_differences,
    grs_test,
    grs_tests_all_models,
)
from step5_probabilistic import (
    probabilistic_evaluation,
    calibration_plot,
    sharpness_plot,
)

from helper_functions import (
    load_factor_data,
    build_monthly_portfolio_returns,
    build_probabilistic_targets,
)


def run_portfolio_evaluation(
    assignments_csv: str | Path,
    stock_prices_csv: str | Path,
    factors_csv: str | Path,
    output_dir: str | Path,
    market_cap_csv: str | Path = "data/processed_data_lseg/historical_market_cap_nok.csv",
    dividends_csv: str | Path | None = "data/processed_data_lseg/dividends_monthly_nok.csv",
    n_portfolios: int = 5,
    nw_lags: int = 12,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    assignments_csv = Path(assignments_csv)
    stock_prices_csv = Path(stock_prices_csv)
    market_cap_csv = Path(market_cap_csv)
    dividends_csv = Path(dividends_csv) if dividends_csv is not None else None
    factors_csv = Path(factors_csv)

    # --------------------------------------------------
    # Load inputs
    # --------------------------------------------------
    assignments = pd.read_csv(assignments_csv)
    factors = load_factor_data(factors_csv)

    # --------------------------------------------------
    # Build monthly portfolio returns using:
    # - fixed annual membership from assignments
    # - monthly total returns from stock prices plus dividends
    # - monthly value weights from lagged monthly market cap
    # --------------------------------------------------
    prepared = build_monthly_portfolio_returns(
        assignments=assignments,
        stock_prices_csv=stock_prices_csv,
        market_cap_csv=market_cap_csv,
        factors=factors,
        n_portfolios=n_portfolios,
        dividends_csv=dividends_csv,
    )

    returns_wide = prepared["returns_wide"]
    ls_returns = prepared["ls_returns"]
    q5_returns = prepared["q5_returns"]
    rf = prepared["rf"]
    monthly_holdings = prepared.get("monthly_holdings")

    # Save monthly portfolio returns
    monthly_portfolio_returns_csv = output_dir / "monthly_portfolio_returns.csv"
    returns_wide.to_csv(monthly_portfolio_returns_csv)

    if monthly_holdings is not None:
        monthly_holdings_csv = output_dir / "monthly_holdings.csv"
        monthly_holdings.to_csv(monthly_holdings_csv, index=False)
    else:
        monthly_holdings_csv = None

    # --------------------------------------------------
    # 5.1 Raw performance
    # --------------------------------------------------
    raw_df = raw_performance(returns_wide, rf)
    raw_csv = output_dir / "raw_performance.csv"
    raw_df.to_csv(raw_csv, index=False)

    # --------------------------------------------------
    # 5.2 Risk-adjusted performance
    # --------------------------------------------------
    risk_frames = []

    for method in returns_wide.columns.get_level_values(0).unique():
        for portfolio in returns_wide[method].columns:
            series = returns_wide[(method, portfolio)]
            res = risk_adjusted_performance(
                portfolio_returns=series,
                factors=factors,
                rf=rf,
                lags=nw_lags,
            )
            res["Method"] = method
            res["Portfolio"] = portfolio
            risk_frames.append(res)

    risk_df = pd.concat(risk_frames, ignore_index=True)
    risk_df = risk_df[
        ["Method", "Portfolio", "FactorModel", "alpha", "t_stat", "p_value", "r_squared", "n_obs"]
    ]
    risk_csv = output_dir / "risk_adjusted_performance.csv"
    risk_df.to_csv(risk_csv, index=False)

    # --------------------------------------------------
    # 5.3 Alpha differences
    # --------------------------------------------------
    alpha_diff_df = alpha_differences(
        ls_returns=ls_returns,
        factors=factors,
        rf=rf,
        model="FF5_MOM",
        lags=nw_lags,
        base_method="Method1_ObservedQuality",
    )
    alpha_diff_csv = output_dir / "alpha_differences.csv"
    alpha_diff_df.to_csv(alpha_diff_csv, index=False)

    # --------------------------------------------------
    # 5.4 GRS test
    # --------------------------------------------------
    grs = grs_test(
        q5_returns=q5_returns,
        factors=factors,
        rf=rf,
        model="FF5_MOM",
    )
    grs_json = output_dir / "grs_test.json"
    with open(grs_json, "w", encoding="utf-8") as f:
        json.dump(grs, f, indent=2)

    # --------------------------------------------------
    # 5.5 Probabilistic evaluation
    # --------------------------------------------------
    y_true, y_prob = build_probabilistic_targets(
        assignments=assignments,
        stock_prices_csv=stock_prices_csv,
        dividends_csv=dividends_csv,
    )

    prob_metrics = probabilistic_evaluation(y_true=y_true, y_prob=y_prob)
    prob_df = pd.DataFrame([prob_metrics])
    prob_csv = output_dir / "probabilistic_metrics.csv"
    prob_df.to_csv(prob_csv, index=False)

    # Save plots
    fig, ax = plt.subplots(figsize=(6, 6))
    calibration_plot(y_true=y_true, y_prob=y_prob, n_bins=10, ax=ax)
    calibration_png = plots_dir / "calibration_plot.png"
    fig.tight_layout()
    fig.savefig(calibration_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sharpness_plot(y_prob=y_prob, ax=ax)
    sharpness_png = plots_dir / "sharpness_plot.png"
    fig.tight_layout()
    fig.savefig(sharpness_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_dir": str(output_dir),
        "monthly_portfolio_returns_csv": str(monthly_portfolio_returns_csv),
        "monthly_holdings_csv": str(monthly_holdings_csv) if monthly_holdings_csv else None,
        "raw_performance_csv": str(raw_csv),
        "risk_adjusted_performance_csv": str(risk_csv),
        "alpha_differences_csv": str(alpha_diff_csv),
        "grs_test_json": str(grs_json),
        "probabilistic_metrics_csv": str(prob_csv),
        "calibration_plot_png": str(calibration_png),
        "sharpness_plot_png": str(sharpness_png),
    }
