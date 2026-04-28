from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_risk_adjusted_table_data as vw
from helper_functions import find_project_root, load_factor_data, resolve_path


LONG_PORTFOLIO = "Q4"
SHORT_PORTFOLIO = "Q2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data for value-weighted risk-adjusted performance "
            "using Q4 and Q2 instead of Q5 and Q1."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional results run directory, e.g. results/current_res.",
    )
    parser.add_argument(
        "--portfolio-source",
        type=Path,
        default=None,
        help="Optional portfolio source CSV. Prefer monthly_holdings.csv with WeightedReturn.",
    )
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=None,
        help="Optional factor returns CSV. Defaults to run_config.json or results/extraction_static/factor_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "<portfolio_evaluation_dir>/thesis_risk_adjusted_tables_q4_q2."
        ),
    )
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=12,
        help="Newey-West/HAC lags. Default is 12.",
    )
    return parser.parse_args()


def resolve_cli_path(path: Path | None, project_root: Path) -> Path | None:
    if path is None:
        return None
    return resolve_path(path, project_root)


def validate_methods_and_portfolios(monthly_returns: pd.DataFrame) -> None:
    available_methods = sorted(monthly_returns["Method"].dropna().unique())
    missing_methods = [m for m in vw.METHODS if m not in available_methods]
    if missing_methods:
        raise ValueError(
            "Monthly portfolio returns are missing required sorting methods.\n"
            f"Missing: {missing_methods}\n"
            f"Available: {available_methods}"
        )

    missing = []
    for method in vw.METHODS:
        portfolios = set(monthly_returns.loc[monthly_returns["Method"] == method, "Portfolio"])
        for portfolio in [SHORT_PORTFOLIO, LONG_PORTFOLIO]:
            if portfolio not in portfolios:
                missing.append(f"{method}/{portfolio}")
    if missing:
        raise ValueError(
            "Missing required Q2/Q4 portfolios for Q4-Q2 construction: " + ", ".join(missing)
        )


def build_strategy_returns(
    monthly_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.DataFrame]:
    validate_methods_and_portfolios(monthly_returns)

    q4_returns: dict[str, pd.Series] = {}
    ls_returns: dict[str, pd.Series] = {}
    used_rows = []

    for method in vw.METHODS:
        sub = monthly_returns.loc[monthly_returns["Method"] == method]
        q2 = (
            sub.loc[sub["Portfolio"] == SHORT_PORTFOLIO, ["Date", "Return"]]
            .drop_duplicates("Date")
            .set_index("Date")["Return"]
            .sort_index()
        )
        q4 = (
            sub.loc[sub["Portfolio"] == LONG_PORTFOLIO, ["Date", "Return"]]
            .drop_duplicates("Date")
            .set_index("Date")["Return"]
            .sort_index()
        )

        aligned = pd.concat({SHORT_PORTFOLIO: q2, LONG_PORTFOLIO: q4}, axis=1).dropna()
        aligned = aligned.loc[aligned.index.intersection(factors.index)].sort_index()
        if aligned.empty:
            raise ValueError(f"No factor-aligned Q2/Q4 observations for {method}.")

        q4_series = aligned[LONG_PORTFOLIO].rename(method)
        ls_series = (aligned[LONG_PORTFOLIO] - aligned[SHORT_PORTFOLIO]).rename(method)
        q4_returns[method] = q4_series
        ls_returns[method] = ls_series

        used_rows.append(
            pd.DataFrame(
                {
                    "Date": q4_series.index,
                    "Method": method,
                    "PortfolioStrategy": "Q4",
                    "Return": q4_series.values,
                }
            )
        )
        used_rows.append(
            pd.DataFrame(
                {
                    "Date": ls_series.index,
                    "Method": method,
                    "PortfolioStrategy": "LongShort",
                    "Return": ls_series.values,
                }
            )
        )

    used = pd.concat(used_rows, ignore_index=True)
    used["Date"] = pd.to_datetime(used["Date"]).dt.strftime("%Y-%m-%d")
    return q4_returns, ls_returns, used


def save_cumulative_return_plots(monthly_used: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cumulative = vw.build_cumulative_returns(monthly_used)
    outputs: dict[str, Path] = {}

    strategy_titles = {
        "LongShort": f"Cumulative Returns: Long-Short {LONG_PORTFOLIO} - {SHORT_PORTFOLIO}",
        "Q4": f"Cumulative Returns: Pure {LONG_PORTFOLIO}",
    }
    strategy_filenames = {
        "LongShort": "cumulative_returns_longshort_q4_q2.png",
        "Q4": "cumulative_returns_q4.png",
    }

    for strategy, title in strategy_titles.items():
        sub = cumulative.loc[cumulative["PortfolioStrategy"] == strategy].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.axhline(0.0, color="#2f3b4a", linewidth=0.9, linestyle="--", alpha=0.75)

        for method in vw.METHODS:
            method_sub = sub.loc[sub["Method"] == method].sort_values("Date")
            if method_sub.empty:
                continue
            ax.plot(
                method_sub["Date"],
                method_sub["CumulativeReturn"],
                label=method,
                color=vw.METHOD_COLORS.get(method),
                linewidth=2.1,
            )

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

        path = plot_dir / strategy_filenames[strategy]
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs[strategy] = path

    return outputs


def save_outputs(
    output_dir: Path,
    ls_levels: pd.DataFrame,
    ls_diffs: pd.DataFrame,
    q4_levels: pd.DataFrame,
    q4_diffs: pd.DataFrame,
    monthly_used: pd.DataFrame,
    preview: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "table_ls_alpha_levels": output_dir / "table_ls_alpha_levels.csv",
        "table_ls_alpha_differences": output_dir / "table_ls_alpha_differences.csv",
        "table_q4_alpha_levels": output_dir / "table_q4_alpha_levels.csv",
        "table_q4_alpha_differences": output_dir / "table_q4_alpha_differences.csv",
        "monthly_portfolio_returns_used": output_dir / "monthly_portfolio_returns_used.csv",
        "risk_adjusted_table_preview": output_dir / "risk_adjusted_table_preview.csv",
    }

    ls_levels.to_csv(outputs["table_ls_alpha_levels"], index=False)
    ls_diffs.to_csv(outputs["table_ls_alpha_differences"], index=False)
    q4_levels.to_csv(outputs["table_q4_alpha_levels"], index=False)
    q4_diffs.to_csv(outputs["table_q4_alpha_differences"], index=False)
    monthly_used.to_csv(outputs["monthly_portfolio_returns_used"], index=False)
    preview.to_csv(outputs["risk_adjusted_table_preview"], index=False)

    return outputs


def print_identification(
    run_dir: Path,
    portfolio_eval_dir: Path,
    portfolio_source: Path,
    source_type: str,
    factors_csv: Path,
    output_dir: Path,
    nw_lags: int,
) -> None:
    print("\nIdentified inputs and reused helpers")
    print(f"  run_dir: {run_dir}")
    print(f"  portfolio_evaluation_dir: {portfolio_eval_dir}")
    print(f"  portfolio constituent / weighted-return data: {portfolio_source}")
    print(f"  portfolio source type: {source_type}")
    print(f"  strategy construction: {LONG_PORTFOLIO} and {SHORT_PORTFOLIO}")
    print(f"  monthly factor returns: {factors_csv}")
    print("  risk-adjusted regression helper: modelling/shared/step5_evaluation.py::risk_adjusted_performance")
    print("  Newey-West/HAC helper: modelling/shared/step5_evaluation.py::_ols_newey_west_full")
    print("  alpha-difference helper: modelling/shared/step5_evaluation.py::alpha_differences")
    print("  factor loader helper: modelling/shared/helper_functions.py::load_factor_data")
    print(f"  HAC lags: {nw_lags}")
    print(f"  output_dir: {output_dir}")


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    run_dir = vw.choose_run_dir(project_root, resolve_cli_path(args.run_dir, project_root))
    portfolio_source, source_type, portfolio_eval_dir = vw.choose_portfolio_source(
        run_dir=run_dir,
        requested_source=resolve_cli_path(args.portfolio_source, project_root),
    )
    factors_csv = vw.choose_factor_csv(
        project_root=project_root,
        run_dir=run_dir,
        requested_factors=resolve_cli_path(args.factors_csv, project_root),
    )
    output_dir = resolve_cli_path(args.output_dir, project_root)
    if output_dir is None:
        output_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_q4_q2"

    print_identification(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        portfolio_source=portfolio_source,
        source_type=source_type,
        factors_csv=factors_csv,
        output_dir=output_dir,
        nw_lags=args.nw_lags,
    )

    monthly_returns = vw.load_monthly_portfolio_returns(portfolio_source, source_type)
    factors = load_factor_data(factors_csv)
    rf = factors["RF"].copy()
    zero_rf = pd.Series(0.0, index=rf.index, name="RF")

    q4_returns, ls_returns, monthly_used = build_strategy_returns(monthly_returns, factors)

    ls_levels = vw.run_level_regressions(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
        nw_lags=args.nw_lags,
    )
    q4_levels = vw.run_level_regressions(
        strategy_returns=q4_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q4",
        nw_lags=args.nw_lags,
    )
    ls_diffs = vw.run_alpha_difference_tests(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
        nw_lags=args.nw_lags,
    )
    q4_diffs = vw.run_alpha_difference_tests(
        strategy_returns=q4_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q4",
        nw_lags=args.nw_lags,
    )

    vw.assert_expected_shapes(ls_levels, ls_diffs, q4_levels, q4_diffs)
    preview = vw.build_preview(
        levels=pd.concat([ls_levels, q4_levels], ignore_index=True),
        differences=pd.concat([ls_diffs, q4_diffs], ignore_index=True),
    )
    outputs = save_outputs(
        output_dir=output_dir,
        ls_levels=ls_levels,
        ls_diffs=ls_diffs,
        q4_levels=q4_levels,
        q4_diffs=q4_diffs,
        monthly_used=monthly_used,
        preview=preview,
    )
    plot_outputs = save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    print("\nCreated CSV files")
    row_counts = {
        "table_ls_alpha_levels": len(ls_levels),
        "table_ls_alpha_differences": len(ls_diffs),
        "table_q4_alpha_levels": len(q4_levels),
        "table_q4_alpha_differences": len(q4_diffs),
        "monthly_portfolio_returns_used": len(monthly_used),
        "risk_adjusted_table_preview": len(preview),
    }
    for key, path in outputs.items():
        print(f"  {path} ({row_counts[key]} rows)")

    print("\nCreated plot files")
    for path in plot_outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
