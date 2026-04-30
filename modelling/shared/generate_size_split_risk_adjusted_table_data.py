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
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_risk_adjusted_table_data as vw
import generate_capped_weight_risk_adjusted_table_data as ucits
from helper_functions import find_project_root, load_factor_data, parse_month_series, resolve_path


SIZE_GROUPS = ["SmallCap", "LargeCap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data for UCITS 5/10/40 weighted Q5/Q1 risk-adjusted "
            "performance separately for small-cap and large-cap firms."
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
        help="Optional constituent CSV. Must contain Method, Portfolio, Date, Return, and LagMarketCap.",
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
            "<portfolio_evaluation_dir>/thesis_risk_adjusted_tables_size_split_ucits_5_10_40."
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


def choose_constituent_source(
    run_dir: Path,
    requested_source: Path | None,
) -> tuple[Path, Path]:
    portfolio_eval_dir = vw.portfolio_eval_dir_for_run(run_dir)
    if portfolio_eval_dir is None:
        portfolio_eval_dir = run_dir / "portfolio_evaluation"

    default_holdings = portfolio_eval_dir / "monthly_holdings.csv"

    if requested_source is not None:
        if not requested_source.exists():
            raise FileNotFoundError(
                "Requested size-split portfolio source does not exist.\n"
                f"Requested: {requested_source}\n"
                f"Default checked: {default_holdings}"
            )
        return requested_source, requested_source.parent

    if default_holdings.exists():
        return default_holdings, portfolio_eval_dir

    raise FileNotFoundError(
        "Size-split UCITS-weighted returns require constituent-level monthly holdings.\n"
        "Could not locate monthly_holdings.csv at:\n"
        f"{default_holdings}\n"
        "The wide monthly_portfolio_returns.csv file is already aggregated and cannot be split by firm size."
    )


def assign_size_groups(holdings: pd.DataFrame) -> pd.DataFrame:
    firm_size = (
        holdings[["Ticker", "FormationYear", "Date", "LagMarketCap"]]
        .drop_duplicates()
        .dropna(subset=["LagMarketCap"])
        .groupby(["Ticker", "FormationYear"], as_index=False)["LagMarketCap"]
        .median()
        .rename(columns={"LagMarketCap": "FirmYearMedianLagMarketCap"})
    )

    firm_size = firm_size.sort_values(
        ["FormationYear", "FirmYearMedianLagMarketCap", "Ticker"]
    ).reset_index(drop=True)
    firm_size["_rank"] = firm_size.groupby("FormationYear").cumcount() + 1
    firm_size["_n"] = firm_size.groupby("FormationYear")["Ticker"].transform("count")
    firm_size["SizeGroup"] = np.where(
        firm_size["_rank"] <= firm_size["_n"] / 2,
        "SmallCap",
        "LargeCap",
    )

    return holdings.merge(
        firm_size[["Ticker", "FormationYear", "SizeGroup", "FirmYearMedianLagMarketCap"]],
        on=["Ticker", "FormationYear"],
        how="inner",
        validate="many_to_one",
    )


def load_size_split_ucits_weighted_monthly_returns(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    required = ["Ticker", "FormationYear", "Method", "Portfolio", "Date", "Return", "LagMarketCap"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required constituent columns for size-split value weighting: {missing}"
        )

    out = df[required].copy()
    out["Date"] = parse_month_series(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out["Method"] = out["Method"].astype(str).str.strip()
    out["Portfolio"] = out["Portfolio"].astype(str).str.strip()
    out["FormationYear"] = pd.to_numeric(out["FormationYear"], errors="coerce")
    out["Return"] = pd.to_numeric(out["Return"], errors="coerce")
    out["LagMarketCap"] = pd.to_numeric(out["LagMarketCap"], errors="coerce")
    out = out.dropna(
        subset=["Ticker", "FormationYear", "Date", "Method", "Portfolio", "Return", "LagMarketCap"]
    )
    out = out.loc[out["LagMarketCap"] > 0].copy()
    out["FormationYear"] = out["FormationYear"].astype(int)

    out = assign_size_groups(out)
    out = out.loc[out["Method"].isin(vw.METHODS)].copy()
    group_cols = ["SizeGroup", "Method", "Portfolio", "Date"]

    ucits_holdings = pd.concat(
        [
            ucits.apply_ucits_weights(
                group,
                single_issuer_cap=ucits.UCITS_SINGLE_ISSUER_CAP,
                large_position_threshold=ucits.UCITS_LARGE_POSITION_THRESHOLD,
                large_position_aggregate_cap=ucits.UCITS_LARGE_POSITION_AGGREGATE_CAP,
            )
            for _, group in out.groupby(group_cols, sort=False)
        ],
        ignore_index=True,
    )

    monthly = (
        ucits_holdings.groupby(group_cols, as_index=False)
        .agg(
            Return=("UCITSWeightedReturn", "sum"),
            cash_weight=("CashWeight", "first"),
            n_firms=("Ticker", "nunique"),
            total_lag_mcap=("LagMarketCap", "sum"),
            max_raw_weight=("RawValueWeight", "max"),
            max_ucits_weight=("UCITSWeight", "max"),
            large_position_weight=(
                "UCITSWeight",
                lambda s: s[s > ucits.UCITS_LARGE_POSITION_THRESHOLD + 1e-10].sum(),
            ),
            n_large_positions=("AboveLargePositionThreshold", "sum"),
            n_names_single_cap_binding=("SingleIssuerCapBinding", "sum"),
            aggregate_cap_relaxed=("AggregateCapRelaxed", "first"),
        )
        .sort_values(["SizeGroup", "Method", "Portfolio", "Date"])
        .reset_index(drop=True)
    )

    diagnostics = (
        ucits_holdings.groupby(group_cols, as_index=False)
        .agg(
            n_firms=("Ticker", "nunique"),
            raw_weight_sum=("RawValueWeight", "sum"),
            ucits_weight_sum=("UCITSWeight", "sum"),
            max_raw_weight=("RawValueWeight", "max"),
            max_ucits_weight=("UCITSWeight", "max"),
            cash_weight=("CashWeight", "first"),
            large_position_weight=(
                "UCITSWeight",
                lambda s: s[s > ucits.UCITS_LARGE_POSITION_THRESHOLD + 1e-10].sum(),
            ),
            n_large_positions=("AboveLargePositionThreshold", "sum"),
            n_names_single_cap_binding=("SingleIssuerCapBinding", "sum"),
            aggregate_cap_relaxed=("AggregateCapRelaxed", "first"),
        )
        .sort_values(["SizeGroup", "Date", "Method", "Portfolio"])
        .reset_index(drop=True)
    )

    return monthly, ucits_holdings, diagnostics


def build_strategy_returns_for_size(
    monthly_returns: pd.DataFrame,
    factors: pd.DataFrame,
    size_group: str,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.DataFrame]:
    sub = monthly_returns.loc[monthly_returns["SizeGroup"] == size_group].copy()
    q5_returns, ls_returns, used = vw.build_strategy_returns(sub, factors)
    used.insert(0, "SizeGroup", size_group)
    return q5_returns, ls_returns, used


def add_size_group(df: pd.DataFrame, size_group: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "SizeGroup", size_group)
    return out


def run_size_group_analysis(
    monthly_returns: pd.DataFrame,
    factors: pd.DataFrame,
    rf: pd.Series,
    zero_rf: pd.Series,
    nw_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ls_level_frames = []
    ls_diff_frames = []
    q5_level_frames = []
    q5_diff_frames = []
    used_frames = []

    for size_group in SIZE_GROUPS:
        q5_returns, ls_returns, used = build_strategy_returns_for_size(
            monthly_returns=monthly_returns,
            factors=factors,
            size_group=size_group,
        )

        ls_levels = vw.run_level_regressions(
            strategy_returns=ls_returns,
            factors=factors,
            rf=zero_rf,
            strategy_label="LongShort",
            nw_lags=nw_lags,
        )
        q5_levels = vw.run_level_regressions(
            strategy_returns=q5_returns,
            factors=factors,
            rf=rf,
            strategy_label="Q5",
            nw_lags=nw_lags,
        )
        ls_diffs = vw.run_alpha_difference_tests(
            strategy_returns=ls_returns,
            factors=factors,
            rf=zero_rf,
            strategy_label="LongShort",
            nw_lags=nw_lags,
        )
        q5_diffs = vw.run_alpha_difference_tests(
            strategy_returns=q5_returns,
            factors=factors,
            rf=rf,
            strategy_label="Q5",
            nw_lags=nw_lags,
        )

        vw.assert_expected_shapes(ls_levels, ls_diffs, q5_levels, q5_diffs)
        ls_level_frames.append(add_size_group(ls_levels, size_group))
        ls_diff_frames.append(add_size_group(ls_diffs, size_group))
        q5_level_frames.append(add_size_group(q5_levels, size_group))
        q5_diff_frames.append(add_size_group(q5_diffs, size_group))
        used_frames.append(used)

    return (
        pd.concat(ls_level_frames, ignore_index=True),
        pd.concat(ls_diff_frames, ignore_index=True),
        pd.concat(q5_level_frames, ignore_index=True),
        pd.concat(q5_diff_frames, ignore_index=True),
        pd.concat(used_frames, ignore_index=True),
    )


def build_size_split_preview(levels: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    level_wide = levels.pivot_table(
        index=["SizeGroup", "PortfolioStrategy", "FactorModel"],
        columns="Method",
        values=["alpha_annualized", "t_stat", "p_value"],
        aggfunc="first",
    )
    level_wide.columns = [f"{method}_{metric}" for metric, method in level_wide.columns]
    level_wide = level_wide.reset_index()

    diff_wide = differences.pivot_table(
        index=["SizeGroup", "PortfolioStrategy", "FactorModel"],
        columns="Comparison",
        values=["alpha_difference_annualized", "t_stat", "p_value"],
        aggfunc="first",
        observed=False,
    )
    diff_wide.columns = [
        f"{comparison}_{metric}".replace(" ", "_").replace("+", "plus")
        for metric, comparison in diff_wide.columns
    ]
    diff_wide = diff_wide.reset_index()

    return level_wide.merge(diff_wide, on=["SizeGroup", "PortfolioStrategy", "FactorModel"], how="left")


def assert_size_split_shapes(
    ls_levels: pd.DataFrame,
    ls_diffs: pd.DataFrame,
    q5_levels: pd.DataFrame,
    q5_diffs: pd.DataFrame,
) -> None:
    expected = {
        "Long-short alpha levels": (ls_levels, 30),
        "Long-short alpha differences": (ls_diffs, 20),
        "Q5 alpha levels": (q5_levels, 30),
        "Q5 alpha differences": (q5_diffs, 20),
    }
    bad = [f"{name}: expected {n}, got {len(df)}" for name, (df, n) in expected.items() if len(df) != n]
    if bad:
        raise RuntimeError("Unexpected output row counts:\n" + "\n".join(bad))


def save_outputs(
    output_dir: Path,
    ls_levels: pd.DataFrame,
    ls_diffs: pd.DataFrame,
    q5_levels: pd.DataFrame,
    q5_diffs: pd.DataFrame,
    monthly_used: pd.DataFrame,
    preview: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "table_ls_alpha_levels_by_size": output_dir / "table_ls_alpha_levels_by_size.csv",
        "table_ls_alpha_differences_by_size": output_dir / "table_ls_alpha_differences_by_size.csv",
        "table_q5_alpha_levels_by_size": output_dir / "table_q5_alpha_levels_by_size.csv",
        "table_q5_alpha_differences_by_size": output_dir / "table_q5_alpha_differences_by_size.csv",
        "monthly_portfolio_returns_used_by_size": output_dir / "monthly_portfolio_returns_used_by_size.csv",
        "risk_adjusted_table_preview_by_size": output_dir / "risk_adjusted_table_preview_by_size.csv",
    }

    ls_levels.to_csv(outputs["table_ls_alpha_levels_by_size"], index=False)
    ls_diffs.to_csv(outputs["table_ls_alpha_differences_by_size"], index=False)
    q5_levels.to_csv(outputs["table_q5_alpha_levels_by_size"], index=False)
    q5_diffs.to_csv(outputs["table_q5_alpha_differences_by_size"], index=False)
    monthly_used.to_csv(outputs["monthly_portfolio_returns_used_by_size"], index=False)
    preview.to_csv(outputs["risk_adjusted_table_preview_by_size"], index=False)

    return outputs


def save_ucits_audit_outputs(
    output_dir: Path,
    ucits_holdings: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "size_split_ucits_weight_monthly_holdings": output_dir / "size_split_ucits_weight_monthly_holdings.csv",
        "size_split_ucits_weight_diagnostics": output_dir / "size_split_ucits_weight_diagnostics.csv",
    }
    ucits_holdings.to_csv(outputs["size_split_ucits_weight_monthly_holdings"], index=False)
    diagnostics.to_csv(outputs["size_split_ucits_weight_diagnostics"], index=False)
    return outputs


def save_cumulative_return_plots(monthly_used: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cumulative = monthly_used.copy()
    cumulative["Date"] = pd.to_datetime(cumulative["Date"], errors="coerce")
    cumulative["Return"] = pd.to_numeric(cumulative["Return"], errors="coerce")
    cumulative = cumulative.dropna(subset=["SizeGroup", "Date", "Method", "PortfolioStrategy", "Return"])
    cumulative = cumulative.sort_values(["SizeGroup", "PortfolioStrategy", "Method", "Date"])
    cumulative["CumulativeReturn"] = (
        cumulative.groupby(["SizeGroup", "PortfolioStrategy", "Method"])["Return"]
        .transform(lambda s: (1.0 + s).cumprod() - 1.0)
    )
    outputs: dict[str, Path] = {}
    strategy_titles = {
        "LongShort": "Cumulative Returns: Long-Short Q5 - Q1",
        "Q5": "Cumulative Returns: Pure Q5",
    }

    for size_group in SIZE_GROUPS:
        size_sub = cumulative.loc[cumulative["SizeGroup"] == size_group].copy()
        for strategy, title in strategy_titles.items():
            sub = size_sub.loc[size_sub["PortfolioStrategy"] == strategy].copy()
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

            ax.set_title(f"{title} ({size_group})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative return")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()

            filename = f"cumulative_returns_{strategy.lower()}_{size_group.lower()}.png"
            path = plot_dir / filename
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            outputs[f"{strategy}_{size_group}"] = path

    return outputs


def print_identification(
    run_dir: Path,
    portfolio_eval_dir: Path,
    portfolio_source: Path,
    factors_csv: Path,
    output_dir: Path,
    nw_lags: int,
) -> None:
    print("\nIdentified inputs and reused helpers")
    print(f"  run_dir: {run_dir}")
    print(f"  portfolio_evaluation_dir: {portfolio_eval_dir}")
    print(f"  size-split constituent source: {portfolio_source}")
    print("  size split: median monthly LagMarketCap by Ticker x FormationYear, ranked within FormationYear")
    print("  weighting rule: UCITS-style 5/10/40 within SizeGroup x Method x Portfolio x Date")
    print(f"    single issuer cap: {ucits.UCITS_SINGLE_ISSUER_CAP:.2%}")
    print(f"    large position threshold: {ucits.UCITS_LARGE_POSITION_THRESHOLD:.2%}")
    print(f"    aggregate cap for positions above threshold: {ucits.UCITS_LARGE_POSITION_AGGREGATE_CAP:.2%}")
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
    portfolio_source, portfolio_eval_dir = choose_constituent_source(
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
        output_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_size_split_ucits_5_10_40"

    print_identification(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        portfolio_source=portfolio_source,
        factors_csv=factors_csv,
        output_dir=output_dir,
        nw_lags=args.nw_lags,
    )

    factors = load_factor_data(factors_csv)
    rf = factors["RF"].copy()
    zero_rf = pd.Series(0.0, index=rf.index, name="RF")
    monthly_returns, ucits_holdings, diagnostics = load_size_split_ucits_weighted_monthly_returns(
        portfolio_source,
    )

    ls_levels, ls_diffs, q5_levels, q5_diffs, monthly_used = run_size_group_analysis(
        monthly_returns=monthly_returns,
        factors=factors,
        rf=rf,
        zero_rf=zero_rf,
        nw_lags=args.nw_lags,
    )

    assert_size_split_shapes(ls_levels, ls_diffs, q5_levels, q5_diffs)
    preview = build_size_split_preview(
        levels=pd.concat([ls_levels, q5_levels], ignore_index=True),
        differences=pd.concat([ls_diffs, q5_diffs], ignore_index=True),
    )

    outputs = save_outputs(
        output_dir=output_dir,
        ls_levels=ls_levels,
        ls_diffs=ls_diffs,
        q5_levels=q5_levels,
        q5_diffs=q5_diffs,
        monthly_used=monthly_used,
        preview=preview,
    )
    audit_outputs = save_ucits_audit_outputs(
        output_dir=output_dir,
        ucits_holdings=ucits_holdings,
        diagnostics=diagnostics,
    )
    plot_outputs = save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    print("\nUCITS-weight diagnostics")
    print(f"  UCITS holdings rows: {len(ucits_holdings)}")
    print(f"  monthly portfolio groups: {len(diagnostics)}")
    print(f"  largest UCITS weight: {diagnostics['max_ucits_weight'].max():.4%}")
    print(f"  largest aggregate weight above threshold: {diagnostics['large_position_weight'].max():.4%}")
    print(f"  groups where aggregate cap was relaxed: {int(diagnostics['aggregate_cap_relaxed'].sum())}")
    print(f"  largest cash weight: {diagnostics['cash_weight'].max():.4%}")

    print("\nCreated CSV files")
    row_counts = {
        "table_ls_alpha_levels_by_size": len(ls_levels),
        "table_ls_alpha_differences_by_size": len(ls_diffs),
        "table_q5_alpha_levels_by_size": len(q5_levels),
        "table_q5_alpha_differences_by_size": len(q5_diffs),
        "monthly_portfolio_returns_used_by_size": len(monthly_used),
        "risk_adjusted_table_preview_by_size": len(preview),
    }
    for key, path in outputs.items():
        print(f"  {path} ({row_counts[key]} rows)")

    print("\nCreated audit CSV files")
    audit_counts = {
        "size_split_ucits_weight_monthly_holdings": len(ucits_holdings),
        "size_split_ucits_weight_diagnostics": len(diagnostics),
    }
    for key, path in audit_outputs.items():
        print(f"  {path} ({audit_counts[key]} rows)")

    print("\nCreated plot files")
    for path in plot_outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
