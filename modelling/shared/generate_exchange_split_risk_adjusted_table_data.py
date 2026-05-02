from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_capped_weight_risk_adjusted_table_data as ucits
import generate_risk_adjusted_table_data as vw
from helper_functions import find_project_root, load_factor_data, parse_month_series, resolve_path


EXCHANGE_CODE_LABELS = {
    "CO": "Copenhagen",
    "HE": "Helsinki",
    "OL": "Oslo",
    "ST": "Stockholm",
    "IC": "Iceland",
}
EXCHANGES = ["Copenhagen", "Helsinki", "Oslo", "Stockholm"]
EXCLUDED_EXCHANGES = {"Iceland"}
PORTFOLIOS_USED = ["Q1", "Q5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data for UCITS-weighted Q5/Q1 risk-adjusted "
            "performance separately by stock exchange."
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
        help="Optional constituent CSV. Must contain Ticker, Method, Portfolio, Date, Return, and LagMarketCap.",
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
            "<portfolio_evaluation_dir>/thesis_risk_adjusted_tables_exchange_split_ucits_5_10_40."
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


def exchange_slug(exchange: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", exchange.lower()).strip("_")


def add_exchange_labels(holdings: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    out = holdings.copy()
    out["ExchangeCode"] = out["Ticker"].astype(str).str.extract(r"\.([A-Z]+)$", expand=False)

    missing_suffix = out.loc[out["ExchangeCode"].isna(), "Ticker"].drop_duplicates().head(10)
    if not missing_suffix.empty:
        raise ValueError(
            "Could not infer exchange from ticker suffix for some holdings.\n"
            f"Source: {source_path}\n"
            "Examples:\n" + "\n".join(missing_suffix.astype(str))
        )

    unknown_codes = sorted(set(out["ExchangeCode"].dropna()) - set(EXCHANGE_CODE_LABELS))
    if unknown_codes:
        raise ValueError(
            "Found ticker suffixes that are not mapped to exchanges.\n"
            f"Source: {source_path}\n"
            f"Unknown suffixes: {unknown_codes}\n"
            f"Known suffix mapping: {EXCHANGE_CODE_LABELS}"
        )

    out["Exchange"] = out["ExchangeCode"].map(EXCHANGE_CODE_LABELS)
    return out


def build_exchange_summary(holdings: pd.DataFrame) -> pd.DataFrame:
    summary = (
        holdings.groupby(["ExchangeCode", "Exchange"], as_index=False)
        .agg(
            holding_rows=("Ticker", "size"),
            unique_firms=("Ticker", "nunique"),
            months=("Date", "nunique"),
        )
        .sort_values(["Exchange", "ExchangeCode"])
        .reset_index(drop=True)
    )
    summary["included"] = ~summary["Exchange"].isin(EXCLUDED_EXCHANGES)
    return summary


def split_infeasible_groups(
    holdings: pd.DataFrame,
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_firms = holdings.groupby(group_cols)["Ticker"].nunique()
    infeasible = n_firms[n_firms * ucits.UCITS_SINGLE_ISSUER_CAP < 1 - 1e-12]
    if infeasible.empty:
        return holdings.copy(), pd.DataFrame(columns=group_cols + ["n_firms", "reason"])

    infeasible_groups = infeasible.reset_index(name="n_firms")
    infeasible_groups["reason"] = (
        "fewer than "
        + str(int(round(1 / ucits.UCITS_SINGLE_ISSUER_CAP)))
        + " names, so a fully invested 10% single-issuer cap is impossible"
    )

    marked = holdings.merge(
        infeasible_groups[group_cols].assign(_infeasible=True),
        on=group_cols,
        how="left",
    )
    feasible = marked.loc[marked["_infeasible"].isna()].drop(columns="_infeasible").copy()
    return feasible, infeasible_groups.sort_values(group_cols).reset_index(drop=True)


def load_exchange_split_ucits_weighted_monthly_returns(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    required = ["Ticker", "Method", "Portfolio", "Date", "Return", "LagMarketCap"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required constituent columns for exchange-split value weighting: {missing}"
        )

    out = df[required].copy()
    out["Date"] = parse_month_series(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out["Method"] = out["Method"].astype(str).str.strip()
    out["Portfolio"] = out["Portfolio"].astype(str).str.strip()
    out["Return"] = pd.to_numeric(out["Return"], errors="coerce")
    out["LagMarketCap"] = pd.to_numeric(out["LagMarketCap"], errors="coerce")
    out = out.dropna(subset=["Ticker", "Date", "Method", "Portfolio", "Return", "LagMarketCap"])
    out = out.loc[out["LagMarketCap"] > 0].copy()
    out = add_exchange_labels(out, path)

    exchange_summary = build_exchange_summary(out)
    out = out.loc[~out["Exchange"].isin(EXCLUDED_EXCHANGES)].copy()
    out = out.loc[out["Method"].isin(vw.METHODS) & out["Portfolio"].isin(PORTFOLIOS_USED)].copy()
    if out.empty:
        raise ValueError(
            f"{path} has no usable rows after excluding {sorted(EXCLUDED_EXCHANGES)} "
            f"and keeping methods={vw.METHODS}, portfolios={PORTFOLIOS_USED}."
        )

    available_exchanges = sorted(out["Exchange"].dropna().unique())
    missing_exchanges = [exchange for exchange in EXCHANGES if exchange not in available_exchanges]
    if missing_exchanges:
        raise ValueError(
            "Some expected non-Iceland exchanges were not found in the holdings.\n"
            f"Missing: {missing_exchanges}\n"
            f"Available: {available_exchanges}"
        )

    group_cols = ["Exchange", "Method", "Portfolio", "Date"]
    feasible, infeasible_groups = split_infeasible_groups(out, group_cols)
    if feasible.empty:
        raise ValueError(
            "All exchange-split portfolio-month groups are infeasible under the 10% single-issuer cap."
        )

    ucits_holdings = pd.concat(
        [
            ucits.apply_ucits_weights(
                group,
                single_issuer_cap=ucits.UCITS_SINGLE_ISSUER_CAP,
                large_position_threshold=ucits.UCITS_LARGE_POSITION_THRESHOLD,
                large_position_aggregate_cap=ucits.UCITS_LARGE_POSITION_AGGREGATE_CAP,
            )
            for _, group in feasible.groupby(group_cols, sort=False)
        ],
        ignore_index=True,
    )

    monthly = (
        ucits_holdings.groupby(group_cols, as_index=False)
        .agg(
            Return=("UCITSWeightedReturn", "sum"),
            n_firms=("Ticker", "nunique"),
            total_lag_mcap=("LagMarketCap", "sum"),
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
        .sort_values(["Exchange", "Method", "Portfolio", "Date"])
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
        .sort_values(["Exchange", "Date", "Method", "Portfolio"])
        .reset_index(drop=True)
    )

    return monthly, ucits_holdings, diagnostics, exchange_summary, infeasible_groups


def build_strategy_returns_for_exchange(
    monthly_returns: pd.DataFrame,
    factors: pd.DataFrame,
    exchange: str,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.DataFrame]:
    sub = monthly_returns.loc[monthly_returns["Exchange"] == exchange].copy()
    q5_returns, ls_returns, used = vw.build_strategy_returns(sub, factors)
    used.insert(0, "Exchange", exchange)
    return q5_returns, ls_returns, used


def add_exchange(df: pd.DataFrame, exchange: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "Exchange", exchange)
    return out


def run_exchange_analysis(
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

    for exchange in EXCHANGES:
        q5_returns, ls_returns, used = build_strategy_returns_for_exchange(
            monthly_returns=monthly_returns,
            factors=factors,
            exchange=exchange,
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
        ls_level_frames.append(add_exchange(ls_levels, exchange))
        ls_diff_frames.append(add_exchange(ls_diffs, exchange))
        q5_level_frames.append(add_exchange(q5_levels, exchange))
        q5_diff_frames.append(add_exchange(q5_diffs, exchange))
        used_frames.append(used)

    return (
        pd.concat(ls_level_frames, ignore_index=True),
        pd.concat(ls_diff_frames, ignore_index=True),
        pd.concat(q5_level_frames, ignore_index=True),
        pd.concat(q5_diff_frames, ignore_index=True),
        pd.concat(used_frames, ignore_index=True),
    )


def build_exchange_preview(levels: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    level_wide = levels.pivot_table(
        index=["Exchange", "PortfolioStrategy", "FactorModel"],
        columns="Method",
        values=["alpha_annualized", "t_stat", "p_value"],
        aggfunc="first",
    )
    level_wide.columns = [f"{method}_{metric}" for metric, method in level_wide.columns]
    level_wide = level_wide.reset_index()

    diff_wide = differences.pivot_table(
        index=["Exchange", "PortfolioStrategy", "FactorModel"],
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

    return level_wide.merge(diff_wide, on=["Exchange", "PortfolioStrategy", "FactorModel"], how="left")


def assert_exchange_shapes(
    ls_levels: pd.DataFrame,
    ls_diffs: pd.DataFrame,
    q5_levels: pd.DataFrame,
    q5_diffs: pd.DataFrame,
) -> None:
    n_exchanges = len(EXCHANGES)
    n_level_rows = n_exchanges * len(vw.INTERNAL_MODELS) * len(vw.METHODS)
    n_difference_rows = n_exchanges * len(vw.INTERNAL_MODELS) * (len(vw.METHODS) - 1)
    expected = {
        "Long-short alpha levels": (ls_levels, n_level_rows),
        "Long-short alpha differences": (ls_diffs, n_difference_rows),
        "Q5 alpha levels": (q5_levels, n_level_rows),
        "Q5 alpha differences": (q5_diffs, n_difference_rows),
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
        "table_ls_alpha_levels_by_exchange": output_dir / "table_ls_alpha_levels_by_exchange.csv",
        "table_ls_alpha_differences_by_exchange": output_dir / "table_ls_alpha_differences_by_exchange.csv",
        "table_q5_alpha_levels_by_exchange": output_dir / "table_q5_alpha_levels_by_exchange.csv",
        "table_q5_alpha_differences_by_exchange": output_dir / "table_q5_alpha_differences_by_exchange.csv",
        "monthly_portfolio_returns_used_by_exchange": output_dir / "monthly_portfolio_returns_used_by_exchange.csv",
        "risk_adjusted_table_preview_by_exchange": output_dir / "risk_adjusted_table_preview_by_exchange.csv",
    }

    ls_levels.to_csv(outputs["table_ls_alpha_levels_by_exchange"], index=False)
    ls_diffs.to_csv(outputs["table_ls_alpha_differences_by_exchange"], index=False)
    q5_levels.to_csv(outputs["table_q5_alpha_levels_by_exchange"], index=False)
    q5_diffs.to_csv(outputs["table_q5_alpha_differences_by_exchange"], index=False)
    monthly_used.to_csv(outputs["monthly_portfolio_returns_used_by_exchange"], index=False)
    preview.to_csv(outputs["risk_adjusted_table_preview_by_exchange"], index=False)

    return outputs


def save_audit_outputs(
    output_dir: Path,
    ucits_holdings: pd.DataFrame,
    diagnostics: pd.DataFrame,
    exchange_summary: pd.DataFrame,
    infeasible_groups: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "exchange_ucits_weight_monthly_holdings": output_dir / "exchange_ucits_weight_monthly_holdings.csv",
        "exchange_ucits_weight_diagnostics": output_dir / "exchange_ucits_weight_diagnostics.csv",
        "exchange_summary": output_dir / "exchange_summary.csv",
        "exchange_infeasible_groups_dropped": output_dir / "exchange_infeasible_groups_dropped.csv",
    }
    ucits_holdings.to_csv(outputs["exchange_ucits_weight_monthly_holdings"], index=False)
    diagnostics.to_csv(outputs["exchange_ucits_weight_diagnostics"], index=False)
    exchange_summary.to_csv(outputs["exchange_summary"], index=False)
    infeasible_groups.to_csv(outputs["exchange_infeasible_groups_dropped"], index=False)
    return outputs


def save_cumulative_return_plots(monthly_used: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cumulative = monthly_used.copy()
    cumulative["Date"] = pd.to_datetime(cumulative["Date"], errors="coerce")
    cumulative["Return"] = pd.to_numeric(cumulative["Return"], errors="coerce")
    cumulative = cumulative.dropna(subset=["Exchange", "Date", "Method", "PortfolioStrategy", "Return"])
    cumulative = cumulative.sort_values(["Exchange", "PortfolioStrategy", "Method", "Date"])
    cumulative["CumulativeReturn"] = (
        cumulative.groupby(["Exchange", "PortfolioStrategy", "Method"])["Return"]
        .transform(lambda s: (1.0 + s).cumprod() - 1.0)
    )

    outputs: dict[str, Path] = {}
    strategy_titles = {
        "LongShort": "Cumulative Returns: Long-Short Q5 - Q1",
        "Q5": "Cumulative Returns: Pure Q5",
    }

    for exchange in EXCHANGES:
        exchange_sub = cumulative.loc[cumulative["Exchange"] == exchange].copy()
        for strategy, title in strategy_titles.items():
            sub = exchange_sub.loc[exchange_sub["PortfolioStrategy"] == strategy].copy()
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
                    label=vw.METHOD_DISPLAY_LABELS.get(method, method),
                    color=vw.METHOD_COLORS.get(method),
                    linewidth=2.1,
                )

            ax.set_title(f"{title} ({exchange})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative return")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()

            filename = f"cumulative_returns_{strategy.lower()}_{exchange_slug(exchange)}.png"
            path = plot_dir / filename
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            outputs[f"{strategy}_{exchange}"] = path

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
    print(f"  exchange-split constituent source: {portfolio_source}")
    print(f"  exchange source: ticker suffix mapping {EXCHANGE_CODE_LABELS}")
    print(f"  included exchanges: {EXCHANGES}")
    print(f"  excluded exchanges: {sorted(EXCLUDED_EXCHANGES)}")
    print("  weighting rule: 10% single-issuer cap, with 5/10/40 used where feasible")
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
    portfolio_source, portfolio_eval_dir = ucits.choose_constituent_source(
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
        output_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_exchange_split_ucits_5_10_40"

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
    monthly_returns, ucits_holdings, diagnostics, exchange_summary, infeasible_groups = (
        load_exchange_split_ucits_weighted_monthly_returns(portfolio_source)
    )

    ls_levels, ls_diffs, q5_levels, q5_diffs, monthly_used = run_exchange_analysis(
        monthly_returns=monthly_returns,
        factors=factors,
        rf=rf,
        zero_rf=zero_rf,
        nw_lags=args.nw_lags,
    )

    assert_exchange_shapes(ls_levels, ls_diffs, q5_levels, q5_diffs)
    preview = build_exchange_preview(
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
    audit_outputs = save_audit_outputs(
        output_dir=output_dir,
        ucits_holdings=ucits_holdings,
        diagnostics=diagnostics,
        exchange_summary=exchange_summary,
        infeasible_groups=infeasible_groups,
    )
    plot_outputs = save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    diagnostic_summary = (
        diagnostics.groupby("Exchange", as_index=False)
        .agg(
            monthly_groups=("Exchange", "size"),
            min_n_firms=("n_firms", "min"),
            max_ucits_weight=("max_ucits_weight", "max"),
            max_large_position_weight=("large_position_weight", "max"),
            relaxed_groups=("aggregate_cap_relaxed", "sum"),
        )
        .sort_values("Exchange")
    )

    print("\nExchange split diagnostics")
    print(f"  UCITS holdings rows: {len(ucits_holdings)}")
    print(f"  monthly portfolio groups used: {len(diagnostics)}")
    print(f"  infeasible groups dropped before regressions: {len(infeasible_groups)}")
    print(f"  largest UCITS weight: {diagnostics['max_ucits_weight'].max():.4%}")
    print(f"  largest aggregate weight above threshold: {diagnostics['large_position_weight'].max():.4%}")
    print(f"  groups where aggregate cap was relaxed: {int(diagnostics['aggregate_cap_relaxed'].sum())}")
    print(f"  largest cash weight: {diagnostics['cash_weight'].max():.4%}")
    print("\nBy-exchange diagnostics")
    print(diagnostic_summary.to_string(index=False))

    print("\nCreated CSV files")
    row_counts = {
        "table_ls_alpha_levels_by_exchange": len(ls_levels),
        "table_ls_alpha_differences_by_exchange": len(ls_diffs),
        "table_q5_alpha_levels_by_exchange": len(q5_levels),
        "table_q5_alpha_differences_by_exchange": len(q5_diffs),
        "monthly_portfolio_returns_used_by_exchange": len(monthly_used),
        "risk_adjusted_table_preview_by_exchange": len(preview),
    }
    for key, path in outputs.items():
        print(f"  {path} ({row_counts[key]} rows)")

    print("\nCreated audit CSV files")
    audit_counts = {
        "exchange_ucits_weight_monthly_holdings": len(ucits_holdings),
        "exchange_ucits_weight_diagnostics": len(diagnostics),
        "exchange_summary": len(exchange_summary),
        "exchange_infeasible_groups_dropped": len(infeasible_groups),
    }
    for key, path in audit_outputs.items():
        print(f"  {path} ({audit_counts[key]} rows)")

    print("\nCreated plot files")
    for path in plot_outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
