from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_capped_weight_risk_adjusted_table_data as ucits
import generate_exchange_neutral_risk_adjusted_table_data as exchange_neutral
import generate_exchange_split_risk_adjusted_table_data as exchange_split
import generate_risk_adjusted_table_data as vw
from helper_functions import build_monthly_portfolio_returns, find_project_root, load_factor_data, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data by exchange after forming Q1-Q5 within each "
            "FormationYear x Exchange x Method."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional results run directory, e.g. results/current_res.",
    )
    parser.add_argument(
        "--latent-source",
        type=Path,
        default=None,
        help="Optional latent_prof_firm_year.csv source with sorting signals.",
    )
    parser.add_argument(
        "--stock-prices-csv",
        type=Path,
        default=None,
        help="Optional monthly stock prices CSV. Defaults to run_config.json.",
    )
    parser.add_argument(
        "--market-cap-csv",
        type=Path,
        default=None,
        help="Optional monthly market-cap CSV. Defaults to run_config.json.",
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
            "<portfolio_evaluation_dir>/thesis_risk_adjusted_tables_exchange_neutral_by_exchange_ucits_5_10_40."
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


def build_ucits_monthly_returns_by_exchange(
    monthly_holdings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["Ticker", "Exchange", "Method", "Portfolio", "Date", "Return", "LagMarketCap"]
    missing = [c for c in required if c not in monthly_holdings.columns]
    if missing:
        raise ValueError(f"Exchange-neutral by-exchange monthly holdings missing required columns: {missing}")

    out = monthly_holdings[required].copy()
    out = out.loc[
        out["Exchange"].isin(exchange_split.EXCHANGES)
        & out["Method"].isin(vw.METHODS)
        & out["Portfolio"].isin(exchange_split.PORTFOLIOS_USED)
    ].copy()
    out = out.dropna(subset=["Ticker", "Exchange", "Method", "Portfolio", "Date", "Return", "LagMarketCap"])
    out = out.loc[out["LagMarketCap"] > 0].copy()

    if out.empty:
        raise ValueError("No usable Q1/Q5 monthly holdings after exchange-neutral by-exchange filtering.")

    group_cols = ["Exchange", "Method", "Portfolio", "Date"]
    feasible, infeasible_groups = exchange_split.split_infeasible_groups(out, group_cols)
    if feasible.empty:
        raise ValueError(
            "All exchange-neutral by-exchange portfolio-month groups are infeasible under the 10% cap."
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

    return monthly, ucits_holdings, diagnostics, infeasible_groups


def save_auxiliary_outputs(
    output_dir: Path,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    skipped_groups: pd.DataFrame,
    exchange_summary: pd.DataFrame,
    ucits_holdings: pd.DataFrame,
    diagnostics: pd.DataFrame,
    infeasible_groups: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "exchange_neutral_by_exchange_portfolio_assignments_long": output_dir
        / "exchange_neutral_by_exchange_portfolio_assignments_long.csv",
        "exchange_neutral_by_exchange_portfolio_formation_summary": output_dir
        / "exchange_neutral_by_exchange_portfolio_formation_summary.csv",
        "exchange_neutral_by_exchange_skipped_exchange_years": output_dir
        / "exchange_neutral_by_exchange_skipped_exchange_years.csv",
        "exchange_neutral_by_exchange_exchange_summary": output_dir
        / "exchange_neutral_by_exchange_exchange_summary.csv",
        "exchange_neutral_by_exchange_ucits_weight_monthly_holdings": output_dir
        / "exchange_neutral_by_exchange_ucits_weight_monthly_holdings.csv",
        "exchange_neutral_by_exchange_ucits_weight_diagnostics": output_dir
        / "exchange_neutral_by_exchange_ucits_weight_diagnostics.csv",
        "exchange_neutral_by_exchange_infeasible_groups_dropped": output_dir
        / "exchange_neutral_by_exchange_infeasible_groups_dropped.csv",
    }

    assignments.to_csv(outputs["exchange_neutral_by_exchange_portfolio_assignments_long"], index=False)
    summary.to_csv(outputs["exchange_neutral_by_exchange_portfolio_formation_summary"], index=False)
    skipped_groups.to_csv(outputs["exchange_neutral_by_exchange_skipped_exchange_years"], index=False)
    exchange_summary.to_csv(outputs["exchange_neutral_by_exchange_exchange_summary"], index=False)
    ucits_holdings.to_csv(outputs["exchange_neutral_by_exchange_ucits_weight_monthly_holdings"], index=False)
    diagnostics.to_csv(outputs["exchange_neutral_by_exchange_ucits_weight_diagnostics"], index=False)
    infeasible_groups.to_csv(outputs["exchange_neutral_by_exchange_infeasible_groups_dropped"], index=False)
    return outputs


def print_identification(
    run_dir: Path,
    portfolio_eval_dir: Path,
    latent_source: Path,
    stock_prices_csv: Path,
    market_cap_csv: Path,
    factors_csv: Path,
    output_dir: Path,
    nw_lags: int,
) -> None:
    print("\nIdentified inputs and reused helpers")
    print(f"  run_dir: {run_dir}")
    print(f"  portfolio_evaluation_dir used for defaults: {portfolio_eval_dir}")
    print(f"  exchange-neutral firm-year source: {latent_source}")
    print(f"  exchange source: ticker suffix mapping {exchange_split.EXCHANGE_CODE_LABELS}")
    print(f"  included exchanges: {exchange_split.EXCHANGES}")
    print(f"  excluded exchanges: {sorted(exchange_split.EXCLUDED_EXCHANGES)}")
    print(f"  monthly stock prices: {stock_prices_csv}")
    print(f"  monthly market caps: {market_cap_csv}")
    print(f"  monthly factor returns: {factors_csv}")
    print("  quantile rule: sort within FormationYear x Exchange x Method")
    print("  output rule: keep each exchange as a separate return series")
    print("  weighting rule: 10% single-issuer cap, with 5/10/40 used where feasible")
    print(f"    single issuer cap: {ucits.UCITS_SINGLE_ISSUER_CAP:.2%}")
    print(f"    large position threshold: {ucits.UCITS_LARGE_POSITION_THRESHOLD:.2%}")
    print(f"    aggregate cap for positions above threshold: {ucits.UCITS_LARGE_POSITION_AGGREGATE_CAP:.2%}")
    print("  quantile helper: modelling/shared/portfolio_formation.py::assign_quantile_portfolios")
    print("  portfolio aggregation helper: modelling/shared/helper_functions.py::build_monthly_portfolio_returns")
    print("  risk-adjusted regression helper: modelling/shared/step5_evaluation.py::risk_adjusted_performance")
    print("  Newey-West/HAC helper: modelling/shared/step5_evaluation.py::_ols_newey_west_full")
    print("  alpha-difference helper: modelling/shared/step5_evaluation.py::alpha_differences")
    print(f"  HAC lags: {nw_lags}")
    print(f"  output_dir: {output_dir}")


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    run_dir = vw.choose_run_dir(project_root, resolve_cli_path(args.run_dir, project_root))
    portfolio_eval_dir = vw.portfolio_eval_dir_for_run(run_dir)
    if portfolio_eval_dir is None:
        portfolio_eval_dir = run_dir / "portfolio_evaluation"

    latent_source = exchange_neutral.choose_latent_source(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        requested_source=resolve_cli_path(args.latent_source, project_root),
    )
    stock_prices_csv = exchange_neutral.choose_run_config_path(
        project_root=project_root,
        run_dir=run_dir,
        requested_path=resolve_cli_path(args.stock_prices_csv, project_root),
        config_key="returns_csv",
        fallback="data/processed_data_lseg/all_stock_prices_nok.csv",
    )
    market_cap_csv = exchange_neutral.choose_run_config_path(
        project_root=project_root,
        run_dir=run_dir,
        requested_path=resolve_cli_path(args.market_cap_csv, project_root),
        config_key="market_cap_csv",
        fallback="data/processed_data_lseg/historical_market_cap_nok.csv",
    )
    factors_csv = vw.choose_factor_csv(
        project_root=project_root,
        run_dir=run_dir,
        requested_factors=resolve_cli_path(args.factors_csv, project_root),
    )
    output_dir = resolve_cli_path(args.output_dir, project_root)
    if output_dir is None:
        output_dir = (
            portfolio_eval_dir / "thesis_risk_adjusted_tables_exchange_neutral_by_exchange_ucits_5_10_40"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print_identification(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        latent_source=latent_source,
        stock_prices_csv=stock_prices_csv,
        market_cap_csv=market_cap_csv,
        factors_csv=factors_csv,
        output_dir=output_dir,
        nw_lags=args.nw_lags,
    )

    firm_year, exchange_summary = exchange_neutral.load_exchange_firm_year(latent_source)
    assignments, skipped_groups = exchange_neutral.form_exchange_neutral_assignments(firm_year)
    summary = exchange_neutral.build_exchange_neutral_summary(assignments)

    factors = load_factor_data(factors_csv)
    prepared = build_monthly_portfolio_returns(
        assignments=assignments,
        stock_prices_csv=stock_prices_csv,
        market_cap_csv=market_cap_csv,
        factors=factors,
        n_portfolios=exchange_neutral.N_PORTFOLIOS,
    )
    monthly_holdings = prepared["monthly_holdings"]
    monthly_returns, ucits_holdings, diagnostics, infeasible_groups = (
        build_ucits_monthly_returns_by_exchange(monthly_holdings)
    )

    rf = factors["RF"].copy()
    zero_rf = pd.Series(0.0, index=rf.index, name="RF")
    ls_levels, ls_diffs, q5_levels, q5_diffs, monthly_used = exchange_split.run_exchange_analysis(
        monthly_returns=monthly_returns,
        factors=factors,
        rf=rf,
        zero_rf=zero_rf,
        nw_lags=args.nw_lags,
    )

    exchange_split.assert_exchange_shapes(ls_levels, ls_diffs, q5_levels, q5_diffs)
    preview = exchange_split.build_exchange_preview(
        levels=pd.concat([ls_levels, q5_levels], ignore_index=True),
        differences=pd.concat([ls_diffs, q5_diffs], ignore_index=True),
    )

    table_outputs = exchange_split.save_outputs(
        output_dir=output_dir,
        ls_levels=ls_levels,
        ls_diffs=ls_diffs,
        q5_levels=q5_levels,
        q5_diffs=q5_diffs,
        monthly_used=monthly_used,
        preview=preview,
    )
    auxiliary_outputs = save_auxiliary_outputs(
        output_dir=output_dir,
        assignments=assignments,
        summary=summary,
        skipped_groups=skipped_groups,
        exchange_summary=exchange_summary,
        ucits_holdings=ucits_holdings,
        diagnostics=diagnostics,
        infeasible_groups=infeasible_groups,
    )
    plot_outputs = exchange_split.save_cumulative_return_plots(
        monthly_used=monthly_used,
        output_dir=output_dir,
    )

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

    print("\nExchange-neutral by-exchange assignment summary")
    print(f"  firm-year rows loaded after excluding Iceland: {len(firm_year)}")
    print(f"  assignment rows created: {len(assignments)}")
    print(f"  valid assignment rows: {int(assignments['PortfolioNum'].notna().sum())}")
    print(f"  skipped exchange-year-method groups: {len(skipped_groups)}")

    print("\nUCITS-weight diagnostics")
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
    for key, path in table_outputs.items():
        print(f"  {path} ({row_counts[key]} rows)")

    print("\nCreated audit CSV files")
    audit_counts = {
        "exchange_neutral_by_exchange_portfolio_assignments_long": len(assignments),
        "exchange_neutral_by_exchange_portfolio_formation_summary": len(summary),
        "exchange_neutral_by_exchange_skipped_exchange_years": len(skipped_groups),
        "exchange_neutral_by_exchange_exchange_summary": len(exchange_summary),
        "exchange_neutral_by_exchange_ucits_weight_monthly_holdings": len(ucits_holdings),
        "exchange_neutral_by_exchange_ucits_weight_diagnostics": len(diagnostics),
        "exchange_neutral_by_exchange_infeasible_groups_dropped": len(infeasible_groups),
    }
    for key, path in auxiliary_outputs.items():
        print(f"  {path} ({audit_counts[key]} rows)")

    print("\nCreated plot files")
    for path in plot_outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
