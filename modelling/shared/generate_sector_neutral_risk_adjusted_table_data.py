from __future__ import annotations

import argparse
import json
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

import generate_risk_adjusted_table_data as vw
from helper_functions import build_monthly_portfolio_returns, find_project_root, load_factor_data, resolve_path
from portfolio_formation import METHOD_SPECS, assign_quantile_portfolios, clean_input


SECTOR_COL = "Sector"
N_PORTFOLIOS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data for value-weighted Q5/Q1 risk-adjusted "
            "performance after forming sector-neutral quintiles."
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
        help="Optional latent_prof_firm_year.csv source with Sector and sorting signals.",
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
            "<portfolio_evaluation_dir>/thesis_risk_adjusted_tables_sector_neutral."
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


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_variant_name(portfolio_eval_dir: Path) -> str | None:
    if portfolio_eval_dir.parent.name == "portfolio_evaluation":
        return portfolio_eval_dir.name
    return None


def choose_latent_source(
    run_dir: Path,
    portfolio_eval_dir: Path,
    requested_source: Path | None,
) -> Path:
    searched: list[Path] = []

    if requested_source is not None:
        if requested_source.exists():
            return requested_source
        raise FileNotFoundError(f"Requested latent source does not exist: {requested_source}")

    variant = infer_variant_name(portfolio_eval_dir)
    if variant is not None:
        candidate = run_dir / "latent_prof_model" / variant / "latent_prof_firm_year.csv"
        searched.append(candidate)
        if candidate.exists():
            return candidate

    preferred = run_dir / "latent_prof_model" / "HB" / "latent_prof_firm_year.csv"
    searched.append(preferred)
    if preferred.exists():
        return preferred

    globbed = sorted((run_dir / "latent_prof_model").glob("*/latent_prof_firm_year.csv"))
    searched.extend(globbed)
    if globbed:
        return globbed[0]

    raise FileNotFoundError(
        "Could not locate a latent firm-year source with sectors and sorting signals.\n"
        "Searched:\n" + "\n".join(str(p) for p in searched)
    )


def choose_run_config_path(
    project_root: Path,
    run_dir: Path,
    requested_path: Path | None,
    config_key: str,
    fallback: str,
) -> Path:
    if requested_path is not None:
        if requested_path.exists():
            return requested_path
        raise FileNotFoundError(f"Requested {config_key} path does not exist: {requested_path}")

    run_config = read_json_if_exists(run_dir / "run_config.json")
    searched: list[Path] = []

    if config_key in run_config:
        candidate = resolve_path(run_config[config_key], project_root)
        searched.append(candidate)
        if candidate.exists():
            return candidate

    fallback_path = resolve_path(fallback, project_root)
    searched.append(fallback_path)
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Could not locate {config_key}.\nSearched:\n" + "\n".join(str(p) for p in searched)
    )


def validate_sector_source(df: pd.DataFrame, source_path: Path) -> None:
    required = [SECTOR_COL]
    for method in vw.METHODS:
        if method not in METHOD_SPECS:
            raise ValueError(f"{method} is missing from portfolio_formation.METHOD_SPECS.")
        required.append(METHOD_SPECS[method]["signal_col"])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_path} is missing required sector-neutral columns: {missing}")


def load_latent_firm_year(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_sector_source(df, path)

    df = clean_input(df)
    df[SECTOR_COL] = df[SECTOR_COL].astype(str).str.strip()
    df = df[df[SECTOR_COL].notna() & (df[SECTOR_COL] != "") & (df[SECTOR_COL].str.lower() != "nan")].copy()

    if df.empty:
        raise ValueError(f"{path} has no usable rows with non-missing {SECTOR_COL}.")

    return df


def form_sector_neutral_assignments(
    firm_year: pd.DataFrame,
    n_portfolios: int = N_PORTFOLIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_frames = []
    skipped_groups = []

    for method in vw.METHODS:
        signal_col = METHOD_SPECS[method]["signal_col"]
        pieces = []

        for (year, sector), sub in firm_year.groupby(["FormationYear", SECTOR_COL], sort=True):
            assigned = assign_quantile_portfolios(
                sub=sub.copy(),
                signal_col=signal_col,
                n_portfolios=n_portfolios,
            )
            assigned["Method"] = method
            assigned["SignalUsed"] = signal_col
            pieces.append(assigned)

            n_valid = int(sub[signal_col].notna().sum())
            if n_valid < n_portfolios:
                skipped_groups.append(
                    {
                        "FormationYear": year,
                        SECTOR_COL: sector,
                        "Method": method,
                        "SignalUsed": signal_col,
                        "n_valid": n_valid,
                        "reason": f"fewer than {n_portfolios} valid firms",
                    }
                )

        method_frames.append(pd.concat(pieces, ignore_index=True))

    long_df = pd.concat(method_frames, ignore_index=True)

    keep_cols = [
        "Ticker",
        "FormationYear",
        SECTOR_COL,
        "theta_obs",
        "theta_post_mean",
        "p_q5",
        "p_median",
        "sigma_acc",
        "MarketCap",
        "Method",
        "SignalUsed",
        "PortfolioNum",
        "Portfolio",
    ]
    optional_cols = [c for c in ["CompanyName", "Industry"] if c in long_df.columns]
    keep_cols = keep_cols[:2] + optional_cols + keep_cols[2:]

    long_df = long_df[keep_cols].copy()
    long_df["PortfolioNum"] = long_df["PortfolioNum"].astype("Int64")
    long_df = long_df.sort_values(
        ["FormationYear", SECTOR_COL, "Method", "PortfolioNum", "Ticker"],
        na_position="last",
    ).reset_index(drop=True)

    skipped = pd.DataFrame(skipped_groups).drop_duplicates().reset_index(drop=True)
    return long_df, skipped


def build_sector_neutral_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    valid = assignments[assignments["PortfolioNum"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    return (
        valid.groupby(["FormationYear", SECTOR_COL, "Method", "PortfolioNum", "Portfolio"], as_index=False)
        .agg(
            n_firms=("Ticker", "nunique"),
            total_marketcap=("MarketCap", "sum"),
            avg_theta_obs=("theta_obs", "mean"),
            avg_theta_post_mean=("theta_post_mean", "mean"),
            avg_p_q5=("p_q5", "mean"),
            avg_sigma_acc=("sigma_acc", "mean"),
        )
        .sort_values(["FormationYear", SECTOR_COL, "Method", "PortfolioNum"])
        .reset_index(drop=True)
    )


def aggregate_monthly_returns(monthly_holdings: pd.DataFrame) -> pd.DataFrame:
    required = ["Method", "Portfolio", "Date", "WeightedReturn"]
    missing = [c for c in required if c not in monthly_holdings.columns]
    if missing:
        raise ValueError(f"Sector-neutral monthly holdings missing required columns: {missing}")

    monthly = (
        monthly_holdings.groupby(["Method", "Portfolio", "Date"], as_index=False)["WeightedReturn"]
        .sum()
        .rename(columns={"WeightedReturn": "Return"})
        .sort_values(["Method", "Portfolio", "Date"])
        .reset_index(drop=True)
    )
    return monthly


def save_auxiliary_outputs(
    output_dir: Path,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    skipped_groups: pd.DataFrame,
    monthly_holdings: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "sector_neutral_portfolio_assignments_long": output_dir / "sector_neutral_portfolio_assignments_long.csv",
        "sector_neutral_portfolio_formation_summary": output_dir / "sector_neutral_portfolio_formation_summary.csv",
        "sector_neutral_skipped_sector_years": output_dir / "sector_neutral_skipped_sector_years.csv",
        "sector_neutral_monthly_holdings": output_dir / "sector_neutral_monthly_holdings.csv",
    }

    assignments.to_csv(outputs["sector_neutral_portfolio_assignments_long"], index=False)
    summary.to_csv(outputs["sector_neutral_portfolio_formation_summary"], index=False)
    skipped_groups.to_csv(outputs["sector_neutral_skipped_sector_years"], index=False)
    monthly_holdings.to_csv(outputs["sector_neutral_monthly_holdings"], index=False)

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
    print(f"  sector-neutral firm-year source: {latent_source}")
    print(f"  monthly stock prices: {stock_prices_csv}")
    print(f"  monthly market caps: {market_cap_csv}")
    print(f"  monthly factor returns: {factors_csv}")
    print("  sector quantile helper: modelling/shared/portfolio_formation.py::assign_quantile_portfolios")
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

    latent_source = choose_latent_source(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        requested_source=resolve_cli_path(args.latent_source, project_root),
    )
    stock_prices_csv = choose_run_config_path(
        project_root=project_root,
        run_dir=run_dir,
        requested_path=resolve_cli_path(args.stock_prices_csv, project_root),
        config_key="returns_csv",
        fallback="data/processed_data_lseg/all_stock_prices_nok.csv",
    )
    market_cap_csv = choose_run_config_path(
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
        output_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_sector_neutral"

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

    firm_year = load_latent_firm_year(latent_source)
    assignments, skipped_groups = form_sector_neutral_assignments(firm_year, n_portfolios=N_PORTFOLIOS)
    summary = build_sector_neutral_summary(assignments)

    factors = load_factor_data(factors_csv)
    prepared = build_monthly_portfolio_returns(
        assignments=assignments,
        stock_prices_csv=stock_prices_csv,
        market_cap_csv=market_cap_csv,
        factors=factors,
        n_portfolios=N_PORTFOLIOS,
    )
    monthly_holdings = prepared["monthly_holdings"]
    monthly_returns = aggregate_monthly_returns(monthly_holdings)

    rf = factors["RF"].copy()
    zero_rf = pd.Series(0.0, index=rf.index, name="RF")

    q5_returns, ls_returns, monthly_used = vw.build_strategy_returns(monthly_returns, factors)

    ls_levels = vw.run_level_regressions(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
        nw_lags=args.nw_lags,
    )
    q5_levels = vw.run_level_regressions(
        strategy_returns=q5_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q5",
        nw_lags=args.nw_lags,
    )
    ls_diffs = vw.run_alpha_difference_tests(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
        nw_lags=args.nw_lags,
    )
    q5_diffs = vw.run_alpha_difference_tests(
        strategy_returns=q5_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q5",
        nw_lags=args.nw_lags,
    )

    vw.assert_expected_shapes(ls_levels, ls_diffs, q5_levels, q5_diffs)
    preview = vw.build_preview(
        levels=pd.concat([ls_levels, q5_levels], ignore_index=True),
        differences=pd.concat([ls_diffs, q5_diffs], ignore_index=True),
    )

    table_outputs = vw.save_outputs(
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
        monthly_holdings=monthly_holdings,
    )
    plot_outputs = vw.save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    print("\nSector-neutral assignment summary")
    print(f"  firm-year rows loaded: {len(firm_year)}")
    print(f"  assignment rows created: {len(assignments)}")
    print(f"  valid assignment rows: {int(assignments['PortfolioNum'].notna().sum())}")
    print(f"  skipped sector-year-method groups: {len(skipped_groups)}")

    print("\nCreated CSV files")
    row_counts = {
        "table_ls_alpha_levels": len(ls_levels),
        "table_ls_alpha_differences": len(ls_diffs),
        "table_q5_alpha_levels": len(q5_levels),
        "table_q5_alpha_differences": len(q5_diffs),
        "monthly_portfolio_returns_used": len(monthly_used),
        "risk_adjusted_table_preview": len(preview),
    }
    for key, path in table_outputs.items():
        print(f"  {path} ({row_counts[key]} rows)")

    print("\nCreated audit CSV files")
    audit_counts = {
        "sector_neutral_portfolio_assignments_long": len(assignments),
        "sector_neutral_portfolio_formation_summary": len(summary),
        "sector_neutral_skipped_sector_years": len(skipped_groups),
        "sector_neutral_monthly_holdings": len(monthly_holdings),
    }
    for key, path in auxiliary_outputs.items():
        print(f"  {path} ({audit_counts[key]} rows)")

    print("\nCreated plot files")
    for path in plot_outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
