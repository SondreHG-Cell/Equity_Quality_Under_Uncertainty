from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_capped_weight_risk_adjusted_table_data as capped
import generate_risk_adjusted_table_data as vw
import portfolio_formation as pf
from helper_functions import find_project_root, load_factor_data, resolve_path


DEFAULT_MIN_BE_TO_ASSET_CUTOFFS = (0.10, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate risk-adjusted thesis table data after excluding firm-years "
            "with low Book Equity / Total Assets before portfolio formation."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Results run directory, e.g. results/current_res.",
    )
    parser.add_argument(
        "--latent-source",
        type=Path,
        default=None,
        help=(
            "Optional latent_prof_firm_year.csv source. Defaults to "
            "<run-dir>/latent_prof_model/latent_prof_firm_year.csv."
        ),
    )
    parser.add_argument(
        "--portfolio-source",
        type=Path,
        default=None,
        help=(
            "Optional monthly holdings source used to recover monthly stock returns. "
            "Defaults to <run-dir>/portfolio_evaluation/monthly_holdings.csv."
        ),
    )
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=None,
        help="Optional monthly factor CSV. Defaults to run_config.json or factor_data.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for output folders. Defaults to "
            "<run-dir>/portfolio_evaluation."
        ),
    )
    parser.add_argument(
        "--cutoffs",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MIN_BE_TO_ASSET_CUTOFFS),
        help="Comma-separated minimum BE/AT cutoffs. Default: 0.10,0.05.",
    )
    parser.add_argument(
        "--single-issuer-cap",
        type=float,
        default=capped.UCITS_SINGLE_ISSUER_CAP,
        help="UCITS single-issuer cap. Default is 0.10.",
    )
    parser.add_argument(
        "--large-position-threshold",
        type=float,
        default=capped.UCITS_LARGE_POSITION_THRESHOLD,
        help="UCITS large-position threshold. Default is 0.05.",
    )
    parser.add_argument(
        "--large-position-aggregate-cap",
        type=float,
        default=capped.UCITS_LARGE_POSITION_AGGREGATE_CAP,
        help="UCITS aggregate cap for positions above the threshold. Default is 0.40.",
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


def parse_cutoffs(raw: str) -> list[float]:
    cutoffs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Cutoffs must be non-negative finite decimals. Invalid: {part}")
        cutoffs.append(value)
    if not cutoffs:
        raise ValueError("At least one BE/AT cutoff must be supplied.")
    return cutoffs


def cutoff_label(cutoff: float) -> str:
    return f"{cutoff * 100:g}pct".replace(".", "_")


def choose_latent_source(run_dir: Path, requested_source: Path | None) -> Path:
    default = run_dir / "latent_prof_model" / "latent_prof_firm_year.csv"
    if requested_source is not None:
        if requested_source.exists():
            return requested_source
        raise FileNotFoundError(
            "Requested latent source does not exist.\n"
            f"Requested: {requested_source}\n"
            f"Default checked: {default}"
        )
    if default.exists():
        return default
    raise FileNotFoundError(
        "Could not locate latent profitability firm-year source.\n"
        f"Searched: {default}"
    )


def load_latent_with_be_asset_ratio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    pf.validate_input_columns(df)
    missing = [col for col in ["BE", "AT"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns needed for BE/AT filter: {missing}")

    df = pf.clean_input(df)
    df["BE"] = pd.to_numeric(df["BE"], errors="coerce")
    df["AT"] = pd.to_numeric(df["AT"], errors="coerce")
    df["BookEquityToAssets"] = np.where(df["AT"] > 0, df["BE"] / df["AT"], np.nan)
    return df


def build_filter_summary(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_valid_ratio"] = tmp["BookEquityToAssets"].notna()
    tmp["_kept"] = tmp["BookEquityToAssets"] >= cutoff
    summary = (
        tmp.groupby("FormationYear", as_index=False)
        .agg(
            cutoff=("BookEquityToAssets", lambda _: cutoff),
            total_firm_years=("Ticker", "nunique"),
            valid_be_to_assets=("_valid_ratio", "sum"),
            kept_firm_years=("_kept", "sum"),
            min_be_to_assets=("BookEquityToAssets", "min"),
            median_be_to_assets=("BookEquityToAssets", "median"),
            p10_be_to_assets=("BookEquityToAssets", lambda s: s.quantile(0.10)),
        )
        .sort_values("FormationYear")
        .reset_index(drop=True)
    )
    summary["excluded_missing_or_nonpositive_assets"] = (
        summary["total_firm_years"] - summary["valid_be_to_assets"]
    )
    summary["excluded_below_cutoff"] = summary["valid_be_to_assets"] - summary["kept_firm_years"]
    summary["kept_share"] = summary["kept_firm_years"] / summary["total_firm_years"]
    cols = [
        "FormationYear",
        "cutoff",
        "total_firm_years",
        "valid_be_to_assets",
        "kept_firm_years",
        "excluded_missing_or_nonpositive_assets",
        "excluded_below_cutoff",
        "kept_share",
        "min_be_to_assets",
        "p10_be_to_assets",
        "median_be_to_assets",
    ]
    return summary[cols]


def augment_assignments_with_ratio(assignments: pd.DataFrame, filtered_latent: pd.DataFrame) -> pd.DataFrame:
    ratio_cols = filtered_latent[
        ["Ticker", "FormationYear", "BE", "AT", "BookEquityToAssets"]
    ].drop_duplicates(["Ticker", "FormationYear"])
    return assignments.merge(ratio_cols, on=["Ticker", "FormationYear"], how="left")


def build_filtered_monthly_holdings_source(
    base_holdings_path: Path,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.read_csv(base_holdings_path)
    required = ["Ticker", "Date", "FormationYear", "Return", "LagMarketCap"]
    missing = [col for col in required if col not in base.columns]
    if missing:
        raise ValueError(f"{base_holdings_path} is missing required stock-return columns: {missing}")

    stock_panel = base[required].copy()
    stock_panel["Ticker"] = stock_panel["Ticker"].astype(str).str.strip()
    stock_panel["FormationYear"] = pd.to_numeric(
        stock_panel["FormationYear"], errors="coerce"
    ).astype("Int64")
    stock_panel["Return"] = pd.to_numeric(stock_panel["Return"], errors="coerce")
    stock_panel["LagMarketCap"] = pd.to_numeric(stock_panel["LagMarketCap"], errors="coerce")
    stock_panel = stock_panel.dropna(
        subset=["Ticker", "Date", "FormationYear", "Return", "LagMarketCap"]
    ).copy()
    stock_panel["FormationYear"] = stock_panel["FormationYear"].astype(int)
    stock_panel = stock_panel.drop_duplicates(["Ticker", "Date"])

    keep_cols = [
        "Ticker",
        "FormationYear",
        "Method",
        "SignalUsed",
        "PortfolioNum",
        "Portfolio",
        "MethodLabel",
        "BE",
        "AT",
        "BookEquityToAssets",
    ]
    assignments = assignments.loc[assignments["PortfolioNum"].notna(), keep_cols].copy()
    assignments["Ticker"] = assignments["Ticker"].astype(str).str.strip()
    assignments["FormationYear"] = assignments["FormationYear"].astype(int)

    monthly = stock_panel.merge(
        assignments,
        on=["Ticker", "FormationYear"],
        how="inner",
        validate="many_to_many",
    )
    monthly = monthly[
        monthly["Return"].notna()
        & monthly["LagMarketCap"].notna()
        & (monthly["LagMarketCap"] > 0)
    ].copy()
    return monthly.sort_values(["Date", "Method", "Portfolio", "Ticker"]).reset_index(drop=True)


def save_assignment_outputs(
    output_dir: Path,
    filtered_latent: pd.DataFrame,
    long_assignments: pd.DataFrame,
    wide_assignments: pd.DataFrame,
    formation_summary: pd.DataFrame,
    filter_summary: pd.DataFrame,
    monthly_source: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "filtered_latent_prof_firm_year": output_dir / "filtered_latent_prof_firm_year.csv",
        "portfolio_assignments_long_filtered": output_dir / "portfolio_assignments_long_filtered.csv",
        "portfolio_assignments_wide_filtered": output_dir / "portfolio_assignments_wide_filtered.csv",
        "portfolio_formation_summary_filtered": output_dir / "portfolio_formation_summary_filtered.csv",
        "book_equity_assets_filter_summary": output_dir / "book_equity_assets_filter_summary.csv",
        "filtered_monthly_holdings_source": output_dir / "filtered_monthly_holdings_source.csv",
    }
    filtered_latent.to_csv(outputs["filtered_latent_prof_firm_year"], index=False)
    long_assignments.to_csv(outputs["portfolio_assignments_long_filtered"], index=False)
    wide_assignments.to_csv(outputs["portfolio_assignments_wide_filtered"], index=False)
    formation_summary.to_csv(outputs["portfolio_formation_summary_filtered"], index=False)
    filter_summary.to_csv(outputs["book_equity_assets_filter_summary"], index=False)
    monthly_source.to_csv(outputs["filtered_monthly_holdings_source"], index=False)
    return outputs


def print_identification(
    run_dir: Path,
    latent_source: Path,
    base_holdings_source: Path,
    factors_csv: Path,
    output_root: Path,
    cutoffs: list[float],
    nw_lags: int,
) -> None:
    print("\nIdentified inputs and reused helpers")
    print(f"  run_dir: {run_dir}")
    print(f"  latent firm-year source: {latent_source}")
    print(f"  monthly stock-return source: {base_holdings_source}")
    print(f"  monthly factor returns: {factors_csv}")
    print(f"  BE/AT cutoffs: {', '.join(f'{c:.2%}' for c in cutoffs)}")
    print("  portfolio formation helper: modelling/shared/portfolio_formation.py::build_long_output")
    print("  weighting helper: modelling/shared/generate_capped_weight_risk_adjusted_table_data.py::load_ucits_weight_monthly_returns")
    print("  risk-adjusted regression helper: modelling/shared/step5_evaluation.py::risk_adjusted_performance")
    print("  alpha-difference helper: modelling/shared/step5_evaluation.py::alpha_differences")
    print(f"  HAC lags: {nw_lags}")
    print(f"  output_root: {output_root}")


def run_cutoff(
    cutoff: float,
    latent: pd.DataFrame,
    base_holdings_source: Path,
    factors: pd.DataFrame,
    output_root: Path,
    nw_lags: int,
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
) -> dict[str, Path]:
    label = cutoff_label(cutoff)
    output_dir = output_root / f"thesis_risk_adjusted_tables_be_at_min_{label}_ucits_5_10_40"
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_summary = build_filter_summary(latent, cutoff=cutoff)
    filtered_latent = latent.loc[latent["BookEquityToAssets"] >= cutoff].copy()
    if filtered_latent.empty:
        raise ValueError(f"BE/AT cutoff {cutoff:.2%} leaves no firm-years.")

    assignments_long = pf.build_long_output(filtered_latent, n_portfolios=5)
    assignments_long = augment_assignments_with_ratio(assignments_long, filtered_latent)
    assignments_wide = pf.build_wide_output(assignments_long)
    formation_summary = pf.build_summary_output(assignments_long)

    monthly_source = build_filtered_monthly_holdings_source(
        base_holdings_path=base_holdings_source,
        assignments=assignments_long,
    )
    if monthly_source.empty:
        raise ValueError(f"BE/AT cutoff {cutoff:.2%} produced no monthly holdings.")

    assignment_outputs = save_assignment_outputs(
        output_dir=output_dir,
        filtered_latent=filtered_latent,
        long_assignments=assignments_long,
        wide_assignments=assignments_wide,
        formation_summary=formation_summary,
        filter_summary=filter_summary,
        monthly_source=monthly_source,
    )

    monthly_returns, ucits_holdings, diagnostics = capped.load_ucits_weight_monthly_returns(
        path=assignment_outputs["filtered_monthly_holdings_source"],
        single_issuer_cap=single_issuer_cap,
        large_position_threshold=large_position_threshold,
        large_position_aggregate_cap=large_position_aggregate_cap,
    )

    rf = factors["RF"].copy()
    zero_rf = pd.Series(0.0, index=rf.index, name="RF")
    q5_returns, ls_returns, monthly_used = vw.build_strategy_returns(monthly_returns, factors)

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
    ls_grs, ls_grs_alpha = vw.run_grs_tests(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
    )
    q5_grs, q5_grs_alpha = vw.run_grs_tests(
        strategy_returns=q5_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q5",
    )

    grs_tests = pd.concat([ls_grs, q5_grs], ignore_index=True)
    grs_alpha_components = pd.concat([ls_grs_alpha, q5_grs_alpha], ignore_index=True)
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
        rf=rf,
        grs_tests=grs_tests,
        grs_alpha_components=grs_alpha_components,
    )
    audit_outputs = capped.save_ucits_audit_outputs(
        output_dir=output_dir,
        ucits_holdings=ucits_holdings,
        diagnostics=diagnostics,
    )
    plot_outputs = vw.save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    print(f"\nBE/AT cutoff {cutoff:.2%}")
    print(f"  output_dir: {output_dir}")
    print(f"  firm-years before filter: {len(latent)}")
    print(f"  firm-years kept: {len(filtered_latent)}")
    print(f"  firm-years excluded: {len(latent) - len(filtered_latent)}")
    print(f"  monthly holdings source rows: {len(monthly_source)}")
    print(f"  UCITS holdings rows: {len(ucits_holdings)}")
    print(f"  monthly portfolio groups: {len(diagnostics)}")
    print(f"  groups where aggregate cap was relaxed: {int(diagnostics['aggregate_cap_relaxed'].sum())}")

    outputs = {}
    outputs.update(assignment_outputs)
    outputs.update(table_outputs)
    outputs.update(audit_outputs)
    outputs.update(plot_outputs)
    return outputs


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    capped.validate_ucits_parameters(
        single_issuer_cap=args.single_issuer_cap,
        large_position_threshold=args.large_position_threshold,
        large_position_aggregate_cap=args.large_position_aggregate_cap,
    )

    cutoffs = parse_cutoffs(args.cutoffs)
    run_dir = vw.choose_run_dir(project_root, resolve_cli_path(args.run_dir, project_root))
    latent_source = choose_latent_source(
        run_dir=run_dir,
        requested_source=resolve_cli_path(args.latent_source, project_root),
    )
    base_holdings_source, portfolio_eval_dir = capped.choose_constituent_source(
        run_dir=run_dir,
        requested_source=resolve_cli_path(args.portfolio_source, project_root),
    )
    factors_csv = vw.choose_factor_csv(
        project_root=project_root,
        run_dir=run_dir,
        requested_factors=resolve_cli_path(args.factors_csv, project_root),
    )
    output_root = resolve_cli_path(args.output_root, project_root)
    if output_root is None:
        output_root = portfolio_eval_dir

    print_identification(
        run_dir=run_dir,
        latent_source=latent_source,
        base_holdings_source=base_holdings_source,
        factors_csv=factors_csv,
        output_root=output_root,
        cutoffs=cutoffs,
        nw_lags=args.nw_lags,
    )

    latent = load_latent_with_be_asset_ratio(latent_source)
    factors = load_factor_data(factors_csv)

    all_outputs: dict[str, dict[str, Path]] = {}
    for cutoff in cutoffs:
        all_outputs[cutoff_label(cutoff)] = run_cutoff(
            cutoff=cutoff,
            latent=latent,
            base_holdings_source=base_holdings_source,
            factors=factors,
            output_root=output_root,
            nw_lags=args.nw_lags,
            single_issuer_cap=args.single_issuer_cap,
            large_position_threshold=args.large_position_threshold,
            large_position_aggregate_cap=args.large_position_aggregate_cap,
        )

    print("\nCreated robustness output folders")
    for label, outputs in all_outputs.items():
        output_dir = outputs["risk_adjusted_table_preview"].parent
        preview_rows = len(pd.read_csv(outputs["risk_adjusted_table_preview"]))
        raw_rows = len(pd.read_csv(outputs["table_raw_performance"]))
        print(f"  {label}: {output_dir}")
        print(f"    risk_adjusted_table_preview.csv ({preview_rows} rows)")
        print(f"    table_raw_performance.csv ({raw_rows} rows)")
        print(f"    book_equity_assets_filter_summary.csv")


if __name__ == "__main__":
    main()
