from __future__ import annotations

import argparse
import os
import shutil
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

from helper_functions import find_project_root, load_factor_data, resolve_path
from latent_prof_model import DEFAULT_GAMMA, run_latent_prof_model
from portfolio_evaluation import run_portfolio_evaluation
from portfolio_formation import ALL_METHODS, run_portfolio_formation

import generate_capped_weight_risk_adjusted_table_data as capped
import generate_risk_adjusted_table_data as vw


ANALYST_LABEL = "Analyst CFO hybrid HB"
BASELINE_LABEL = "Baseline no-lead HB matched sample"

ANALYST_KEY = "analyst_cfo_hybrid_point"
BASELINE_KEY = "no_cfo_lead_matched_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched analyst-CFO vs no-CFO-lead point-estimate portfolio "
            "comparison outputs from a run directory produced with "
            "hb_run_model_specification='both'."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/cur_res_analyst_cfo"),
        help="Run directory containing uncertainty_model/hb_results_* outputs.",
    )
    parser.add_argument(
        "--analyst-table-dir",
        type=Path,
        default=None,
        help="Optional precomputed analyst-CFO UCITS thesis table directory.",
    )
    parser.add_argument(
        "--n-sigma-draws",
        type=int,
        default=None,
        help="Number of HB sigma draws for latent full propagation. Default: all available.",
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--nw-lags", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="Regenerate existing downstream files.")
    return parser.parse_args()


def required_table_files(table_dir: Path) -> list[Path]:
    names = [
        "table_ls_alpha_levels.csv",
        "table_q5_alpha_levels.csv",
        "table_ls_alpha_differences.csv",
        "table_q5_alpha_differences.csv",
        "table_raw_performance.csv",
        "monthly_portfolio_returns_used.csv",
        "risk_adjusted_table_preview.csv",
    ]
    return [table_dir / name for name in names]


def ensure_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_cli_path(path: Path | None, project_root: Path) -> Path | None:
    if path is None:
        return None
    return resolve_path(path, project_root)


def generate_ucits_tables(
    portfolio_eval_dir: Path,
    factors_csv: Path,
    output_dir: Path,
    nw_lags: int,
) -> None:
    holdings = portfolio_eval_dir / "monthly_holdings.csv"
    ensure_exists(holdings, "monthly holdings")

    monthly_returns, ucits_holdings, diagnostics = capped.load_ucits_weight_monthly_returns(
        path=holdings,
        single_issuer_cap=capped.UCITS_SINGLE_ISSUER_CAP,
        large_position_threshold=capped.UCITS_LARGE_POSITION_THRESHOLD,
        large_position_aggregate_cap=capped.UCITS_LARGE_POSITION_AGGREGATE_CAP,
    )
    factors = load_factor_data(factors_csv)
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

    levels = pd.concat([ls_levels, q5_levels], ignore_index=True)
    diffs = pd.concat([ls_diffs, q5_diffs], ignore_index=True)
    preview = vw.build_preview(levels=levels, differences=diffs)

    vw.save_outputs(
        output_dir=output_dir,
        ls_levels=ls_levels,
        ls_diffs=ls_diffs,
        q5_levels=q5_levels,
        q5_diffs=q5_diffs,
        monthly_used=monthly_used,
        preview=preview,
        rf=rf,
        grs_tests=pd.concat([ls_grs, q5_grs], ignore_index=True),
        grs_alpha_components=pd.concat([ls_grs_alpha, q5_grs_alpha], ignore_index=True),
    )
    capped.save_ucits_audit_outputs(
        output_dir=output_dir,
        ucits_holdings=ucits_holdings,
        diagnostics=diagnostics,
    )
    vw.save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)


def run_downstream_branch(
    run_dir: Path,
    spec_key: str,
    label: str,
    uncertainty_subdir: str,
    factors_csv: Path,
    force: bool,
    gamma: float,
    nw_lags: int,
) -> Path:
    branch_dir = run_dir / "portfolio_return_comparison" / spec_key
    latent_dir = branch_dir / "latent_prof_model"
    formation_dir = branch_dir / "portfolio_formation"
    evaluation_dir = branch_dir / "portfolio_evaluation"
    table_dir = evaluation_dir / "thesis_risk_adjusted_tables_ucits_5_10_40"

    uncertainty_csv = (
        run_dir
        / "uncertainty_model"
        / uncertainty_subdir
        / "uncertainty_firm_year.csv"
    )
    ensure_exists(uncertainty_csv, f"{label} uncertainty firm-year CSV")

    latent_csv = latent_dir / "latent_prof_firm_year.csv"
    if force or not latent_csv.exists():
        print(f"\nRunning latent quality for {label}")
        run_latent_prof_model(
            input_csv=uncertainty_csv,
            output_dir=latent_dir,
            uncertainty_method="HB",
            gamma=gamma,
            use_full_propagation=False,
            hb_full_posterior_parquet=None,
            n_sigma_draws=None,
            checkpoint_every_draws=50,
        )

    assignments_csv = formation_dir / "portfolio_assignments_long.csv"
    if force or not assignments_csv.exists():
        print(f"\nRunning portfolio formation for {label}")
        run_portfolio_formation(
            input_csv=latent_csv,
            output_dir=formation_dir,
            n_portfolios=5,
        )

    monthly_holdings = evaluation_dir / "monthly_holdings.csv"
    if force or not monthly_holdings.exists():
        print(f"\nRunning portfolio evaluation for {label}")
        run_portfolio_evaluation(
            assignments_csv=assignments_csv,
            stock_prices_csv=PROJECT_ROOT / "data/processed_data_lseg/all_stock_prices_nok.csv",
            dividends_csv=PROJECT_ROOT / "data/processed_data_lseg/dividends_monthly_nok.csv",
            market_cap_csv=PROJECT_ROOT / "data/processed_data_lseg/historical_market_cap_nok.csv",
            factors_csv=factors_csv,
            output_dir=evaluation_dir,
            n_portfolios=5,
            nw_lags=nw_lags,
        )

    if force or any(not path.exists() for path in required_table_files(table_dir)):
        print(f"\nGenerating UCITS thesis tables for {label}")
        generate_ucits_tables(
            portfolio_eval_dir=evaluation_dir,
            factors_csv=factors_csv,
            output_dir=table_dir,
            nw_lags=nw_lags,
        )

    return table_dir


def read_levels(table_dir: Path, spec_key: str, label: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(table_dir / "table_ls_alpha_levels.csv"),
        pd.read_csv(table_dir / "table_q5_alpha_levels.csv"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "Specification", label)
    out.insert(0, "SpecificationKey", spec_key)
    return out


def read_differences(table_dir: Path, spec_key: str, label: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(table_dir / "table_ls_alpha_differences.csv"),
        pd.read_csv(table_dir / "table_q5_alpha_differences.csv"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "Specification", label)
    out.insert(0, "SpecificationKey", spec_key)
    return out


def read_with_spec(path: Path, spec_key: str, label: str) -> pd.DataFrame:
    out = pd.read_csv(path)
    out.insert(0, "Specification", label)
    out.insert(0, "SpecificationKey", spec_key)
    return out


def build_ff5_mom_comparison(levels: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    ff5 = levels.loc[levels["FactorModel"].eq("FF5+MOM")].copy()
    raw_keep = [
        "SpecificationKey",
        "PortfolioStrategy",
        "Method",
        "MethodLabel",
        "annualized_return",
        "annualized_excess_return",
        "volatility_ann",
        "sharpe_ratio",
        "max_drawdown",
        "n_obs",
    ]
    merged = ff5.merge(
        raw[raw_keep],
        on=["SpecificationKey", "PortfolioStrategy", "Method"],
        how="left",
        validate="1:1",
    )
    return merged


def build_delta_table(comparison: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "annualized_return",
        "annualized_excess_return",
        "volatility_ann",
        "sharpe_ratio",
        "max_drawdown",
        "alpha_annualized",
        "t_stat",
        "p_value",
        "r_squared",
        "n_obs_x",
    ]
    rows = []
    for keys, sub in comparison.groupby(["PortfolioStrategy", "Method"], sort=True):
        indexed = sub.set_index("SpecificationKey")
        if ANALYST_KEY not in indexed.index or BASELINE_KEY not in indexed.index:
            continue
        row = {"PortfolioStrategy": keys[0], "Method": keys[1]}
        analyst = indexed.loc[ANALYST_KEY]
        baseline = indexed.loc[BASELINE_KEY]
        for col in value_cols:
            if col not in indexed.columns:
                continue
            row[f"{col}_{ANALYST_LABEL}"] = analyst[col]
            row[f"{col}_{BASELINE_LABEL}"] = baseline[col]
        row["delta_annualized_return_analyst_minus_baseline"] = (
            analyst["annualized_return"] - baseline["annualized_return"]
        )
        row["delta_annualized_excess_return_analyst_minus_baseline"] = (
            analyst["annualized_excess_return"] - baseline["annualized_excess_return"]
        )
        row["delta_volatility_ann_analyst_minus_baseline"] = (
            analyst["volatility_ann"] - baseline["volatility_ann"]
        )
        row["delta_sharpe_ratio_analyst_minus_baseline"] = (
            analyst["sharpe_ratio"] - baseline["sharpe_ratio"]
        )
        row["delta_max_drawdown_analyst_minus_baseline"] = (
            analyst["max_drawdown"] - baseline["max_drawdown"]
        )
        row["delta_alpha_annualized_analyst_minus_baseline"] = (
            analyst["alpha_annualized"] - baseline["alpha_annualized"]
        )
        row["delta_t_stat_analyst_minus_baseline"] = analyst["t_stat"] - baseline["t_stat"]
        row["delta_p_value_analyst_minus_baseline"] = analyst["p_value"] - baseline["p_value"]
        row["delta_r_squared_analyst_minus_baseline"] = (
            analyst["r_squared"] - baseline["r_squared"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_assignment_changes(run_dir: Path, baseline_branch_dir: Path) -> pd.DataFrame:
    analyst_wide = run_dir / "portfolio_formation" / "portfolio_assignments_wide.csv"
    baseline_wide = baseline_branch_dir / "portfolio_formation" / "portfolio_assignments_wide.csv"
    if not analyst_wide.exists() or not baseline_wide.exists():
        return pd.DataFrame()

    a = pd.read_csv(analyst_wide)
    b = pd.read_csv(baseline_wide)
    merged = a.merge(
        b,
        on=["Ticker", "FormationYear"],
        how="inner",
        suffixes=("_analyst", "_baseline"),
        validate="1:1",
    )
    rows = []
    for method in ALL_METHODS:
        col = f"{method}_Portfolio"
        a_col = f"{col}_analyst"
        b_col = f"{col}_baseline"
        if a_col not in merged.columns or b_col not in merged.columns:
            continue
        changed = merged[a_col].fillna("<NA>").ne(merged[b_col].fillna("<NA>"))
        rows.append(
            {
                "Method": method,
                "n_common_firm_years": int(len(merged)),
                "n_reassigned_firm_years": int(changed.sum()),
                "share_reassigned_firm_years": float(changed.mean()) if len(merged) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_extract(
    run_dir: Path,
    analyst_table_dir: Path,
    baseline_table_dir: Path,
    extract_dir: Path,
) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (ANALYST_KEY, ANALYST_LABEL, analyst_table_dir),
        (BASELINE_KEY, BASELINE_LABEL, baseline_table_dir),
    ]

    levels = pd.concat([read_levels(td, key, label) for key, label, td in specs], ignore_index=True)
    differences = pd.concat(
        [read_differences(td, key, label) for key, label, td in specs],
        ignore_index=True,
    )
    raw = pd.concat(
        [read_with_spec(td / "table_raw_performance.csv", key, label) for key, label, td in specs],
        ignore_index=True,
    )
    preview = pd.concat(
        [
            read_with_spec(td / "risk_adjusted_table_preview.csv", key, label)
            for key, label, td in specs
        ],
        ignore_index=True,
    )
    monthly = pd.concat(
        [
            read_with_spec(td / "monthly_portfolio_returns_used.csv", key, label)
            for key, label, td in specs
        ],
        ignore_index=True,
    )

    levels.to_csv(extract_dir / "alpha_levels_all_models_both_specs.csv", index=False)
    differences.to_csv(extract_dir / "alpha_differences_all_models_both_specs.csv", index=False)
    raw.to_csv(extract_dir / "raw_performance_both_specs.csv", index=False)
    preview.to_csv(extract_dir / "risk_adjusted_preview_all_models_both_specs.csv", index=False)
    monthly.to_csv(extract_dir / "monthly_portfolio_returns_used_both_specs.csv", index=False)

    grs_frames = []
    for key, label, td in specs:
        grs_path = td / "table_grs_tests.csv"
        if grs_path.exists():
            grs_frames.append(read_with_spec(grs_path, key, label))
    if grs_frames:
        pd.concat(grs_frames, ignore_index=True).to_csv(
            extract_dir / "grs_tests_both_specs.csv",
            index=False,
        )

    comparison = build_ff5_mom_comparison(levels=levels, raw=raw)
    comparison.to_csv(extract_dir / "portfolio_return_comparison_ff5_mom.csv", index=False)

    deltas = build_delta_table(comparison)
    deltas.to_csv(extract_dir / "analyst_minus_baseline_ff5_mom_deltas.csv", index=False)

    headline_cols = [
        "PortfolioStrategy",
        "Method",
        "delta_annualized_return_analyst_minus_baseline",
        "delta_annualized_excess_return_analyst_minus_baseline",
        "delta_alpha_annualized_analyst_minus_baseline",
        "delta_t_stat_analyst_minus_baseline",
        "delta_p_value_analyst_minus_baseline",
    ]
    deltas[[c for c in headline_cols if c in deltas.columns]].to_csv(
        extract_dir / "headline_ff5_mom_deltas.csv",
        index=False,
    )

    assignment_changes = build_assignment_changes(
        run_dir=run_dir,
        baseline_branch_dir=run_dir / "portfolio_return_comparison" / BASELINE_KEY,
    )
    if not assignment_changes.empty:
        assignment_changes.to_csv(extract_dir / "portfolio_assignment_changes.csv", index=False)

    comparison_dir = run_dir / "uncertainty_model" / "hb_results_comparison"
    copy_map = {
        "hb_no_cfo_lead_matched_sample_vs_analyst_cfo_hybrid_by_year.csv": (
            "hb_model_comparison_by_year.csv"
        ),
        "hb_no_cfo_lead_matched_sample_vs_analyst_cfo_hybrid_overall_summary.csv": (
            "hb_model_comparison_overall.csv"
        ),
        "hb_no_cfo_lead_matched_sample_vs_analyst_cfo_hybrid_overlap_firm_years.csv": (
            "hb_model_comparison_overlap_firm_years.csv"
        ),
    }
    for source_name, dest_name in copy_map.items():
        source = comparison_dir / source_name
        if source.exists():
            shutil.copy2(source, extract_dir / dest_name)

    readme = extract_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Analyst CFO vs no-CFO-lead matched baseline portfolio comparison",
                "",
                "This folder compares two point-estimate portfolio pipelines on the matched analyst-CFO sample:",
                "",
                f"1. {ANALYST_LABEL}",
                f"   Source tables: {analyst_table_dir}",
                "   Formation-year CFO_t+1 input uses analyst CFO forecasts; historical training observations can use realized CFO_t+1 once observable.",
                "",
                f"2. {BASELINE_LABEL}",
                f"   Source tables: {baseline_table_dir}",
                "   Baseline HB omits CFO_t+1 and is restricted to the same analyst-forecast matched firm-year window.",
                "",
                "Main files:",
                "- portfolio_return_comparison_ff5_mom.csv: FF5+MOM alpha levels plus raw performance for both specs.",
                "- analyst_minus_baseline_ff5_mom_deltas.csv: analyst-minus-baseline deltas by method and strategy.",
                "- headline_ff5_mom_deltas.csv: compact thesis-facing deltas.",
                "- raw_performance_both_specs.csv: unadjusted Q5 and long-short performance.",
                "- alpha_levels_all_models_both_specs.csv: all alpha-level regressions.",
                "- alpha_differences_all_models_both_specs.csv: all alpha-difference tests.",
                "- risk_adjusted_preview_all_models_both_specs.csv: wide preview from each generator.",
                "- monthly_portfolio_returns_used_both_specs.csv: monthly Q5 and long-short returns used.",
                "- portfolio_assignment_changes.csv: firm-year assignment changes by method.",
                "- hb_model_comparison_by_year.csv and hb_model_comparison_overall.csv: upstream HB expected-accrual comparison diagnostics.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    run_dir = resolve_path(args.run_dir, project_root)
    ensure_exists(run_dir, "run directory")

    factors_csv = project_root / "results/extraction_static/factor_data.csv"
    ensure_exists(factors_csv, "factor data CSV")

    analyst_table_dir = resolve_cli_path(args.analyst_table_dir, project_root)
    if analyst_table_dir is None:
        analyst_table_dir = run_downstream_branch(
            run_dir=run_dir,
            spec_key=ANALYST_KEY,
            label=ANALYST_LABEL,
            uncertainty_subdir="hb_results_analyst_cfo",
            factors_csv=factors_csv,
            force=args.force,
            gamma=args.gamma,
            nw_lags=args.nw_lags,
        )
    for path in required_table_files(analyst_table_dir):
        ensure_exists(path, "analyst-CFO table file")

    baseline_table_dir = run_downstream_branch(
        run_dir=run_dir,
        spec_key=BASELINE_KEY,
        label=BASELINE_LABEL,
        uncertainty_subdir="hb_results_no_cfo_lead_matched_sample",
        factors_csv=factors_csv,
        force=args.force,
        gamma=args.gamma,
        nw_lags=args.nw_lags,
    )

    extract_dir = (
        run_dir
        / "portfolio_return_comparison"
        / "analyst_vs_baseline_extract"
    )
    write_extract(
        run_dir=run_dir,
        analyst_table_dir=analyst_table_dir,
        baseline_table_dir=baseline_table_dir,
        extract_dir=extract_dir,
    )

    print("\nCreated analyst-CFO portfolio comparison extract")
    print(f"  {extract_dir}")
    for path in sorted(extract_dir.glob("*.csv")):
        try:
            n_rows = len(pd.read_csv(path))
            print(f"  {path.name}: {n_rows} rows")
        except Exception:
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
