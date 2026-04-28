from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_risk_adjusted_table_data as vw
from helper_functions import find_project_root, load_factor_data, parse_month_series, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data for equal-weighted risk-adjusted performance "
            "using monthly constituent holdings and existing HAC regression helpers."
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
        help="Optional constituent CSV. Must contain Method, Portfolio, Date, and Return.",
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
            "<portfolio_evaluation_dir>/thesis_risk_adjusted_tables_equal_weighted."
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


def choose_equal_weight_source(
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
                "Requested equal-weighting portfolio source does not exist.\n"
                f"Requested: {requested_source}\n"
                f"Default checked: {default_holdings}"
            )
        return requested_source, requested_source.parent

    if default_holdings.exists():
        return default_holdings, portfolio_eval_dir

    raise FileNotFoundError(
        "Equal-weighted returns require constituent-level monthly holdings.\n"
        "Could not locate monthly_holdings.csv at:\n"
        f"{default_holdings}\n"
        "The wide monthly_portfolio_returns.csv file is already aggregated and cannot be reweighted equally."
    )


def load_equal_weighted_monthly_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["Method", "Portfolio", "Date", "Return"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required constituent-return columns for equal weighting: {missing}"
        )

    out = df[required].copy()
    out["Date"] = parse_month_series(out["Date"])
    out["Method"] = out["Method"].astype(str).str.strip()
    out["Portfolio"] = out["Portfolio"].astype(str).str.strip()
    out["Return"] = pd.to_numeric(out["Return"], errors="coerce")
    out = out.dropna(subset=["Date", "Method", "Portfolio", "Return"])

    monthly = (
        out.groupby(["Method", "Portfolio", "Date"], as_index=False)["Return"]
        .mean()
        .sort_values(["Method", "Portfolio", "Date"])
        .reset_index(drop=True)
    )
    return monthly


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
    print(f"  equal-weighted constituent source: {portfolio_source}")
    print("  portfolio aggregation: simple mean of constituent Return by Method x Portfolio x Date")
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
    portfolio_source, portfolio_eval_dir = choose_equal_weight_source(
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
        output_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables_equal_weighted"

    print_identification(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        portfolio_source=portfolio_source,
        factors_csv=factors_csv,
        output_dir=output_dir,
        nw_lags=args.nw_lags,
    )

    monthly_returns = load_equal_weighted_monthly_returns(portfolio_source)
    factors = load_factor_data(factors_csv)
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

    outputs = vw.save_outputs(
        output_dir=output_dir,
        ls_levels=ls_levels,
        ls_diffs=ls_diffs,
        q5_levels=q5_levels,
        q5_diffs=q5_diffs,
        monthly_used=monthly_used,
        preview=preview,
    )
    plot_outputs = vw.save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    print("\nCreated CSV files")
    row_counts = {
        "table_ls_alpha_levels": len(ls_levels),
        "table_ls_alpha_differences": len(ls_diffs),
        "table_q5_alpha_levels": len(q5_levels),
        "table_q5_alpha_differences": len(q5_diffs),
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
