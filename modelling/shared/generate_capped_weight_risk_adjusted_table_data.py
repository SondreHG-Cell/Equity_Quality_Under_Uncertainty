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

import generate_risk_adjusted_table_data as vw
from helper_functions import find_project_root, load_factor_data, parse_month_series, resolve_path


# UCITS-style 5/10/40 rule parameters. Adjust these at the top of the file if needed.
UCITS_SINGLE_ISSUER_CAP = 0.10
UCITS_LARGE_POSITION_THRESHOLD = 0.05
UCITS_LARGE_POSITION_AGGREGATE_CAP = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data for risk-adjusted performance using "
            "UCITS-style 5/10/40 value weights."
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
            "<portfolio_evaluation_dir>/thesis_risk_adjusted_tables_ucits_5_10_40."
        ),
    )
    parser.add_argument(
        "--single-issuer-cap",
        type=float,
        default=None,
        help=(
            "Optional override for UCITS_SINGLE_ISSUER_CAP. "
            f"Default is {UCITS_SINGLE_ISSUER_CAP}."
        ),
    )
    parser.add_argument(
        "--large-position-threshold",
        type=float,
        default=None,
        help=(
            "Optional override for UCITS_LARGE_POSITION_THRESHOLD. "
            f"Default is {UCITS_LARGE_POSITION_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--large-position-aggregate-cap",
        type=float,
        default=None,
        help=(
            "Optional override for UCITS_LARGE_POSITION_AGGREGATE_CAP. "
            f"Default is {UCITS_LARGE_POSITION_AGGREGATE_CAP}."
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
                "Requested UCITS-weight portfolio source does not exist.\n"
                f"Requested: {requested_source}\n"
                f"Default checked: {default_holdings}"
            )
        return requested_source, requested_source.parent

    if default_holdings.exists():
        return default_holdings, portfolio_eval_dir

    raise FileNotFoundError(
        "UCITS-weight returns require constituent-level monthly holdings.\n"
        "Could not locate monthly_holdings.csv at:\n"
        f"{default_holdings}\n"
        "The wide monthly_portfolio_returns.csv file is already aggregated and cannot be reweighted."
    )


def validate_ucits_parameters(
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
) -> None:
    params = {
        "single_issuer_cap": single_issuer_cap,
        "large_position_threshold": large_position_threshold,
        "large_position_aggregate_cap": large_position_aggregate_cap,
    }
    bad = [name for name, value in params.items() if not np.isfinite(value) or value <= 0 or value > 1]
    if bad:
        raise ValueError(f"UCITS parameters must be in (0, 1]. Invalid: {bad}")
    if single_issuer_cap < large_position_threshold:
        raise ValueError("single_issuer_cap must be >= large_position_threshold.")
    if large_position_aggregate_cap < large_position_threshold:
        raise ValueError("large_position_aggregate_cap must be >= large_position_threshold.")


def ucits_label(
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
) -> str:
    return (
        f"ucits_{single_issuer_cap * 100:g}_"
        f"{large_position_threshold * 100:g}_"
        f"{large_position_aggregate_cap * 100:g}"
    ).replace(".", "_")


def allocate_with_bounds(
    base_weights: np.ndarray,
    total_weight: float,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    base = np.asarray(base_weights, dtype=float)
    n = len(base)
    if n == 0:
        if abs(total_weight) <= 1e-12:
            return base.copy()
        raise ValueError("Cannot allocate positive weight to an empty set.")

    if total_weight < n * lower_bound - 1e-12 or total_weight > n * upper_bound + 1e-12:
        raise ValueError(
            "Infeasible bounded allocation: "
            f"total={total_weight:.8f}, n={n}, lower={lower_bound:.4%}, upper={upper_bound:.4%}."
        )

    alloc = np.full(n, lower_bound, dtype=float)
    caps = np.full(n, upper_bound - lower_bound, dtype=float)
    remaining_weight = total_weight - alloc.sum()

    if remaining_weight <= 1e-12:
        return alloc

    active = np.arange(n)
    base = np.where(np.isfinite(base) & (base > 0), base, 0.0)

    while len(active) > 0 and remaining_weight > 1e-12:
        active_base = base[active]
        if active_base.sum() <= 0:
            proposed = np.full(len(active), remaining_weight / len(active))
        else:
            proposed = active_base / active_base.sum() * remaining_weight

        over_cap = proposed > caps[active] + 1e-12
        if not over_cap.any():
            alloc[active] += proposed
            remaining_weight = 0.0
            break

        capped_active = active[over_cap]
        alloc[capped_active] += caps[capped_active]
        remaining_weight -= caps[capped_active].sum()
        caps[capped_active] = 0.0
        active = active[~over_cap]

    if remaining_weight > 1e-8:
        raise RuntimeError("Bounded allocation failed to distribute all weight.")

    return alloc


def is_ucits_compliant(
    weights: np.ndarray,
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
    require_full_investment: bool = True,
) -> bool:
    weight_sum = weights.sum()
    large_sum = weights[weights > large_position_threshold + 1e-10].sum()
    investment_ok = (
        abs(weight_sum - 1.0) <= 1e-8
        if require_full_investment
        else -1e-10 <= weight_sum <= 1.0 + 1e-8
    )
    return (
        investment_ok
        and weights.min() >= -1e-10
        and weights.max() <= single_issuer_cap + 1e-10
        and large_sum <= large_position_aggregate_cap + 1e-10
    )


def ucits_5_10_40_weights(
    raw_weights: np.ndarray,
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
) -> tuple[np.ndarray, bool]:
    validate_ucits_parameters(
        single_issuer_cap=single_issuer_cap,
        large_position_threshold=large_position_threshold,
        large_position_aggregate_cap=large_position_aggregate_cap,
    )

    raw = np.asarray(raw_weights, dtype=float)
    if len(raw) == 0:
        return raw
    if np.any(raw < 0) or not np.isfinite(raw).all():
        raise ValueError("Raw weights must be finite and non-negative.")

    total = raw.sum()
    if total <= 0:
        raise ValueError("Cannot build UCITS weights from non-positive total raw weight.")

    raw = raw / total
    n = len(raw)

    single_capped = allocate_with_bounds(
        base_weights=raw,
        total_weight=1.0,
        lower_bound=0.0,
        upper_bound=single_issuer_cap,
    )
    if is_ucits_compliant(
        single_capped,
        single_issuer_cap,
        large_position_threshold,
        large_position_aggregate_cap,
        require_full_investment=True,
    ):
        return single_capped, False

    order = np.argsort(-raw, kind="mergesort")
    max_large_names = min(
        n,
        int(np.floor((large_position_aggregate_cap - 1e-12) / large_position_threshold)),
    )
    best_weights = None
    best_objective = np.inf

    for n_large in range(max_large_names + 1):
        large_idx = order[:n_large]
        small_idx = order[n_large:]
        n_small = len(small_idx)

        large_min_total = n_large * large_position_threshold
        large_max_total = min(large_position_aggregate_cap, n_large * single_issuer_cap, 1.0)
        small_max_total = n_small * large_position_threshold
        target_total = 1.0

        lower_large_total = max(large_min_total, target_total - small_max_total)
        upper_large_total = min(large_max_total, target_total)
        if lower_large_total > upper_large_total + 1e-12:
            continue

        preferred_large_total = single_capped[large_idx].sum() if n_large else 0.0
        large_total = float(np.clip(preferred_large_total, lower_large_total, upper_large_total))
        small_total = target_total - large_total

        weights = np.zeros(n, dtype=float)
        if n_large:
            weights[large_idx] = allocate_with_bounds(
                base_weights=raw[large_idx],
                total_weight=large_total,
                lower_bound=large_position_threshold,
                upper_bound=single_issuer_cap,
            )
        if n_small:
            weights[small_idx] = allocate_with_bounds(
                base_weights=raw[small_idx],
                total_weight=small_total,
                lower_bound=0.0,
                upper_bound=large_position_threshold,
            )

        if not is_ucits_compliant(
            weights,
            single_issuer_cap,
            large_position_threshold,
            large_position_aggregate_cap,
            require_full_investment=True,
        ):
            continue

        objective = float(np.sum((weights - single_capped) ** 2))
        if objective < best_objective:
            best_objective = objective
            best_weights = weights

    if best_weights is None:
        return single_capped, True

    return best_weights, False


def apply_ucits_weights(
    group: pd.DataFrame,
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
) -> pd.DataFrame:
    group = group.copy()
    raw_weights = group["LagMarketCap"].to_numpy(dtype=float)
    raw_weights = raw_weights / raw_weights.sum()
    ucits_weights, aggregate_cap_relaxed = ucits_5_10_40_weights(
        raw_weights=raw_weights,
        single_issuer_cap=single_issuer_cap,
        large_position_threshold=large_position_threshold,
        large_position_aggregate_cap=large_position_aggregate_cap,
    )

    group["RawValueWeight"] = raw_weights
    group["UCITSWeight"] = ucits_weights
    group["CashWeight"] = 0.0
    group["UCITSWeightedReturn"] = group["UCITSWeight"] * group["Return"]
    group["AboveLargePositionThreshold"] = group["UCITSWeight"] > large_position_threshold + 1e-10
    group["SingleIssuerCapBinding"] = group["RawValueWeight"] > single_issuer_cap + 1e-12
    group["AggregateCapRelaxed"] = aggregate_cap_relaxed
    return group


def load_ucits_weight_monthly_returns(
    path: Path,
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_ucits_parameters(
        single_issuer_cap=single_issuer_cap,
        large_position_threshold=large_position_threshold,
        large_position_aggregate_cap=large_position_aggregate_cap,
    )
    df = pd.read_csv(path)

    required = ["Ticker", "Method", "Portfolio", "Date", "Return", "LagMarketCap"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required constituent columns for UCITS weights: {missing}")

    out = df[required].copy()
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out["Method"] = out["Method"].astype(str).str.strip()
    out["Portfolio"] = out["Portfolio"].astype(str).str.strip()
    out["Date"] = parse_month_series(out["Date"])
    out["Return"] = pd.to_numeric(out["Return"], errors="coerce")
    out["LagMarketCap"] = pd.to_numeric(out["LagMarketCap"], errors="coerce")
    out = out.dropna(subset=["Ticker", "Method", "Portfolio", "Date", "Return", "LagMarketCap"])
    out = out.loc[out["LagMarketCap"] > 0].copy()

    out = out.loc[out["Method"].isin(vw.METHODS)].copy()
    if out.empty:
        raise ValueError(f"{path} has no rows for required methods: {vw.METHODS}")

    group_cols = ["Method", "Portfolio", "Date"]
    n_firms = out.groupby(group_cols)["Ticker"].nunique()
    impossible = n_firms[n_firms * single_issuer_cap < 1 - 1e-12]
    if not impossible.empty:
        examples = impossible.head(10).to_string()
        raise ValueError(
            "Some portfolios have too few firms for a fully invested 10% single-issuer cap.\n"
            f"Single issuer cap: {single_issuer_cap:.2%}\n"
            f"Examples:\n{examples}"
        )

    ucits_holdings = pd.concat(
        [
            apply_ucits_weights(
                group,
                single_issuer_cap=single_issuer_cap,
                large_position_threshold=large_position_threshold,
                large_position_aggregate_cap=large_position_aggregate_cap,
            )
            for _, group in out.groupby(group_cols, sort=False)
        ],
        ignore_index=True,
    )

    monthly = (
        ucits_holdings.groupby(group_cols, as_index=False)
        .agg(
            Return=("UCITSWeightedReturn", "sum"),
            n_firms=("Ticker", "nunique"),
            max_raw_weight=("RawValueWeight", "max"),
            max_ucits_weight=("UCITSWeight", "max"),
            cash_weight=("CashWeight", "first"),
            large_position_weight=("UCITSWeight", lambda s: s[s > large_position_threshold + 1e-10].sum()),
            n_large_positions=("AboveLargePositionThreshold", "sum"),
            n_names_single_cap_binding=("SingleIssuerCapBinding", "sum"),
            aggregate_cap_relaxed=("AggregateCapRelaxed", "first"),
        )
        .sort_values(["Method", "Portfolio", "Date"])
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
            large_position_weight=("UCITSWeight", lambda s: s[s > large_position_threshold + 1e-10].sum()),
            n_large_positions=("AboveLargePositionThreshold", "sum"),
            n_names_single_cap_binding=("SingleIssuerCapBinding", "sum"),
            aggregate_cap_relaxed=("AggregateCapRelaxed", "first"),
        )
        .sort_values(["Date", "Method", "Portfolio"])
        .reset_index(drop=True)
    )

    return monthly, ucits_holdings, diagnostics


def save_ucits_audit_outputs(
    output_dir: Path,
    ucits_holdings: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "ucits_weight_monthly_holdings": output_dir / "ucits_weight_monthly_holdings.csv",
        "ucits_weight_diagnostics": output_dir / "ucits_weight_diagnostics.csv",
    }
    ucits_holdings.to_csv(outputs["ucits_weight_monthly_holdings"], index=False)
    diagnostics.to_csv(outputs["ucits_weight_diagnostics"], index=False)
    return outputs


def print_identification(
    run_dir: Path,
    portfolio_eval_dir: Path,
    portfolio_source: Path,
    factors_csv: Path,
    output_dir: Path,
    single_issuer_cap: float,
    large_position_threshold: float,
    large_position_aggregate_cap: float,
    nw_lags: int,
) -> None:
    print("\nIdentified inputs and reused helpers")
    print(f"  run_dir: {run_dir}")
    print(f"  portfolio_evaluation_dir: {portfolio_eval_dir}")
    print(f"  constituent holdings source: {portfolio_source}")
    print(f"  monthly factor returns: {factors_csv}")
    print("  weighting rule: UCITS-style 5/10/40")
    print(f"    single issuer cap: {single_issuer_cap:.2%}")
    print(f"    large position threshold: {large_position_threshold:.2%}")
    print(f"    aggregate cap for positions above threshold: {large_position_aggregate_cap:.2%}")
    print("  portfolio return construction: UCITS-adjusted LagMarketCap weights within Date x Method x Portfolio")
    print("  risk-adjusted regression helper: modelling/shared/step5_evaluation.py::risk_adjusted_performance")
    print("  Newey-West/HAC helper: modelling/shared/step5_evaluation.py::_ols_newey_west_full")
    print("  alpha-difference helper: modelling/shared/step5_evaluation.py::alpha_differences")
    print(f"  HAC lags: {nw_lags}")
    print(f"  output_dir: {output_dir}")


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    single_issuer_cap = (
        UCITS_SINGLE_ISSUER_CAP if args.single_issuer_cap is None else args.single_issuer_cap
    )
    large_position_threshold = (
        UCITS_LARGE_POSITION_THRESHOLD
        if args.large_position_threshold is None
        else args.large_position_threshold
    )
    large_position_aggregate_cap = (
        UCITS_LARGE_POSITION_AGGREGATE_CAP
        if args.large_position_aggregate_cap is None
        else args.large_position_aggregate_cap
    )
    validate_ucits_parameters(
        single_issuer_cap=single_issuer_cap,
        large_position_threshold=large_position_threshold,
        large_position_aggregate_cap=large_position_aggregate_cap,
    )

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
        if (
            np.isclose(single_issuer_cap, 0.10)
            and np.isclose(large_position_threshold, 0.05)
            and np.isclose(large_position_aggregate_cap, 0.40)
        ):
            output_name = "thesis_risk_adjusted_tables_ucits_5_10_40"
        else:
            output_name = "thesis_risk_adjusted_tables_" + ucits_label(
                single_issuer_cap,
                large_position_threshold,
                large_position_aggregate_cap,
            )
        output_dir = portfolio_eval_dir / output_name

    print_identification(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        portfolio_source=portfolio_source,
        factors_csv=factors_csv,
        output_dir=output_dir,
        single_issuer_cap=single_issuer_cap,
        large_position_threshold=large_position_threshold,
        large_position_aggregate_cap=large_position_aggregate_cap,
        nw_lags=args.nw_lags,
    )

    monthly_returns, ucits_holdings, diagnostics = load_ucits_weight_monthly_returns(
        path=portfolio_source,
        single_issuer_cap=single_issuer_cap,
        large_position_threshold=large_position_threshold,
        large_position_aggregate_cap=large_position_aggregate_cap,
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
        rf=rf,
    )
    audit_outputs = save_ucits_audit_outputs(
        output_dir=output_dir,
        ucits_holdings=ucits_holdings,
        diagnostics=diagnostics,
    )
    plot_outputs = vw.save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    print("\nUCITS-weight diagnostics")
    print(f"  UCITS holdings rows: {len(ucits_holdings)}")
    print(f"  monthly portfolio groups: {len(diagnostics)}")
    print(f"  largest UCITS weight: {diagnostics['max_ucits_weight'].max():.4%}")
    print(f"  largest aggregate weight above threshold: {diagnostics['large_position_weight'].max():.4%}")
    print(f"  groups with at least one raw single-cap breach: {int((diagnostics['n_names_single_cap_binding'] > 0).sum())}")
    print(f"  groups where aggregate cap was relaxed: {int(diagnostics['aggregate_cap_relaxed'].sum())}")

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
        n_rows = row_counts.get(key)
        if n_rows is None and path.suffix.lower() == ".csv":
            n_rows = len(pd.read_csv(path))
        if n_rows is None:
            print(f"  {path}")
        else:
            print(f"  {path} ({n_rows} rows)")

    print("\nCreated audit CSV files")
    audit_counts = {
        "ucits_weight_monthly_holdings": len(ucits_holdings),
        "ucits_weight_diagnostics": len(diagnostics),
    }
    for key, path in audit_outputs.items():
        print(f"  {path} ({audit_counts[key]} rows)")

    print("\nCreated plot files")
    for path in plot_outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
