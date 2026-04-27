# run_diagnostics_real.py
# ============================================================================
# Run validation diagnostics on REAL firm-year data (not synthetic).
#
# This file replaces demo_diagnostics.py's synthetic generator with a call
# to the production data-prep pipeline (hb_shared_utils). Everything else
# uses the same generic diagnostic functions from hb_diagnostics.py.
#
# Diagnostics run:
#   1. Prior predictive check
#   2. Posterior predictive check (with Bayesian p-values)
#   3. Posterior contraction
#   4. Sensitivity grid (basic / extended / combined / all)
#
# Typical usage
# -------------
#   # Quick check, basic 5-variant grid:
#   python run_diagnostics_real.py \
#       --input_csv data/prepared.csv \
#       --output_dir diagnostics/2024 \
#       --portfolio_year 2024 \
#       --grid basic
#
#   # Thorough run with combined worst-case variants:
#   python run_diagnostics_real.py \
#       --input_csv data/prepared.csv \
#       --output_dir diagnostics/2024_thorough \
#       --portfolio_year 2024 \
#       --grid combined
#
#   # Full sweep (slowest):
#   python run_diagnostics_real.py \
#       --input_csv data/prepared.csv \
#       --output_dir diagnostics/2024_full \
#       --portfolio_year 2024 \
#       --grid all
#
# Notes
# -----
# - This file IS dependent on hb_shared_utils for data prep, by design.
#   The diagnostics module (hb_diagnostics.py) remains standalone.
# - Caching is keyed by variant name + hash of priors dict, so re-running
#   with the same configuration reuses traces. Delete the cache directory
#   to force re-fitting.
# - Production sampling defaults (2000 draws, 4000 tune) are heavier than
#   the demo (800/1500). Override with --n_draws and --n_tune if needed.
# ============================================================================

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import arviz as az
import pandas as pd
import pymc as pm

from hb_shared_utils import (
    assign_indices,
    build_estimation_window,
    build_regressors,
    compute_wca,
    mark_usable,
)
from hb_diagnostics import (
    DEFAULT_ACCRUAL_PRIORS,
    build_basic_variants,
    build_combined_variants,
    build_extended_variants,
    make_diagnostic_model,
    posterior_contraction,
    posterior_predictive_check,
    prior_predictive_check,
    save_contraction_plot,
    save_sensitivity_plot,
    sensitivity_grid,
    write_diagnostic_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Variant grid selector
# ---------------------------------------------------------------------------

def select_variants(grid_type: str, baseline: dict) -> dict[str, dict]:
    """Choose a sensitivity grid by name."""
    grid_type = grid_type.lower()
    if grid_type == "basic":
        return build_basic_variants(baseline)
    if grid_type == "extended":
        return build_extended_variants(baseline)
    if grid_type == "combined":
        return build_combined_variants(baseline)
    if grid_type == "all":
        # Merge — basic items already inside extended, so combined is the
        # only thing extended doesn't cover. Use extended ∪ combined.
        return {
            **build_extended_variants(baseline),
            **build_combined_variants(baseline),
        }
    raise ValueError(
        f"grid must be one of: basic, extended, combined, all (got {grid_type!r})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostics_real(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    portfolio_year: int,
    grid_type: str = "basic",
    min_train_years: int = 3,
    max_train_years: int = 5,
    n_draws: int = 2000,
    n_tune: int = 4000,
    n_chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
    n_prior_draws: int = 2000,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    print(f"PyMC {pm.__version__} | ArviZ {az.__version__}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_dir}")
    print(f"Portfolio year: {portfolio_year}")
    print(f"Sensitivity grid: {grid_type}")

    # -----------------------------------------------------------------------
    # Data prep — using the production pipeline
    # -----------------------------------------------------------------------
    data = pd.read_csv(input_csv)
    data = compute_wca(data)
    data = build_regressors(data, include_lead=False)
    data = mark_usable(data)
    data, firm_map, sector_map, firm_sector = assign_indices(data)

    print(
        f"\nPanel: {len(data)} firm-years, {data['Ticker'].nunique()} firms, "
        f"{data['Year'].min()}-{data['Year'].max()}"
    )

    window_df = build_estimation_window(
        data,
        portfolio_year,
        min_train_years=min_train_years,
        max_train_years=max_train_years,
        include_portfolio_year=True,
    )
    if window_df is None:
        raise RuntimeError(
            f"Insufficient training data for portfolio year {portfolio_year}. "
            f"Try a different year or reduce --min_train_years."
        )

    n_obs = len(window_df)
    n_firms = window_df["Ticker"].nunique()
    years_in_window = sorted(int(y) for y in window_df["Year"].unique())
    print(f"Window: {n_obs} obs, {n_firms} firms, years {years_in_window}")

    baseline_priors = DEFAULT_ACCRUAL_PRIORS

    # The factory used by every diagnostic that fits the model
    factory = lambda priors: make_diagnostic_model(
        window_df, firm_sector, priors=priors
    )[0]

    # -----------------------------------------------------------------------
    # 1. Prior predictive check
    # -----------------------------------------------------------------------
    print("\n=== 1/4: Prior predictive check ===")
    prior_pred_model, _ = make_diagnostic_model(
        window_df, firm_sector, priors=baseline_priors
    )
    prior_pred = prior_predictive_check(
        prior_pred_model,
        observed_y=window_df["WCA_scaled"].values,
        n_draws=n_prior_draws,
        random_seed=random_seed,
        figure_path=plot_dir / "prior_predictive.png",
    )
    s = prior_pred.summary
    print(
        f"  observed range: [{s['obs_min']:+.3f}, {s['obs_max']:+.3f}], "
        f"std {s['obs_std']:.3f}"
    )
    print(
        f"  prior 95% CI:   [{s['sim_q025']:+.3f}, {s['sim_q975']:+.3f}], "
        f"std {s['sim_std']:.3f}"
    )
    print(f"  fraction in observed range: {s['frac_sim_in_obs_range']:.1%}")

    # -----------------------------------------------------------------------
    # 2. Baseline fit + posterior contraction
    # -----------------------------------------------------------------------
    print("\n=== 2/4: Baseline fit + posterior contraction ===")
    baseline_only = sensitivity_grid(
        model_factory=factory,
        prior_variants={"baseline": baseline_priors},
        target_var="sigma_firm",
        n_draws=n_draws,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept,
        random_seed=random_seed,
        cache_dir=cache_dir,
    )
    baseline_trace = baseline_only["traces"]["baseline"]

    contraction_df = posterior_contraction(baseline_trace, baseline_priors)
    contraction_plot_path = plot_dir / "contraction.png"
    save_contraction_plot(contraction_df, contraction_plot_path)
    print(contraction_df.to_string(index=False))
    contraction_df.to_csv(output_dir / "contraction.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Posterior predictive check
    # -----------------------------------------------------------------------
    print("\n=== 3/4: Posterior predictive check ===")
    pp_model, _ = make_diagnostic_model(
        window_df, firm_sector, priors=baseline_priors
    )
    posterior_pred = posterior_predictive_check(
        pp_model,
        baseline_trace,
        observed_y=window_df["WCA_scaled"].values,
        figure_path=plot_dir / "posterior_predictive.png",
        random_seed=random_seed,
    )
    print("  Bayesian p-values (close to 0.5 = good fit):")
    for stat, p in posterior_pred.summary["bayesian_pvalues"].items():
        flag = "" if 0.05 <= p <= 0.95 else "  <-- check"
        print(f"    {stat:>10s}: {p:.3f}{flag}")

    # -----------------------------------------------------------------------
    # 4. Sensitivity grid
    # -----------------------------------------------------------------------
    print(f"\n=== 4/4: Sensitivity grid ({grid_type}) ===")
    variants = select_variants(grid_type, baseline_priors)
    print(f"  Variants ({len(variants)}): {list(variants)}")

    sensitivity = sensitivity_grid(
        model_factory=factory,
        prior_variants=variants,
        target_var="sigma_firm",
        n_draws=n_draws,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept,
        random_seed=random_seed,
        cache_dir=cache_dir,
    )

    sensitivity_plot_path = plot_dir / "sensitivity.png"
    save_sensitivity_plot(
        sensitivity["target_means"],
        sensitivity_plot_path,
        baseline="baseline",
        target_label="sigma_i",
    )
    sensitivity["target_means"].to_csv(output_dir / "sensitivity_sigma_means.csv")
    sensitivity["rank_correlations"].to_csv(output_dir / "sensitivity_rank_corr.csv")
    sensitivity["summary"].to_csv(output_dir / "sensitivity_summary.csv", index=False)

    print("\n  Pairwise summary vs baseline:")
    print(sensitivity["summary"].to_string(index=False))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report_path = output_dir / "diagnostics_report.md"
    write_diagnostic_report(
        report_path,
        title=f"HB accrual model -- diagnostics, portfolio year {portfolio_year}",
        metadata={
            "input_csv": str(input_csv),
            "portfolio_year": portfolio_year,
            "n_firms_in_window": n_firms,
            "n_obs_in_window": n_obs,
            "window_years": f"{min(years_in_window)}-{max(years_in_window)}",
            "n_draws": n_draws,
            "n_tune": n_tune,
            "n_chains": n_chains,
            "sensitivity_grid": grid_type,
            "n_variants": len(variants),
        },
        prior_pred=prior_pred,
        posterior_pred=posterior_pred,
        contraction_df=contraction_df,
        sensitivity=sensitivity,
        contraction_plot=contraction_plot_path,
        sensitivity_plot=sensitivity_plot_path,
    )
    print(f"\n=== Done ===\nReport: {report_path}")
    print(f"Plots:  {plot_dir}")
    print(f"Cache:  {cache_dir}")

    return {
        "report_path": str(report_path),
        "plot_dir": str(plot_dir),
        "cache_dir": str(cache_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run HB accrual-model validation diagnostics on real data."
    )
    p.add_argument("--input_csv", type=str, required=True,
                   help="Prepared firm-year panel CSV.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save diagnostic outputs.")
    p.add_argument("--portfolio_year", type=int, required=True,
                   help="Portfolio year to fit the model on.")
    p.add_argument("--grid", type=str, default="basic",
                   choices=["basic", "extended", "combined", "all"],
                   help="Sensitivity grid to run.")
    p.add_argument("--min_train_years", type=int, default=3)
    p.add_argument("--max_train_years", type=int, default=5)
    p.add_argument("--n_draws", type=int, default=2000)
    p.add_argument("--n_tune", type=int, default=4000)
    p.add_argument("--n_chains", type=int, default=4)
    p.add_argument("--target_accept", type=float, default=0.95)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--n_prior_draws", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostics_real(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        portfolio_year=args.portfolio_year,
        grid_type=args.grid,
        min_train_years=args.min_train_years,
        max_train_years=args.max_train_years,
        n_draws=args.n_draws,
        n_tune=args.n_tune,
        n_chains=args.n_chains,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        n_prior_draws=args.n_prior_draws,
    )
