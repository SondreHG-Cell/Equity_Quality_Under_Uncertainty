# demo_diagnostics.py
# ============================================================================
# End-to-end demonstration of the diagnostics in hb_diagnostics.py, using
# synthetic data only. No external data files needed.
#
# What it does:
#   1. Generates a firm-year panel from the accrual model's data-generating
#      process (so we know the "ground truth" firm-level sigma_i values).
#   2. Runs the prior predictive check.
#   3. Fits the baseline model and reports posterior contraction.
#   4. Runs a small sensitivity grid over alternative priors.
#   5. Writes a markdown report with embedded figures.
#
# Runtime: roughly 5-15 minutes on a laptop with the default settings,
# dominated by sensitivity-grid model fits. Reduce N_FIRMS or N_VARIANTS
# at the top of __main__ for a faster smoke-test run.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hb_diagnostics import (
    DEFAULT_ACCRUAL_PRIORS,
    build_basic_variants,
    build_combined_variants,
    make_diagnostic_model,
    posterior_contraction,
    posterior_predictive_check,
    prior_predictive_check,
    save_contraction_plot,
    save_sensitivity_plot,
    sensitivity_grid,
    write_diagnostic_report,
)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPanel:
    data: pd.DataFrame
    firm_sector_map: dict[int, int]
    truth: dict


def generate_synthetic_panel(
    n_firms: int = 80,
    n_sectors: int = 6,
    n_years: int = 10,
    *,
    true_betas: dict[str, float] | None = None,
    true_nu: float = 5.0,
    sigma_i_range: tuple[float, float] = (0.01, 0.08),
    sector_alpha_sd: float = 0.03,
    firm_alpha_sd: float = 0.02,
    predictor_sd: float = 0.15,
    random_seed: int = 42,
) -> SyntheticPanel:
    """
    Generate a firm-year panel from the accrual-model data-generating process.

    Returns a SyntheticPanel with the columns the diagnostics expect:
        firm_idx, sector_idx, Year,
        WCA_scaled, CFO_lag1_scaled, CFO_scaled,
        dREV_scaled, PPE_scaled
    plus a `truth` dict containing the true firm-level sigma_i values.
    """
    rng = np.random.default_rng(random_seed)

    # ---- True coefficients (loosely DD/Jones-style) ----
    if true_betas is None:
        true_betas = {
            "beta_CFO_lag1":  0.15,
            "beta_CFO_curr": -0.55,
            "beta_dREV":      0.10,
            "beta_PPE":      -0.02,
        }

    # ---- Firm-sector assignment ----
    firm_sector_map = {
        i: int(rng.integers(0, n_sectors)) for i in range(n_firms)
    }

    # ---- Hierarchical intercepts ----
    sector_alpha = rng.normal(0.02, sector_alpha_sd, size=n_sectors)
    firm_alpha = np.array([
        sector_alpha[firm_sector_map[i]]
        + rng.normal(0, firm_alpha_sd)
        for i in range(n_firms)
    ])

    # ---- Firm-level noise scale (the quantity of interest) ----
    sigma_i_true = rng.uniform(sigma_i_range[0], sigma_i_range[1], size=n_firms)

    # ---- Build panel rows ----
    rows = []
    for i in range(n_firms):
        # Generate predictors as light AR(1)-ish processes
        cfo = rng.normal(0.05, predictor_sd, size=n_years + 1)
        cfo = 0.4 * np.concatenate([[cfo[0]], cfo[:-1]]) + 0.6 * cfo
        drev = rng.normal(0.03, predictor_sd, size=n_years)
        ppe = np.abs(rng.normal(0.4, 0.15, size=n_years))

        for t in range(n_years):
            cfo_lag = cfo[t]
            cfo_cur = cfo[t + 1]

            mu = (
                firm_alpha[i]
                + true_betas["beta_CFO_lag1"]  * cfo_lag
                + true_betas["beta_CFO_curr"]  * cfo_cur
                + true_betas["beta_dREV"]      * drev[t]
                + true_betas["beta_PPE"]       * ppe[t]
            )
            # Student-t innovation with firm-specific sigma
            eps = sigma_i_true[i] * rng.standard_t(true_nu)
            wca = mu + eps

            rows.append({
                "firm_idx": i,
                "sector_idx": firm_sector_map[i],
                "Year": 2010 + t,
                "WCA_scaled": wca,
                "CFO_lag1_scaled": cfo_lag,
                "CFO_scaled": cfo_cur,
                "dREV_scaled": drev[t],
                "PPE_scaled": ppe[t],
            })

    data = pd.DataFrame(rows)

    truth = {
        "sigma_i": sigma_i_true,
        "firm_alpha": firm_alpha,
        "sector_alpha": sector_alpha,
        "betas": true_betas,
        "nu": true_nu,
    }
    return SyntheticPanel(data=data, firm_sector_map=firm_sector_map, truth=truth)


# ---------------------------------------------------------------------------
# Sensitivity variants
# ---------------------------------------------------------------------------
# The demo runs basic + combined together so the report shows both
# single-parameter perturbations and worst-case combinations.

def build_demo_variants(baseline: dict) -> dict[str, dict]:
    """Combine basic single-parameter sweeps with combined stress tests."""
    return {**build_basic_variants(baseline), **build_combined_variants(baseline)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path("diagnostics_demo_output")
    plot_dir = output_dir / "plots"
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 0. Synthetic data
    # -----------------------------------------------------------------------
    print("Generating synthetic firm-year panel...")
    panel = generate_synthetic_panel(
        n_firms=80, n_sectors=6, n_years=10, random_seed=42
    )
    print(
        f"  Panel: {len(panel.data)} obs, "
        f"{panel.data['firm_idx'].nunique()} firms, "
        f"{panel.data['sector_idx'].nunique()} sectors, "
        f"years {panel.data['Year'].min()}-{panel.data['Year'].max()}"
    )
    print(
        f"  True sigma_i range: "
        f"[{panel.truth['sigma_i'].min():.4f}, "
        f"{panel.truth['sigma_i'].max():.4f}]  "
        f"median {np.median(panel.truth['sigma_i']):.4f}"
    )

    baseline_priors = DEFAULT_ACCRUAL_PRIORS

    # -----------------------------------------------------------------------
    # 1. Prior predictive check
    # -----------------------------------------------------------------------
    print("\n=== Diagnostic 1/3: Prior predictive check ===")
    model_for_prior_pred, _ = make_diagnostic_model(
        panel.data, panel.firm_sector_map, priors=baseline_priors,
    )
    prior_pred = prior_predictive_check(
        model_for_prior_pred,
        observed_y=panel.data["WCA_scaled"].values,
        n_draws=1500,
        random_seed=42,
        figure_path=plot_dir / "prior_predictive.png",
    )
    s = prior_pred.summary
    print(
        f"  obs   range = [{s['obs_min']:+.3f}, {s['obs_max']:+.3f}]  "
        f"std = {s['obs_std']:.3f}"
    )
    print(
        f"  prior 95% CI = [{s['sim_q025']:+.3f}, {s['sim_q975']:+.3f}]  "
        f"std = {s['sim_std']:.3f}"
    )
    print(
        f"  fraction of prior draws within observed range: "
        f"{s['frac_sim_in_obs_range']:.1%}"
    )

    # -----------------------------------------------------------------------
    # 2. Baseline fit + posterior contraction
    # -----------------------------------------------------------------------
    print("\n=== Diagnostic 2/4: Baseline fit + posterior contraction ===")

    # Use sensitivity_grid with just the baseline to get a cached fit;
    # this is the single most efficient way to fit and cache.
    factory = lambda priors: make_diagnostic_model(
        panel.data, panel.firm_sector_map, priors=priors,
    )[0]

    baseline_only = sensitivity_grid(
        model_factory=factory,
        prior_variants={"baseline": baseline_priors},
        target_var="sigma_firm",
        n_draws=800,
        n_tune=1500,
        n_chains=4,
        target_accept=0.95,
        random_seed=42,
        cache_dir=cache_dir,
    )
    baseline_trace = baseline_only["traces"]["baseline"]

    contraction_df = posterior_contraction(baseline_trace, baseline_priors)
    contraction_plot_path = plot_dir / "contraction.png"
    save_contraction_plot(contraction_df, contraction_plot_path)
    print(contraction_df.to_string(index=False))
    contraction_df.to_csv(output_dir / "contraction.csv", index=False)

    # Optional: show how well the posterior recovers the truth
    sigma_post_means = (
        baseline_trace.posterior["sigma_firm"]
        .mean(dim=["chain", "draw"]).values
    )
    sigma_truth = panel.truth["sigma_i"]
    rho = float(
        pd.Series(sigma_post_means).corr(pd.Series(sigma_truth), method="spearman")
    )
    print(f"\n  Recovery check: Spearman rho(true sigma_i, posterior mean) = {rho:.3f}")

    # -----------------------------------------------------------------------
    # 3. Posterior predictive check
    # -----------------------------------------------------------------------
    print("\n=== Diagnostic 3/4: Posterior predictive check ===")
    # Rebuild the model with baseline priors for the posterior predictive
    # call (PyMC needs an explicit model context for posterior_predictive).
    pp_model, _ = make_diagnostic_model(
        panel.data, panel.firm_sector_map, priors=baseline_priors,
    )
    posterior_pred = posterior_predictive_check(
        pp_model,
        baseline_trace,
        observed_y=panel.data["WCA_scaled"].values,
        figure_path=plot_dir / "posterior_predictive.png",
        random_seed=42,
    )
    print("  Bayesian p-values (close to 0.5 = good fit):")
    for stat, p in posterior_pred.summary["bayesian_pvalues"].items():
        flag = "" if 0.05 <= p <= 0.95 else "  <-- check"
        print(f"    {stat:>10s}: {p:.3f}{flag}")

    # -----------------------------------------------------------------------
    # 4. Sensitivity grid
    # -----------------------------------------------------------------------
    print("\n=== Diagnostic 4/4: Sensitivity grid ===")
    variants = build_demo_variants(baseline_priors)
    print(f"  Variants: {list(variants)}")

    sensitivity = sensitivity_grid(
        model_factory=factory,
        prior_variants=variants,
        target_var="sigma_firm",
        n_draws=800,
        n_tune=1500,
        n_chains=4,
        target_accept=0.95,
        random_seed=42,
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
    print("\n  Rank correlations:")
    print(sensitivity["rank_correlations"].round(3).to_string())

    # -----------------------------------------------------------------------
    # 4. Markdown report
    # -----------------------------------------------------------------------
    report_path = output_dir / "diagnostics_report.md"
    write_diagnostic_report(
        report_path,
        title="HB accrual model -- diagnostics demo (synthetic data)",
        metadata={
            "n_firms": panel.data["firm_idx"].nunique(),
            "n_sectors": panel.data["sector_idx"].nunique(),
            "n_obs": len(panel.data),
            "year_range": f"{panel.data['Year'].min()}-{panel.data['Year'].max()}",
            "recovery_spearman_rho": f"{rho:.3f}",
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


if __name__ == "__main__":
    main()
