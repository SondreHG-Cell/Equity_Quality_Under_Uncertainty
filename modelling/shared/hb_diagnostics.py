# hb_diagnostics.py
# ============================================================================
# Self-contained validation diagnostics for hierarchical Bayesian accrual
# models. This module has NO DEPENDENCIES on the production pipeline; you
# can read, run, and evaluate it in isolation.
#
# Three core diagnostic functions:
#   1. prior_predictive_check   — does the prior imply plausible observed data?
#   2. posterior_contraction    — how much did the data update each prior?
#   3. sensitivity_grid         — do conclusions hold under alternative priors?
#
# Plus convenience pieces for the specific accrual model in this project:
#   make_diagnostic_model       — reference model builder (parameterized priors)
#   make_scaled_variant         — build a sensitivity variant by scaling
#                                 one prior field
#   write_diagnostic_report     — emit a single markdown artifact
#
# Design philosophy
# -----------------
# The three diagnostic functions are model-agnostic: they accept a pm.Model
# (or a model_factory: priors -> pm.Model). The reference accrual model is
# provided so the module is immediately runnable, but you can replace it
# with your own model factory without touching the diagnostic logic.
# ============================================================================

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


# ---------------------------------------------------------------------------
# Default priors for the reference accrual model
# ---------------------------------------------------------------------------
# Each entry MUST include "dist" so that posterior_contraction can compute
# the correct prior variance. Supported distributions:
#   "Normal"     : keys mu, sigma
#   "HalfNormal" : key  sigma
# ---------------------------------------------------------------------------
DEFAULT_ACCRUAL_PRIORS: dict[str, dict] = {
    "mu_0":           {"dist": "Normal",      "mu": 0.0, "sigma": 0.1},
    "omega":          {"dist": "HalfNormal",  "sigma": 0.05},
    "tau":            {"dist": "HalfNormal",  "sigma": 0.05},
    "sigma_0":        {"dist": "HalfNormal",  "sigma": 0.05},
    "beta_CFO_lag1":  {"dist": "Normal",      "mu": 0.0, "sigma": 0.3},
    "beta_CFO_curr":  {"dist": "Normal",      "mu": 0.0, "sigma": 0.3},
    "beta_dREV":      {"dist": "Normal",      "mu": 0.0, "sigma": 0.3},
    "beta_PPE":       {"dist": "Normal",      "mu": 0.0, "sigma": 0.3},
    "nu_minus_two":   {"dist": "Exponential", "lam": 0.1},
}


# ---------------------------------------------------------------------------
# Prior variance helpers
# ---------------------------------------------------------------------------

def prior_variance(prior_spec: dict) -> float:
    """
    Variance of a prior distribution given its spec dict.

      Normal(mu, sigma)        : Var = sigma**2
      HalfNormal(sigma)        : Var = sigma**2 * (1 - 2/pi)
      Exponential(lam)         : Var = 1 / lam**2
    """
    dist = prior_spec["dist"]
    if dist == "Normal":
        return float(prior_spec["sigma"]) ** 2
    if dist == "HalfNormal":
        return float(prior_spec["sigma"]) ** 2 * (1.0 - 2.0 / np.pi)
    if dist == "Exponential":
        return 1.0 / (float(prior_spec["lam"]) ** 2)
    raise ValueError(f"Unsupported prior distribution: {dist}")


def make_scaled_variant(
    baseline: dict,
    parameter: str,
    scale: float,
    *,
    field: str = "sigma",
) -> dict:
    """
    Build a variant prior dict by scaling one field of one parameter.

    Examples
    --------
    >>> make_scaled_variant(DEFAULT_ACCRUAL_PRIORS, "sigma_0", 2.0)
        Doubles the HalfNormal prior SD on sigma_0.

    >>> make_scaled_variant(DEFAULT_ACCRUAL_PRIORS, "nu_minus_two", 0.5, field="lam")
        Halves the Exponential rate, doubling E[nu - 2] (heavier tails).
    """
    if parameter not in baseline:
        raise KeyError(f"Parameter {parameter!r} not in baseline priors")
    if field not in baseline[parameter]:
        raise KeyError(
            f"Field {field!r} not in prior spec for {parameter}: "
            f"{list(baseline[parameter])}"
        )
    new = copy.deepcopy(baseline)
    new[parameter][field] = float(baseline[parameter][field]) * float(scale)
    return new


def widen_priors(
    baseline: dict,
    scale: float,
    *,
    prefix: str = "beta_",
    field: str = "sigma",
) -> dict:
    """
    Build a variant prior dict by scaling one field of every parameter
    whose name starts with `prefix`.

    Useful for testing the "data-dominated" claim on a whole group of
    coefficients at once (e.g. widen all betas 3x and check that
    posterior sigma_i estimates are unchanged).
    """
    new = copy.deepcopy(baseline)
    for name, spec in new.items():
        if name.startswith(prefix) and field in spec:
            spec[field] = float(spec[field]) * float(scale)
    return new


def compose_variant(
    baseline: dict,
    scalings: dict[str, dict[str, float]],
) -> dict:
    """
    Build a variant by scaling multiple prior fields at once.

    `scalings` is a nested dict: {parameter_name: {field_name: scale_factor}}.
    The parameters/fields must already exist in the baseline.

    Example
    -------
    >>> compose_variant(DEFAULT_ACCRUAL_PRIORS, {
    ...     "sigma_0": {"sigma": 2.0},
    ...     "omega":   {"sigma": 2.0},
    ...     "nu":      {"alpha": 0.5},
    ... })
    """
    new = copy.deepcopy(baseline)
    for param, field_scales in scalings.items():
        if param not in new:
            raise KeyError(f"Parameter {param!r} not in baseline")
        for field, scale in field_scales.items():
            if field not in new[param]:
                raise KeyError(
                    f"Field {field!r} not in {param}: {list(new[param])}"
                )
            new[param][field] = float(new[param][field]) * float(scale)
    return new


def _scale_label(s: float) -> str:
    """Human-readable label for a scale factor."""
    table = {
        0.125: "eighth",
        0.25: "quarter",
        0.5: "half",
        2.0: "double",
        3.0: "3x",
        4.0: "4x",
        8.0: "8x",
        10.0: "10x",
    }
    if s in table:
        return table[s]
    return f"{s:g}x".replace(".", "p")


def build_basic_variants(baseline: dict) -> dict[str, dict]:
    """
    Small grid (5 variants + baseline) covering the key sensitivity claims:
    one stress-test each for sigma_0, tau, nu, plus widening all betas.
    Suitable for quick checks and the demo.
    """
    return {
        "baseline":       baseline,
        "sigma0_half":    make_scaled_variant(baseline, "sigma_0", 0.5),
        "sigma0_double":  make_scaled_variant(baseline, "sigma_0", 2.0),
        "tau_double":     make_scaled_variant(baseline, "tau", 2.0),
        "nu_loose":       make_scaled_variant(baseline, "nu_minus_two", 0.5, field="lam"),
        "betas_3x_wider": widen_priors(baseline, 3.0, prefix="beta_"),
    }


def build_extended_variants(baseline: dict) -> dict[str, dict]:
    """
    Single-parameter sweep across multiple scales. ~17 variants + baseline.
    For each critical/moderate-tier parameter, we test tightening and
    loosening at multiple magnitudes.
    """
    v: dict[str, dict] = {"baseline": baseline}

    # sigma_0 — most prior-sensitive parameter; sweep widely
    for s in (0.25, 0.5, 2.0, 4.0, 8.0):
        v[f"sigma0_{_scale_label(s)}"] = make_scaled_variant(baseline, "sigma_0", s)

    # omega and tau — pooling strength
    for s in (0.5, 2.0, 4.0):
        v[f"omega_{_scale_label(s)}"] = make_scaled_variant(baseline, "omega", s)
        v[f"tau_{_scale_label(s)}"] = make_scaled_variant(baseline, "tau", s)

    # mu_0 — overall intercept location
    for s in (0.5, 2.0):
        v[f"mu0_{_scale_label(s)}"] = make_scaled_variant(baseline, "mu_0", s)

    # nu — Student-t degrees of freedom
    for s in (0.25, 0.5, 2.0, 4.0):
        v[f"nu_{_scale_label(s)}"] = make_scaled_variant(
            baseline, "nu_minus_two", s, field="lam"
        )

    # All betas at once
    for s in (0.5, 3.0, 10.0):
        v[f"betas_{_scale_label(s)}"] = widen_priors(baseline, s, prefix="beta_")

    return v


def build_combined_variants(baseline: dict) -> dict[str, dict]:
    """
    Multi-parameter perturbations — worst-case combinations. ~5 variants
    + baseline. These test whether interactions between prior choices
    propagate to sigma_i estimates, which single-parameter sweeps can miss.
    """
    return {
        "baseline": baseline,
        # All hierarchical variances loosened together
        "all_variance_2x": compose_variant(baseline, {
            "sigma_0": {"sigma": 2.0},
            "omega":   {"sigma": 2.0},
            "tau":     {"sigma": 2.0},
        }),
        # All hierarchical variances tightened together
        "all_variance_half": compose_variant(baseline, {
            "sigma_0": {"sigma": 0.5},
            "omega":   {"sigma": 0.5},
            "tau":     {"sigma": 0.5},
        }),
        # Heaviest-tail scenario: noise scale up + tails heavier
        "sigma0_2x_nu_loose": compose_variant(baseline, {
            "sigma_0":      {"sigma": 2.0},
            "nu_minus_two": {"lam": 0.5},
        }),
        # Adversarial-max: every uncertainty source amplified
        "stress_max": compose_variant(baseline, {
            "sigma_0":      {"sigma": 4.0},
            "omega":        {"sigma": 2.0},
            "tau":          {"sigma": 2.0},
            "mu_0":         {"sigma": 2.0},
            "nu_minus_two": {"lam": 0.5},
        }),
        # Adversarial-min: minimise variance allowances
        "stress_min": compose_variant(baseline, {
            "sigma_0": {"sigma": 0.5},
            "omega":   {"sigma": 0.5},
            "tau":     {"sigma": 0.5},
            "mu_0":    {"sigma": 0.5},
        }),
    }


# ---------------------------------------------------------------------------
# Reference accrual model builder
# ---------------------------------------------------------------------------
# Same hierarchical structure as the production accrual model:
#
#   alpha_sector ~ N(mu_0, omega)                      (sector intercepts)
#   alpha_firm   ~ N(alpha_sector, tau)                (firm intercepts)
#   sigma_sector ~ HalfNormal(sigma_0)                 (sector noise scale)
#   sigma_firm   ~ HalfNormal(sigma_sector)            (firm noise scale)
#   WCA          ~ StudentT(nu, mu_wca, sigma_firm)
#
#   mu_wca = alpha_firm + sum_k beta_k * predictor_k
#
# Implemented with non-centered parametrization for sampling robustness.
# ---------------------------------------------------------------------------

def make_diagnostic_model(
    window_df: pd.DataFrame,
    firm_sector_map,
    priors: dict | None = None,
) -> tuple[pm.Model, dict]:
    """
    Reference accrual model builder, parameterized by priors.

    Required columns in window_df:
      WCA_scaled, CFO_lag1_scaled, CFO_scaled, dREV_scaled, PPE_scaled,
      firm_idx.

    firm_sector_map : mapping from firm_idx -> sector_idx.
    """
    if priors is None:
        priors = DEFAULT_ACCRUAL_PRIORS

    wdf = window_df.copy()

    window_firms = sorted(wdf["firm_idx"].unique())
    firm_remap = {old: new for new, old in enumerate(window_firms)}
    wdf["w_firm"] = wdf["firm_idx"].map(firm_remap).to_numpy()

    firm_to_sector = np.array([firm_sector_map[old] for old in window_firms])
    window_sectors = sorted(set(firm_to_sector))
    sector_remap = {old: new for new, old in enumerate(window_sectors)}
    firm_to_sector = np.array(
        [sector_remap[s] for s in firm_to_sector], dtype=int
    )

    firm_idx = wdf["w_firm"].values.astype(int)

    y = wdf["WCA_scaled"].values
    cfo_lag1 = wdf["CFO_lag1_scaled"].values
    cfo_curr = wdf["CFO_scaled"].values
    drev = wdf["dREV_scaled"].values
    ppe = wdf["PPE_scaled"].values

    coords = {
        "firm": window_firms,
        "sector": window_sectors,
        "obs": np.arange(len(wdf)),
    }
    p = priors

    with pm.Model(coords=coords) as model:
        mu_0 = pm.Normal(
            "mu_0", mu=p["mu_0"]["mu"], sigma=p["mu_0"]["sigma"]
        )
        omega = pm.HalfNormal("omega", sigma=p["omega"]["sigma"])
        tau = pm.HalfNormal("tau", sigma=p["tau"]["sigma"])
        sigma_0 = pm.HalfNormal("sigma_0", sigma=p["sigma_0"]["sigma"])

        alpha_sector_raw = pm.Normal(
            "alpha_sector_raw", mu=0, sigma=1, dims="sector"
        )
        alpha_sector = pm.Deterministic(
            "alpha_sector",
            mu_0 + omega * alpha_sector_raw,
            dims="sector",
        )

        alpha_firm_raw = pm.Normal(
            "alpha_firm_raw", mu=0, sigma=1, dims="firm"
        )
        alpha_firm = pm.Deterministic(
            "alpha_firm",
            alpha_sector[firm_to_sector] + tau * alpha_firm_raw,
            dims="firm",
        )

        sigma_sector_raw = pm.HalfNormal(
            "sigma_sector_raw", sigma=1, dims="sector"
        )
        sigma_sector = pm.Deterministic(
            "sigma_sector",
            sigma_0 * sigma_sector_raw,
            dims="sector",
        )

        sigma_firm_raw = pm.HalfNormal(
            "sigma_firm_raw", sigma=1, dims="firm"
        )
        sigma_firm = pm.Deterministic(
            "sigma_firm",
            sigma_sector[firm_to_sector] * sigma_firm_raw,
            dims="firm",
        )

        b_lag = pm.Normal(
            "beta_CFO_lag1",
            mu=p["beta_CFO_lag1"]["mu"],
            sigma=p["beta_CFO_lag1"]["sigma"],
        )
        b_cur = pm.Normal(
            "beta_CFO_curr",
            mu=p["beta_CFO_curr"]["mu"],
            sigma=p["beta_CFO_curr"]["sigma"],
        )
        b_rev = pm.Normal(
            "beta_dREV",
            mu=p["beta_dREV"]["mu"],
            sigma=p["beta_dREV"]["sigma"],
        )
        b_ppe = pm.Normal(
            "beta_PPE",
            mu=p["beta_PPE"]["mu"],
            sigma=p["beta_PPE"]["sigma"],
        )

        nu_minus_two = pm.Exponential(
            "nu_minus_two",
            lam=p["nu_minus_two"]["lam"],
        )
        nu = pm.Deterministic("nu", 2.0 + nu_minus_two)

        mu_wca = (
            alpha_firm[firm_idx]
            + b_lag * cfo_lag1
            + b_cur * cfo_curr
            + b_rev * drev
            + b_ppe * ppe
        )

        pm.StudentT(
            "WCA_obs",
            nu=nu,
            mu=mu_wca,
            sigma=sigma_firm[firm_idx],
            observed=y,
            dims="obs",
        )

        sigma_firm_sd = pm.Deterministic(
            "sigma_firm_sd",
            sigma_firm * pm.math.sqrt(nu / (nu - 2.0)),
            dims="firm",
        )

    info = {
        "window_firms": window_firms,
        "window_sectors": window_sectors,
        "firm_to_sector": firm_to_sector,
        "n_obs": int(len(wdf)),
        "priors": priors,
    }
    return model, info


# ---------------------------------------------------------------------------
# 1. Prior predictive check
# ---------------------------------------------------------------------------

@dataclass
class PriorPredictiveResult:
    observed: np.ndarray
    simulated: np.ndarray
    summary: dict[str, float]
    figure_path: Path | None = None


def prior_predictive_check(
    model: pm.Model,
    observed_y: np.ndarray,
    *,
    obs_var_name: str = "WCA_obs",
    n_draws: int = 1000,
    random_seed: int = 42,
    figure_path: Path | None = None,
) -> PriorPredictiveResult:
    """
    Sample observable values from the prior alone and compare to data.

    Generic over any pm.Model with an observed variable named `obs_var_name`.
    """
    with model:
        prior_pred = pm.sample_prior_predictive(
            draws=n_draws, random_seed=random_seed
        )

    observed = np.asarray(observed_y, dtype=float)
    simulated = prior_pred.prior_predictive[obs_var_name].values.flatten()

    obs_lo, obs_hi = float(observed.min()), float(observed.max())
    summary = {
        "n_observed": int(observed.size),
        "n_simulated": int(simulated.size),
        "obs_min": obs_lo,
        "obs_max": obs_hi,
        "obs_q025": float(np.quantile(observed, 0.025)),
        "obs_q975": float(np.quantile(observed, 0.975)),
        "obs_std": float(observed.std()),
        "sim_q025": float(np.quantile(simulated, 0.025)),
        "sim_q975": float(np.quantile(simulated, 0.975)),
        "sim_q0001": float(np.quantile(simulated, 0.0001)),
        "sim_q9999": float(np.quantile(simulated, 0.9999)),
        "sim_std": float(simulated.std()),
        "frac_sim_in_obs_range": float(
            ((simulated >= obs_lo) & (simulated <= obs_hi)).mean()
        ),
        "frac_sim_extreme": float((np.abs(simulated) > 1.0).mean()),
    }

    saved_path: Path | None = None
    if figure_path is not None:
        saved_path = _save_prior_predictive_plot(observed, simulated, figure_path)

    return PriorPredictiveResult(
        observed=observed,
        simulated=simulated,
        summary=summary,
        figure_path=saved_path,
    )


def _save_prior_predictive_plot(
    observed: np.ndarray,
    simulated: np.ndarray,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sim_clipped = np.clip(simulated, -2.0, 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(-1.5, 1.5, 80)

    axes[0].hist(
        sim_clipped, bins=bins, density=True, alpha=0.4,
        label="Prior predictive (clipped to [-2, 2])",
        color="steelblue",
    )
    axes[0].hist(
        observed, bins=bins, density=True, alpha=0.7,
        label="Observed",
        color="darkorange",
    )
    axes[0].axvline(observed.min(), color="black", lw=0.8, ls=":")
    axes[0].axvline(observed.max(), color="black", lw=0.8, ls=":")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prior predictive vs observed (full)")
    axes[0].legend(fontsize=9)

    bins_tail = np.linspace(-3.0, 3.0, 100)
    axes[1].hist(
        simulated, bins=bins_tail, density=True, alpha=0.4,
        label="Prior predictive",
        color="steelblue",
    )
    axes[1].hist(
        observed, bins=bins_tail, density=True, alpha=0.7,
        label="Observed",
        color="darkorange",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Density (log)")
    axes[1].set_title("Tail behaviour")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Posterior predictive check
# ---------------------------------------------------------------------------

@dataclass
class PosteriorPredictiveResult:
    observed: np.ndarray
    simulated: np.ndarray  # shape: (n_replicates, n_obs)
    summary: dict[str, Any]
    figure_path: Path | None = None


def _skewness(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    s = y.std()
    if s == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 3) / s ** 3)


def _excess_kurtosis(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    s = y.std()
    if s == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 4) / s ** 4 - 3.0)


def posterior_predictive_check(
    model: pm.Model,
    trace: az.InferenceData,
    observed_y: np.ndarray,
    *,
    obs_var_name: str = "WCA_obs",
    figure_path: Path | None = None,
    random_seed: int = 42,
) -> PosteriorPredictiveResult:
    """
    Sample observable values from the posterior and compare to data.

    For each test statistic T, computes a Bayesian p-value:
        p = mean( T(y_rep) >= T(y_obs) )

    Values near 0.5 indicate good fit. Values near 0 or 1 indicate the
    model fails to capture that aspect of the data (e.g. a p-value of 0.99
    on the observed maximum means the model rarely produces values as
    large as observed — a tail-fit failure).
    """
    with model:
        post_pred = pm.sample_posterior_predictive(
            trace, random_seed=random_seed
        )

    observed = np.asarray(observed_y, dtype=float)
    sim_arr = post_pred.posterior_predictive[obs_var_name].values
    # Flatten chain x draw, keep observation dimension
    n_chain, n_draw = sim_arr.shape[0], sim_arr.shape[1]
    simulated = sim_arr.reshape(n_chain * n_draw, -1)

    test_stats = {
        "mean":     np.mean,
        "std":      np.std,
        "skew":     _skewness,
        "kurtosis": _excess_kurtosis,
        "min":      np.min,
        "max":      np.max,
        "q05":      lambda y: np.quantile(y, 0.05),
        "q95":      lambda y: np.quantile(y, 0.95),
    }

    bayesian_pvalues = {}
    obs_stats = {}
    rep_means = {}
    for name, T in test_stats.items():
        T_obs = float(T(observed))
        T_rep = np.array([T(simulated[i]) for i in range(simulated.shape[0])])
        p = float(np.mean(T_rep >= T_obs))
        bayesian_pvalues[name] = p
        obs_stats[name] = T_obs
        rep_means[name] = float(T_rep.mean())

    summary = {
        "n_observed": int(observed.size),
        "n_replicates": int(simulated.shape[0]),
        "obs_stats": obs_stats,
        "rep_stat_means": rep_means,
        "bayesian_pvalues": bayesian_pvalues,
    }

    saved_path: Path | None = None
    if figure_path is not None:
        saved_path = _save_posterior_predictive_plot(
            observed, simulated, figure_path
        )

    return PosteriorPredictiveResult(
        observed=observed,
        simulated=simulated,
        summary=summary,
        figure_path=saved_path,
    )


def _save_posterior_predictive_plot(
    observed: np.ndarray,
    simulated: np.ndarray,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pooled = simulated.flatten()
    obs_lo, obs_hi = observed.min(), observed.max()
    pad = 0.5 * max(abs(obs_lo), abs(obs_hi))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: full distribution overlay
    bins = np.linspace(obs_lo - pad, obs_hi + pad, 80)
    axes[0].hist(
        pooled, bins=bins, density=True, alpha=0.4,
        label="Posterior predictive (pooled)", color="steelblue",
    )
    axes[0].hist(
        observed, bins=bins, density=True, alpha=0.7,
        label="Observed", color="darkorange",
    )
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Posterior predictive vs observed (full)")
    axes[0].legend(fontsize=9)

    # Panel 2: tail (log scale)
    sd = float(observed.std())
    bins_tail = np.linspace(-3 * sd, 3 * sd, 100)
    axes[1].hist(
        pooled, bins=bins_tail, density=True, alpha=0.4,
        label="Posterior predictive", color="steelblue",
    )
    axes[1].hist(
        observed, bins=bins_tail, density=True, alpha=0.7,
        label="Observed", color="darkorange",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Density (log)")
    axes[1].set_title("Tail behaviour")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Posterior contraction
# ---------------------------------------------------------------------------

# By default, contraction is reported only for these parameters. The `_raw`
# variables in non-centered parametrization are skipped because their priors
# are by construction Normal(0,1) / HalfNormal(1) — contraction on those is
# uninformative.
_DEFAULT_CONTRACTION_PARAMS = [
    "mu_0", "omega", "tau", "sigma_0",
    "beta_CFO_lag1", "beta_CFO_curr",
    "beta_dREV", "beta_PPE",
    "nu_minus_two",
]


def posterior_contraction(
    trace: az.InferenceData,
    priors: dict,
    *,
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute prior-to-posterior contraction for each scalar parameter.

      contraction = 1 - posterior_var / prior_var

    Values near 1: data dominated. Values near 0: prior dominated.
    Negative: posterior wider than prior — usually a misspecification flag.
    """
    params = parameters or _DEFAULT_CONTRACTION_PARAMS
    rows = []
    for name in params:
        if name not in priors:
            continue
        if name not in trace.posterior:
            continue

        prior_var_val = prior_variance(priors[name])
        post = trace.posterior[name].values
        post_sd = float(post.std())
        post_mean = float(post.mean())
        contraction = (
            1.0 - (post_sd ** 2) / prior_var_val
            if prior_var_val > 0 else np.nan
        )

        rows.append({
            "parameter": name,
            "prior_dist": priors[name]["dist"],
            "prior_sd": float(np.sqrt(prior_var_val)),
            "posterior_mean": post_mean,
            "posterior_sd": post_sd,
            "contraction": float(contraction),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("contraction", ascending=False).reset_index(drop=True)
    return df


def save_contraction_plot(
    contraction_df: pd.DataFrame,
    path: Path,
) -> Path:
    """Bar chart of contraction values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = contraction_df.sort_values("contraction", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(df))))
    colors = [
        "#2ca02c" if c > 0.9 else "#ff7f0e" if c > 0.5 else "#d62728"
        for c in df["contraction"]
    ]
    ax.barh(df["parameter"], df["contraction"], color=colors, alpha=0.8)
    ax.axvline(0.9, color="black", lw=0.6, ls=":", label="0.9 (data dominates)")
    ax.axvline(0.5, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Contraction = 1 - sigma^2_post / sigma^2_prior")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Prior-to-posterior contraction by parameter")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Sensitivity grid
# ---------------------------------------------------------------------------

def _hash_priors(priors: dict) -> str:
    blob = json.dumps(priors, sort_keys=True, default=float).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def sensitivity_grid(
    model_factory: Callable[[dict], pm.Model],
    prior_variants: dict[str, dict],
    *,
    target_var: str = "sigma_firm",
    n_draws: int = 1000,
    n_tune: int = 2000,
    n_chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
    cache_dir: Path | None = None,
    progressbar: bool = True,
) -> dict[str, Any]:
    """
    Fit the model under each prior variant; summarise how the posterior
    of `target_var` (default 'sigma_firm') differs across variants.

    Parameters
    ----------
    model_factory : callable
        Maps a priors dict to a pm.Model. Called once per variant.
    prior_variants : dict[str, dict]
        Mapping from variant name -> priors dict. The first variant is
        treated as baseline for pairwise summaries.
    target_var : str
        Posterior variable to compare across variants (e.g. firm-level
        sigma). Must be a vector indexed by 'firm' or similar.
    cache_dir : Path, optional
        If provided, traces are saved/loaded as NetCDF files keyed by a
        hash of each priors dict.

    Returns
    -------
    dict with keys:
      "traces"            : {variant_name: InferenceData}
      "target_means"      : DataFrame, rows = entries of target_var,
                            columns = variants
      "rank_correlations" : pairwise Spearman correlation matrix
      "summary"           : DataFrame of |delta| stats per variant vs baseline
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    traces: dict[str, az.InferenceData] = {}
    means_per_variant: dict[str, np.ndarray] = {}
    target_index: list | None = None

    for variant_name, priors in prior_variants.items():
        cache_path = None
        if cache_dir is not None:
            tag = _hash_priors(priors)
            cache_path = cache_dir / f"sensitivity_{variant_name}_{tag}.nc"

        if cache_path is not None and cache_path.exists():
            print(f"[sensitivity] {variant_name}: loading cached trace")
            trace = az.from_netcdf(cache_path)
        else:
            print(f"[sensitivity] {variant_name}: fitting model")
            model = model_factory(priors)
            with model:
                trace = pm.sample(
                    draws=n_draws,
                    tune=n_tune,
                    chains=n_chains,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    return_inferencedata=True,
                )
            if cache_path is not None:
                az.to_netcdf(trace, cache_path)

        traces[variant_name] = trace

        if target_var not in trace.posterior:
            raise KeyError(
                f"target_var={target_var!r} not in posterior. "
                f"Available: {list(trace.posterior.data_vars)}"
            )
        target_post = trace.posterior[target_var]
        means = target_post.mean(dim=["chain", "draw"]).values
        means_per_variant[variant_name] = means

        if target_index is None:
            dims = [d for d in target_post.dims if d not in ("chain", "draw")]
            if len(dims) == 1:
                target_index = list(target_post.coords[dims[0]].values)
            else:
                target_index = list(range(len(means)))

    target_means = pd.DataFrame(means_per_variant, index=target_index)
    target_means.index.name = target_var
    rank_corr = target_means.corr(method="spearman")

    baseline_name = next(iter(prior_variants))
    base = target_means[baseline_name]
    pair_rows = []
    for name in target_means.columns:
        if name == baseline_name:
            continue
        diff = (target_means[name] - base).abs()
        rel = diff / base.replace(0, np.nan)
        pair_rows.append({
            "variant": name,
            "vs_baseline": baseline_name,
            "median_abs_diff": float(diff.median()),
            "p95_abs_diff": float(diff.quantile(0.95)),
            "median_rel_diff_pct": float(100 * rel.median()),
            "p95_rel_diff_pct": float(100 * rel.quantile(0.95)),
            "spearman_rho": float(rank_corr.loc[baseline_name, name]),
        })
    summary = pd.DataFrame(pair_rows)

    return {
        "traces": traces,
        "target_means": target_means,
        "rank_correlations": rank_corr,
        "summary": summary,
    }


def save_sensitivity_plot(
    target_means: pd.DataFrame,
    path: Path,
    *,
    baseline: str | None = None,
    target_label: str = "sigma_i",
) -> Path:
    """Scatter of baseline vs each variant."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if baseline is None:
        baseline = target_means.columns[0]
    others = [c for c in target_means.columns if c != baseline]

    if not others:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(target_means[baseline], bins=40, color="steelblue", alpha=0.7)
        ax.set_xlabel(f"{target_label} ({baseline})")
        ax.set_title(f"{target_label} distribution")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    n = len(others)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False,
    )

    base_vals = target_means[baseline].values
    lo = float(min(target_means.values.min(), 0))
    hi = float(target_means.values.max()) * 1.05

    for i, name in enumerate(others):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.scatter(base_vals, target_means[name].values,
                   alpha=0.35, s=10, color="steelblue")
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls="--")
        rho = target_means[[baseline, name]].corr(method="spearman").iloc[0, 1]
        ax.set_xlabel(f"{target_label} ({baseline})")
        ax.set_ylabel(f"{target_label} ({name})")
        ax.set_title(f"{name}  (Spearman rho = {rho:.3f})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _df_to_markdown(
    df: pd.DataFrame,
    *,
    index: bool = True,
    floatfmt: str = ".4f",
) -> str:
    """
    Render a DataFrame as a GitHub-style markdown table without requiring
    the optional 'tabulate' dependency that pandas.to_markdown uses.
    """
    cols = [str(c) for c in df.columns]
    headers = ([str(df.index.name or "")] + cols) if index else cols

    def fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            return format(v, floatfmt)
        return str(v)

    body_rows = []
    for idx, row in df.iterrows():
        cells = [fmt(v) for v in row.values]
        if index:
            cells = [fmt(idx)] + cells
        body_rows.append(cells)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body_rows)
    return "\n".join(lines)


def write_diagnostic_report(
    output_path: Path,
    *,
    title: str = "HB model -- validation diagnostics",
    metadata: dict[str, Any] | None = None,
    prior_pred: PriorPredictiveResult | None = None,
    posterior_pred: PosteriorPredictiveResult | None = None,
    contraction_df: pd.DataFrame | None = None,
    sensitivity: dict | None = None,
    contraction_plot: Path | None = None,
    sensitivity_plot: Path | None = None,
) -> Path:
    """Emit a single markdown file embedding all diagnostic results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"# {title}\n"]

    if metadata:
        for k, v in metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    section = 0

    if prior_pred is not None:
        section += 1
        lines.append(f"## {section}. Prior predictive check\n")
        s = prior_pred.summary
        lines.append(
            "Sampled observable values from the prior alone and compared "
            "to observed data.\n"
        )
        lines.append("| Statistic | Observed | Prior predictive |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Min / Max | {s['obs_min']:+.3f} / {s['obs_max']:+.3f} | -- |"
        )
        lines.append(
            f"| Quantile 0.01% / 99.99% | -- | "
            f"{s['sim_q0001']:+.3f} / {s['sim_q9999']:+.3f} |"
        )
        lines.append(
            f"| Quantile 2.5% / 97.5% | "
            f"{s['obs_q025']:+.3f} / {s['obs_q975']:+.3f} | "
            f"{s['sim_q025']:+.3f} / {s['sim_q975']:+.3f} |"
        )
        lines.append(f"| Std deviation | {s['obs_std']:.3f} | {s['sim_std']:.3f} |")
        lines.append("")
        lines.append(
            f"Fraction of prior-predictive draws within the observed "
            f"range: **{s['frac_sim_in_obs_range']:.1%}**.  "
        )
        lines.append(
            f"Fraction with |y| > 1: **{s['frac_sim_extreme']:.1%}**."
        )
        lines.append("")
        if prior_pred.figure_path is not None:
            rel = _relative_path(prior_pred.figure_path, output_path.parent)
            lines.append(f"![Prior predictive distribution]({rel})\n")

    if posterior_pred is not None:
        section += 1
        lines.append(f"## {section}. Posterior predictive check\n")
        lines.append(
            "Sampled observable values from the posterior and compared to "
            "observed data. Bayesian p-values close to 0.5 indicate the "
            "model captures the corresponding test statistic; values near "
            "0 or 1 indicate misfit.\n"
        )
        s = posterior_pred.summary
        lines.append("| Statistic | Observed | Posterior pred. mean | Bayesian p |")
        lines.append("|---|---|---|---|")
        for stat in ["mean", "std", "skew", "kurtosis", "min", "max", "q05", "q95"]:
            obs_val = s["obs_stats"][stat]
            rep_val = s["rep_stat_means"][stat]
            pval = s["bayesian_pvalues"][stat]
            lines.append(
                f"| {stat} | {obs_val:+.4f} | {rep_val:+.4f} | {pval:.3f} |"
            )
        lines.append("")
        lines.append(
            f"_Based on {s['n_replicates']} posterior predictive "
            f"replicates of {s['n_observed']} observations each._"
        )
        lines.append("")
        if posterior_pred.figure_path is not None:
            rel = _relative_path(posterior_pred.figure_path, output_path.parent)
            lines.append(f"![Posterior predictive distribution]({rel})\n")

    if contraction_df is not None and not contraction_df.empty:
        section += 1
        lines.append(f"## {section}. Prior-to-posterior contraction\n")
        lines.append(
            "Contraction = 1 - sigma^2_post / sigma^2_prior. Values near "
            "1 mean the data dominated; values near 0 mean the prior dominated.\n"
        )
        lines.append(
            _df_to_markdown(contraction_df, index=False, floatfmt=".4f")
        )
        lines.append("")
        if contraction_plot is not None:
            rel = _relative_path(contraction_plot, output_path.parent)
            lines.append(f"\n![Contraction]({rel})\n")

    if sensitivity is not None:
        section += 1
        lines.append(f"## {section}. Sensitivity to alternative priors\n")
        lines.append(
            "Comparison of posterior mean of the target quantity across "
            "prior variants.\n"
        )
        lines.append("### Pairwise summary (vs baseline)\n")
        lines.append(
            _df_to_markdown(sensitivity["summary"], index=False, floatfmt=".4f")
        )
        lines.append("\n### Rank correlations\n")
        lines.append(
            _df_to_markdown(sensitivity["rank_correlations"], floatfmt=".4f")
        )
        lines.append("")
        if sensitivity_plot is not None:
            rel = _relative_path(sensitivity_plot, output_path.parent)
            lines.append(f"\n![Sensitivity scatter]({rel})\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _relative_path(target: Path, base: Path) -> str:
    target, base = Path(target), Path(base)
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target)
