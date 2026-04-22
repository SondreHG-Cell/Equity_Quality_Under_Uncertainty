# uncertainty_model_hb.py

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from hb_shared_utils import (
    compute_wca,
    build_regressors,
    assign_indices,
    mark_usable,
    build_estimation_window,
    summarize_convergence,
    extract_sigma_posteriors,
    build_sigma_summary,
)

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)


# =============================================================================
# Bayesian accrual model
# =============================================================================

def build_hb_accrual_model_ar1(window_df: pd.DataFrame, firm_sector_map) -> tuple[pm.Model, dict]:
    """
    AR(1) extension: joint accrual model + CFO forecast model with
    latent CFO_{t+1} for portfolio-year rows.

    Returns
    -------
    model : pm.Model
    trace_info : dict with index mappings and sanity counts
    """
    wdf = window_df.copy()

    # --- Remap firm/sector indices contiguously ---
    window_firms = sorted(wdf["firm_idx"].unique())
    firm_remap = {old: new for new, old in enumerate(window_firms)}
    wdf["w_firm"] = wdf["firm_idx"].map(firm_remap)

    firm_to_sector = np.array([firm_sector_map[old] for old in window_firms])
    window_sectors = sorted(set(firm_to_sector))
    sector_remap = {old: new for new, old in enumerate(window_sectors)}
    firm_to_sector = np.array([sector_remap[s] for s in firm_to_sector], dtype=int)

    # Sector for each observation (via its firm)
    firm_idx = wdf["w_firm"].values.astype(int)
    sector_of_obs = firm_to_sector[firm_idx]

    # --- Extract arrays ---
    y = wdf["WCA_scaled"].values
    cfo_lag1 = wdf["CFO_lag1_scaled"].values
    cfo_curr = wdf["CFO_scaled"].values
    cfo_lead = wdf["CFO_lead1_scaled"].values
    drev = wdf["dREV_scaled"].values
    ppe = wdf["PPE_scaled"].values

    is_port = wdf["is_portfolio_year"].values

    # --- AR(1) training likelihood mask ---
    # Only consecutive-year transitions, and only on training rows.
    wdf_sorted = wdf.sort_values(["Ticker", "Year"]).copy()
    wdf_sorted["year_next_firm"] = wdf_sorted.groupby("Ticker")["Year"].shift(-1)
    wdf_sorted["consecutive"] = (
        wdf_sorted["CFO_lead1_scaled"].notna()
        & (wdf_sorted["year_next_firm"] == wdf_sorted["Year"] + 1)
    )
    wdf["consecutive"] = wdf_sorted.sort_index()["consecutive"].values
    ar1_obs_mask = (~is_port) & wdf["consecutive"].values

    ar1_sector_counts = pd.Series(sector_of_obs[ar1_obs_mask]).value_counts().sort_index()

    # --- Latent CFO_{t+1} indices: portfolio rows only ---
    latent_idx = np.where(is_port)[0]
    n_latent = len(latent_idx)

    # Placeholder overwritten inside model for latent rows
    cfo_lead_placeholder = np.where(np.isnan(cfo_lead), 0.0, cfo_lead)

    coords = {
        "firm": window_firms,
        "sector": window_sectors,
        "obs": np.arange(len(wdf)),
        "latent": np.arange(n_latent),
        "ar1_obs": np.arange(int(ar1_obs_mask.sum())),
    }

    with pm.Model(coords=coords) as model:
        # ═══ Accrual intercept and noise hierarchy ═══
        mu_0 = pm.Normal("mu_0", mu=0, sigma=0.1)
        omega = pm.HalfNormal("omega", sigma=0.05)
        tau = pm.HalfNormal("tau", sigma=0.05)
        sigma_0 = pm.HalfNormal("sigma_0", sigma=0.05)

        alpha_sector_raw = pm.Normal("alpha_sector_raw", mu=0, sigma=1, dims="sector")
        alpha_sector = pm.Deterministic(
            "alpha_sector",
            mu_0 + omega * alpha_sector_raw,
            dims="sector",
        )

        alpha_firm_raw = pm.Normal("alpha_firm_raw", mu=0, sigma=1, dims="firm")
        alpha_firm = pm.Deterministic(
            "alpha_firm",
            alpha_sector[firm_to_sector] + tau * alpha_firm_raw,
            dims="firm",
        )

        sigma_sector_raw = pm.HalfNormal("sigma_sector_raw", sigma=1, dims="sector")
        sigma_sector = pm.Deterministic(
            "sigma_sector",
            sigma_0 * sigma_sector_raw,
            dims="sector",
        )

        sigma_firm_raw = pm.HalfNormal("sigma_firm_raw", sigma=1, dims="firm")
        sigma_firm = pm.Deterministic(
            "sigma_firm",
            sigma_sector[firm_to_sector] * sigma_firm_raw,
            dims="firm",
        )

        # ═══ Global regression coefficients ═══
        b_lag = pm.Normal("beta_CFO_lag1", mu=0, sigma=0.3)
        b_cur = pm.Normal("beta_CFO_curr", mu=0, sigma=0.3)
        b_lead = pm.Normal("beta_CFO_lead1", mu=0, sigma=0.3)
        b_rev = pm.Normal("beta_dREV", mu=0, sigma=0.3)
        b_ppe = pm.Normal("beta_PPE", mu=0, sigma=0.3)

        # ═══ AR(1) CFO forecast — market level ═══
        mu_cfo_market = pm.Normal("mu_cfo_market", mu=0, sigma=0.2)
        rho_cfo_market = pm.Normal("rho_cfo_market", mu=0.5, sigma=0.3)
        psi_cfo_market = pm.HalfNormal("psi_cfo_market", sigma=0.2)

        # ═══ AR(1) sector-level spreads ═══
        sigma_mu_cfo = pm.HalfNormal("sigma_mu_cfo", sigma=0.1)
        sigma_rho_cfo = pm.HalfNormal("sigma_rho_cfo", sigma=0.1)

        mu_cfo_sector_raw = pm.Normal("mu_cfo_sector_raw", mu=0, sigma=1, dims="sector")
        rho_cfo_sector_raw = pm.Normal("rho_cfo_sector_raw", mu=0, sigma=1, dims="sector")
        psi_cfo_sector_raw = pm.HalfNormal("psi_cfo_sector_raw", sigma=1, dims="sector")

        mu_cfo_sector = pm.Deterministic(
            "mu_cfo_sector",
            mu_cfo_market + sigma_mu_cfo * mu_cfo_sector_raw,
            dims="sector",
        )
        rho_cfo_sector = pm.Deterministic(
            "rho_cfo_sector",
            rho_cfo_market + sigma_rho_cfo * rho_cfo_sector_raw,
            dims="sector",
        )
        psi_cfo_sector = pm.Deterministic(
            "psi_cfo_sector",
            psi_cfo_market * psi_cfo_sector_raw,
            dims="sector",
        )

        # ═══ AR(1) likelihood on observed transitions ═══
        cfo_t_for_ar1 = cfo_curr[ar1_obs_mask]
        cfo_next_for_ar1 = cfo_lead[ar1_obs_mask]
        sector_for_ar1 = sector_of_obs[ar1_obs_mask]

        mu_next = (
            mu_cfo_sector[sector_for_ar1]
            + rho_cfo_sector[sector_for_ar1] * cfo_t_for_ar1
        )
        sigma_next = psi_cfo_sector[sector_for_ar1]

        pm.Normal(
            "cfo_next_obs",
            mu=mu_next,
            sigma=sigma_next,
            observed=cfo_next_for_ar1,
            dims="ar1_obs",
        )

        # ═══ Latent CFO_{t+1} for portfolio rows ═══
        cfo_t_for_latent = cfo_curr[latent_idx]
        sector_for_latent = sector_of_obs[latent_idx]

        mu_latent = (
            mu_cfo_sector[sector_for_latent]
            + rho_cfo_sector[sector_for_latent] * cfo_t_for_latent
        )
        sigma_latent = psi_cfo_sector[sector_for_latent]

        cfo_lead_latent = pm.Normal(
            "cfo_lead_latent",
            mu=mu_latent,
            sigma=sigma_latent,
            dims="latent",
        )

        # ═══ Heavy tails for accrual likelihood ═══
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        # ═══ Splice observed + latent into full CFO_{t+1} vector ═══
        cfo_lead_full = pt.as_tensor_variable(cfo_lead_placeholder)
        cfo_lead_full = pt.set_subtensor(cfo_lead_full[latent_idx], cfo_lead_latent)

        # ═══ Accrual likelihood ═══
        mu_wca = (
            alpha_firm[firm_idx]
            + b_lag * cfo_lag1
            + b_cur * cfo_curr
            + b_lead * cfo_lead_full
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

    trace_info = {
        "window_firms": window_firms,
        "window_sectors": window_sectors,
        "firm_to_sector": firm_to_sector,
        "n_train": int((~is_port).sum()),
        "n_portfolio": int(is_port.sum()),
        "n_ar1_obs": int(ar1_obs_mask.sum()),
        "n_latent": n_latent,
        "ar1_counts_by_sector": ar1_sector_counts.to_dict(),
        "sector_remap": sector_remap,
    }

    return model, trace_info


# =============================================================================
# Diagnostics / plots
# =============================================================================

def _save_sigma_diagnostic_plots(
    sigma_summary: pd.DataFrame,
    data: pd.DataFrame,
    all_results: dict,
    firm_map: dict,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    sector_info = data[["Ticker", "Sector"]].drop_duplicates(subset="Ticker")
    sigma_with_sector = sigma_summary.merge(sector_info, on="Ticker", how="left")

    # 1. Distribution of sigma + posterior uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(
        sigma_summary["sigma_mean"],
        bins=60,
        edgecolor="white",
        alpha=0.85,
        color="steelblue",
    )
    axes[0].axvline(
        sigma_summary["sigma_mean"].median(),
        color="red",
        ls="--",
        label=f"Median: {sigma_summary['sigma_mean'].median():.4f}",
    )
    axes[0].set_xlabel("Posterior mean σ_i")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of firm-level accounting noise")
    axes[0].legend()

    sigma_summary = sigma_summary.copy()
    sigma_summary["ci_width"] = sigma_summary["sigma_q95"] - sigma_summary["sigma_q05"]

    axes[1].scatter(
        sigma_summary["sigma_mean"],
        sigma_summary["ci_width"],
        alpha=0.25,
        s=8,
        color="steelblue",
    )
    axes[1].set_xlabel("Posterior mean σ_i")
    axes[1].set_ylabel("90% credible interval width")
    axes[1].set_title("Posterior uncertainty vs noise level")

    plt.tight_layout()
    plt.savefig(plot_dir / "sigma_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. sigma by sector
    sector_order = (
        sigma_with_sector.groupby("Sector")["sigma_mean"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    box_data = [
        sigma_with_sector.loc[sigma_with_sector["Sector"] == s, "sigma_mean"]
        for s in sector_order
    ]
    bp = ax.boxplot(box_data, labels=sector_order, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    ax.set_ylabel("Posterior mean σ_i")
    ax.set_title("Accounting noise by sector (higher = more accrual uncertainty)")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "sigma_by_sector.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. sigma over time by sector
    fig, ax = plt.subplots(figsize=(11, 5.5))

    overall_median = sigma_summary.groupby("Year")["sigma_mean"].median()
    ax.plot(
        overall_median.index,
        overall_median.values,
        color="black",
        lw=2.5,
        label="Overall median",
        zorder=10,
    )

    for sector in sector_order:
        yearly = (
            sigma_with_sector.loc[sigma_with_sector["Sector"] == sector]
            .groupby("Year")["sigma_mean"]
            .median()
        )
        ax.plot(yearly.index, yearly.values, alpha=0.6, lw=1, label=sector)

    ax.set_xlabel("Portfolio year")
    ax.set_ylabel("Median σ_i")
    ax.set_title("Accounting noise over time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(plot_dir / "sigma_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Example firm posteriors
    if all_results:
        latest_year = max(all_results.keys())
        latest_results = all_results[latest_year]
        latest_sigma_means = {k: np.mean(v) for k, v in latest_results.items()}
        sorted_firms = sorted(latest_sigma_means.items(), key=lambda x: x[1])
        n = len(sorted_firms)

        if n >= 10:
            firm_map_rev = {v: k for k, v in firm_map.items()}
            example_firms = [sorted_firms[int(n * p)][0] for p in [0.1, 0.3, 0.5, 0.7, 0.9]]

            fig, ax = plt.subplots(figsize=(11, 5))
            for firm_idx in example_firms:
                draws = latest_results[firm_idx]
                ticker = firm_map_rev.get(firm_idx, f"firm_{firm_idx}")
                sec = sector_info.loc[sector_info["Ticker"] == ticker, "Sector"].values
                sec_label = sec[0] if len(sec) else "?"
                ax.hist(
                    draws,
                    bins=50,
                    alpha=0.4,
                    density=True,
                    label=f"{ticker} [{sec_label}] (σ̂={np.mean(draws):.3f})",
                )

            ax.set_xlabel("σ_i")
            ax.set_ylabel("Posterior density")
            ax.set_title(f"Example firm posteriors — portfolio year {latest_year}")
            ax.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(plot_dir / "sigma_posterior_examples.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def _save_last_ppc_plot(
    model,
    trace,
    window_df: pd.DataFrame,
    plot_dir: Path,
    random_seed: int = 42,
) -> str | None:
    if model is None or trace is None or window_df is None:
        return None

    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=random_seed)

    observed = window_df["WCA_scaled"].values
    simulated = ppc.posterior_predictive["WCA_obs"].values.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(-3, 3, 60)

    axes[0].hist(
        observed,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Observed",
        color="darkorange",
    )
    axes[0].hist(
        simulated,
        bins=bins,
        density=True,
        alpha=0.4,
        label="Posterior predictive",
        color="steelblue",
    )
    axes[0].set_xlabel("WCA_scaled")
    axes[0].set_title("Full distribution")
    axes[0].legend()

    tail_obs = observed[np.abs(observed) > 0.5]
    tail_sim = simulated[np.abs(simulated) > 0.5]
    bins_tail = np.linspace(-4, 4, 80)

    axes[1].hist(
        tail_obs,
        bins=bins_tail,
        density=True,
        alpha=0.6,
        label="Observed",
        color="darkorange",
    )
    axes[1].hist(
        tail_sim,
        bins=bins_tail,
        density=True,
        alpha=0.4,
        label="Posterior predictive",
        color="steelblue",
    )
    axes[1].set_xlabel("WCA_scaled")
    axes[1].set_title("Tail behaviour (|WCA| > 0.5)")
    axes[1].legend()

    plt.tight_layout()
    out_path = plot_dir / "posterior_predictive_check_last_year.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


# =============================================================================
# Main pipeline function
# =============================================================================

def run_uncertainty_model_hb(
    input_csv: str | Path,
    output_dir: str | Path,
    model_name: str = "ar1",
    year_start: int = 2009,
    year_end: int = 2025,
    n_draws: int = 2000,
    n_tune: int = 4000,
    n_chains: int = 4,
    target_accept: float = 0.95,
    min_train_years: int = 3,
    max_train_years: int = 5,
    random_seed: int = 42,
    save_full_posteriors: bool = True,
    save_plots: bool = True,
) -> dict:
    """
    Step 2 HB uncertainty model.

    Expected input_csv
    ------------------
    A prepared firm-year panel from the extraction step containing the columns
    needed by:
      - compute_wca(...)
      - build_regressors(..., include_lead=True)
      - mark_usable(...)
      - assign_indices(...)
      - build_estimation_window(...)

    Returns
    -------
    dict of saved output paths for run_main.py
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = output_dir / "plots"
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyMC {pm.__version__} | ArviZ {az.__version__} | Model: {model_name}")

    # --------------------------------------------------
    # Load prepared panel
    # --------------------------------------------------
    data = pd.read_csv(input_csv)

    data = compute_wca(data)
    data = build_regressors(data, include_lead=True)
    data = mark_usable(data)
    data, firm_map, sector_map, firm_sector = assign_indices(data)

    print(
        f"Panel: {len(data)} firm-years, {data['Ticker'].nunique()} firms, "
        f"{data['Year'].min()}–{data['Year'].max()}"
    )
    print(f"Usable: {data['usable'].sum()}")
    print(f"CFO_lead1 non-null: {data['CFO_lead1_scaled'].notna().sum()}")

    portfolio_years_to_run = sorted(
        y for y in data["Year"].unique() if year_start <= y <= year_end
    )

    all_results = {}
    last_model = None
    last_trace = None
    last_window_df = None

    for port_year in portfolio_years_to_run:
        checkpoint_path = checkpoint_dir / f"hb_checkpoint_{port_year}.pkl"

        if checkpoint_path.exists():
            print(f"Loading checkpoint for {port_year}")
            with open(checkpoint_path, "rb") as f:
                all_results[port_year] = pickle.load(f)[port_year]
            continue

        print(f"\n{'=' * 60}")
        print(f"Portfolio year: {port_year}")
        print(f"{'=' * 60}")

        window_df = build_estimation_window(
            data,
            port_year,
            min_train_years=min_train_years,
            max_train_years=max_train_years,
            include_portfolio_year=True,
        )

        if window_df is None:
            print("SKIPPED — insufficient training data")
            continue

        n_train = len(window_df)
        n_firms = window_df["Ticker"].nunique()
        train_years = sorted(int(y) for y in window_df["Year"].unique())

        print(f"Training years:  {train_years} ({n_train} obs)")
        print(f"Firms in window: {n_firms}")

        try:
            model, trace_info = build_hb_accrual_model_ar1(window_df, firm_sector)
        except Exception as e:
            print(f"ERROR building model: {e}")
            continue

        print(f"Model built: {trace_info['n_train']} train obs")

        if "ar1_counts_by_sector" in trace_info:
            sector_remap_inv = {v: k for k, v in trace_info["sector_remap"].items()}
            sector_map_inv = {v: k for k, v in sector_map.items()}
            for s_idx, count in trace_info["ar1_counts_by_sector"].items():
                orig_idx = sector_remap_inv[s_idx]
                s_name = sector_map_inv.get(orig_idx, f"sector_{orig_idx}")
                print(f"  AR(1) pairs — {s_name}: {count}")

        try:
            with model:
                trace = pm.sample(
                    draws=n_draws,
                    tune=n_tune,
                    chains=n_chains,
                    target_accept=target_accept,
                    random_seed=random_seed + int(port_year),
                    return_inferencedata=True,
                    progressbar=True,
                )
        except Exception as e:
            print(f"ERROR sampling: {e}")
            continue

        sigma_conv = summarize_convergence(trace, var_name="sigma_firm")
        alpha_conv = summarize_convergence(trace, var_name="alpha_firm")

        n_divergent = sigma_conv["n_divergent"]
        rhat_sigma = sigma_conv["max_rhat"]
        ess_sigma_bulk = sigma_conv["min_ess_bulk"]
        ess_sigma_tail = sigma_conv["min_ess_tail"]

        rhat_alpha = alpha_conv["max_rhat"]
        ess_alpha_bulk = alpha_conv["min_ess_bulk"]

        print(f"Divergences:          {n_divergent}")
        print(f"σ_firm  R̂ / ESS(b) / ESS(t):  {rhat_sigma:.3f} / {ess_sigma_bulk:.0f} / {ess_sigma_tail:.0f}")
        print(f"α_firm  R̂ / ESS(b):            {rhat_alpha:.3f} / {ess_alpha_bulk:.0f}")

        max_rhat = max(rhat_sigma, rhat_alpha)
        if n_divergent > 0:
            print(f"⚠ {n_divergent} divergences — consider raising target_accept")
        if max_rhat > 1.05:
            print("✗ R̂ > 1.05 — DO NOT USE these results, chains did not converge")
        elif max_rhat > 1.01:
            print("⚠ R̂ > 1.01 — investigate convergence before trusting")
        else:
            print("✓ Convergence good")
        if min(ess_sigma_bulk, ess_sigma_tail, ess_alpha_bulk) < 400:
            print("⚠ ESS < 400 for some parameter — credible intervals will be noisy")

        year_results = extract_sigma_posteriors(trace, trace_info)
        all_results[port_year] = year_results

        with open(checkpoint_path, "wb") as f:
            pickle.dump({port_year: year_results}, f)
        print(f"Checkpoint saved: {checkpoint_path.name}")

        last_model = model
        last_trace = trace
        last_window_df = window_df.copy()

    print(f"\nDone! Estimated {len(all_results)} portfolio years.")

    # --------------------------------------------------
    # Save consolidated outputs
    # --------------------------------------------------
    all_results_path = output_dir / "hb_all_results.pkl"
    with open(all_results_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Consolidated results saved to {all_results_path}")

    sigma_summary = build_sigma_summary(all_results, firm_map)
    sigma_summary_path = output_dir / "sigma_posteriors_summary.csv"
    sigma_summary.to_csv(sigma_summary_path, index=False)

    print(f"Saved summary: {sigma_summary_path}")
    if not sigma_summary.empty:
        print(
            f"{len(sigma_summary)} firm-year estimates, "
            f"{sigma_summary['Ticker'].nunique()} unique firms, "
            f"{sigma_summary['Year'].min()}–{sigma_summary['Year'].max()}"
        )
        print("\nPosterior mean σ_i distribution:")
        print(sigma_summary["sigma_mean"].describe().round(4).to_string())

    full_post_path = None
    if save_full_posteriors:
        firm_map_rev = {v: k for k, v in firm_map.items()}
        full_rows = []
        for port_year, year_results in sorted(all_results.items()):
            for firm_idx, draws in year_results.items():
                row = {"Year": port_year, "Ticker": firm_map_rev[firm_idx], "firm_idx": firm_idx}
                for i, d in enumerate(draws):
                    row[f"draw_{i}"] = d
                full_rows.append(row)

        sigma_full = pd.DataFrame(full_rows)
        full_post_path = output_dir / "sigma_posteriors_full.parquet"
        sigma_full.to_parquet(full_post_path, index=False)
        print(f"Saved full posteriors: {full_post_path}")
        print(f"Shape: {sigma_full.shape} ({sigma_full.shape[1] - 3} draws per firm-year)")

    # --------------------------------------------------
    # Merge summary back to firm-year panel for Step 3
    # --------------------------------------------------
    sigma_merged = data.merge(
        sigma_summary,
        on=["Ticker", "Year"],
        how="inner",
        validate="1:1",
    ).copy()

    # Standardized downstream sigma column
    sigma_merged["sigma_acc"] = sigma_merged["sigma_mean"]

    merged_output_path = output_dir / "uncertainty_firm_year.csv"
    sigma_merged.to_csv(merged_output_path, index=False)
    print(f"Saved merged firm-year output: {merged_output_path}")

    # --------------------------------------------------
    # Optional plots
    # --------------------------------------------------
    ppc_plot_path = None
    if save_plots and not sigma_summary.empty:
        _save_sigma_diagnostic_plots(
            sigma_summary=sigma_summary,
            data=data,
            all_results=all_results,
            firm_map=firm_map,
            plot_dir=plot_dir,
        )
        ppc_plot_path = _save_last_ppc_plot(
            model=last_model,
            trace=last_trace,
            window_df=last_window_df,
            plot_dir=plot_dir,
            random_seed=random_seed,
        )
        print(f"Diagnostic plots saved to {plot_dir}")

    # --------------------------------------------------
    # Save config
    # --------------------------------------------------
    config = {
        "model_name": model_name,
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "year_start": year_start,
        "year_end": year_end,
        "n_draws": n_draws,
        "n_tune": n_tune,
        "n_chains": n_chains,
        "target_accept": target_accept,
        "min_train_years": min_train_years,
        "max_train_years": max_train_years,
        "random_seed": random_seed,
        "save_full_posteriors": save_full_posteriors,
        "save_plots": save_plots,
        "n_portfolio_years_completed": len(all_results),
        "n_sigma_rows": int(len(sigma_summary)),
        "full_posteriors_parquet": str(full_post_path) if full_post_path is not None else None,
    }

    config_path = output_dir / "uncertainty_model_hb_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "firm_year_csv": str(merged_output_path),
        "sigma_summary_csv": str(sigma_summary_path),
        "all_results_pkl": str(all_results_path),
        "full_posterior_parquet": str(full_post_path) if full_post_path is not None else None,
        "config_json": str(config_path),
        "plots_dir": str(plot_dir) if save_plots else None,
        "ppc_plot_png": ppc_plot_path,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HB uncertainty model.")
    parser.add_argument("--input_csv", type=str, required=True, help="Prepared firm-year panel for Step 2.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HB outputs.")
    parser.add_argument("--model_name", type=str, default="ar1")
    parser.add_argument("--year_start", type=int, default=2009)
    parser.add_argument("--year_end", type=int, default=2025)
    parser.add_argument("--n_draws", type=int, default=2000)
    parser.add_argument("--n_tune", type=int, default=4000)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.95)
    parser.add_argument("--min_train_years", type=int, default=3)
    parser.add_argument("--max_train_years", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--no_full_posteriors", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_uncertainty_model_hb(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        year_start=args.year_start,
        year_end=args.year_end,
        n_draws=args.n_draws,
        n_tune=args.n_tune,
        n_chains=args.n_chains,
        target_accept=args.target_accept,
        min_train_years=args.min_train_years,
        max_train_years=args.max_train_years,
        random_seed=args.random_seed,
        save_full_posteriors=not args.no_full_posteriors,
        save_plots=not args.no_plots,
    )