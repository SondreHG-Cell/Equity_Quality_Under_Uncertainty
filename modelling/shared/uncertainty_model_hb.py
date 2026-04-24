# uncertainty_model_hb.py
# Two-stage version:
# 1) fit CFO_{t+1} AR(1) model outside the accrual model, using training rows only
# 2) predict CFO_{t+1} for portfolio-year rows
# 3) fit accrual HB model with predicted CFO_{t+1} fixed in the WCA equation

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
# Stage 1: external CFO_{t+1} forecast model
# =============================================================================

def fit_external_cfo_ar1_model(
    window_df: pd.DataFrame,
    random_seed: int,
    draws: int = 1000,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.95,
) -> tuple[dict, az.InferenceData]:
    """
    Fit the external Student-t AR(1) CFO model on observed training transitions only.
    """
    wdf = window_df.copy()

    # --- Remap firm indices contiguously ---
    window_firms = sorted(wdf["firm_idx"].unique())
    firm_remap = {old: new for new, old in enumerate(window_firms)}
    wdf["w_firm"] = wdf["firm_idx"].map(firm_remap)

    # --- Build sector map for window firms ---
    firm_sector_map = (
        wdf.groupby("w_firm")["sector_idx"]
        .first()
        .astype(int)
        .sort_index()
    )
    firm_to_sector_orig = np.array([firm_sector_map[i] for i in range(len(window_firms))], dtype=int)

    window_sectors = sorted(set(firm_to_sector_orig))
    sector_remap = {old: new for new, old in enumerate(window_sectors)}
    firm_to_sector = np.array([sector_remap[s] for s in firm_to_sector_orig], dtype=int)

    # sector for each row
    firm_idx = wdf["w_firm"].values.astype(int)
    sector_of_obs = firm_to_sector[firm_idx]

    cfo_curr = wdf["CFO_scaled"].values
    cfo_lead = wdf["CFO_lead1_scaled"].values
    is_port = wdf["is_portfolio_year"].values

    # observed consecutive transitions for training only
    wdf_sorted = wdf.sort_values(["Ticker", "Year"]).copy()
    wdf_sorted["year_next_firm"] = wdf_sorted.groupby("Ticker")["Year"].shift(-1)
    wdf_sorted["consecutive"] = (
        wdf_sorted["CFO_lead1_scaled"].notna()
        & (wdf_sorted["year_next_firm"] == wdf_sorted["Year"] + 1)
    )
    wdf["consecutive"] = wdf_sorted.sort_index()["consecutive"].values
    ar1_obs_mask = (~is_port) & wdf["consecutive"].values

    # predict for portfolio-year rows and any row with missing lead CFO
    predict_idx = np.where(is_port | np.isnan(cfo_lead))[0]

    cfo_t_for_ar1 = cfo_curr[ar1_obs_mask]
    cfo_next_for_ar1 = cfo_lead[ar1_obs_mask]
    sector_for_ar1 = sector_of_obs[ar1_obs_mask]

    coords = {
        "sector": window_sectors,
        "ar1_obs": np.arange(int(ar1_obs_mask.sum())),
        "predict_obs": np.arange(len(predict_idx)),
    }

    with pm.Model(coords=coords) as model:
        mu_cfo_market = pm.Normal("mu_cfo_market", mu=0, sigma=0.2)
        rho_cfo_market = pm.Normal("rho_cfo_market", mu=0.5, sigma=0.3)
        psi_cfo_market = pm.HalfNormal("psi_cfo_market", sigma=0.2)

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

        nu_cfo = pm.Gamma("nu_cfo", alpha=2, beta=0.1)

        mu_next = (
            mu_cfo_sector[sector_for_ar1]
            + rho_cfo_sector[sector_for_ar1] * cfo_t_for_ar1
        )
        sigma_next = psi_cfo_sector[sector_for_ar1]

        pm.StudentT(
            "cfo_next_obs",
            nu=nu_cfo,
            mu=mu_next,
            sigma=sigma_next,
            observed=cfo_next_for_ar1,
            dims="ar1_obs",
        )

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    cfo_info = {
        "window_firms": window_firms,
        "window_sectors": window_sectors,
        "firm_to_sector": firm_to_sector,
        "predict_idx": predict_idx,
        "sector_of_obs": sector_of_obs,
        "cfo_curr": cfo_curr,
        "ar1_obs_mask": ar1_obs_mask,
        "n_ar1_obs": int(ar1_obs_mask.sum()),
        "n_predict": int(len(predict_idx)),
        "sector_remap": sector_remap,
    }
    return cfo_info, trace


def predict_cfo_lead_for_portfolio_rows(
    window_df: pd.DataFrame,
    cfo_info: dict,
    cfo_trace: az.InferenceData,
    prediction_mode: str = "mean",
) -> pd.DataFrame:
    """
    Predict CFO_{t+1} using the external Student-t CFO model.

    prediction_mode
    ---------------
    "mean"  : posterior mean of the conditional location
    "draw"  : one posterior predictive Student-t draw per row
    """
    wdf = window_df.copy()

    predict_idx = cfo_info["predict_idx"]
    sector_of_obs = cfo_info["sector_of_obs"]
    cfo_curr = cfo_info["cfo_curr"]

    if len(predict_idx) == 0:
        wdf["CFO_lead1_pred_scaled"] = wdf["CFO_lead1_scaled"]
        return wdf

    sector_for_pred = sector_of_obs[predict_idx]
    cfo_t_for_pred = cfo_curr[predict_idx]

    mu_cfo_sector = cfo_trace.posterior["mu_cfo_sector"].values.reshape(
        -1, cfo_trace.posterior["mu_cfo_sector"].shape[-1]
    )
    rho_cfo_sector = cfo_trace.posterior["rho_cfo_sector"].values.reshape(
        -1, cfo_trace.posterior["rho_cfo_sector"].shape[-1]
    )
    psi_cfo_sector = cfo_trace.posterior["psi_cfo_sector"].values.reshape(
        -1, cfo_trace.posterior["psi_cfo_sector"].shape[-1]
    )
    nu_cfo = cfo_trace.posterior["nu_cfo"].values.reshape(-1)

    n_post = mu_cfo_sector.shape[0]
    pred = np.zeros(len(predict_idx), dtype=float)

    if prediction_mode == "mean":
        for j in range(len(predict_idx)):
            s = sector_for_pred[j]
            mu_draw = mu_cfo_sector[:, s] + rho_cfo_sector[:, s] * cfo_t_for_pred[j]
            pred[j] = mu_draw.mean()

    elif prediction_mode == "draw":
        draw_ids = np.random.randint(0, n_post, size=len(predict_idx))
        for j in range(len(predict_idx)):
            s = sector_for_pred[j]
            d = draw_ids[j]
            mu_draw = mu_cfo_sector[d, s] + rho_cfo_sector[d, s] * cfo_t_for_pred[j]
            sigma_draw = psi_cfo_sector[d, s]
            nu_draw = nu_cfo[d]
            pred[j] = mu_draw + sigma_draw * np.random.standard_t(df=nu_draw)

    else:
        raise ValueError("prediction_mode must be 'mean' or 'draw'")

    wdf["CFO_lead1_pred_scaled"] = wdf["CFO_lead1_scaled"]
    wdf.loc[wdf.index[predict_idx], "CFO_lead1_pred_scaled"] = pred

    return wdf


# =============================================================================
# Stage 2: accrual model with fixed predicted CFO_{t+1}
# =============================================================================

def build_hb_accrual_model_fixed_lead(
    window_df: pd.DataFrame,
    firm_sector_map,
    include_cfo_lead: bool = True,
) -> tuple[pm.Model, dict]:
    """
    Accrual model only.

    include_cfo_lead=True:
        use fixed externally predicted CFO_{t+1} in the WCA equation

    include_cfo_lead=False:
        drop CFO_{t+1} entirely from the WCA equation
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

    firm_idx = wdf["w_firm"].values.astype(int)

    y = wdf["WCA_scaled"].values
    cfo_lag1 = wdf["CFO_lag1_scaled"].values
    cfo_curr = wdf["CFO_scaled"].values
    drev = wdf["dREV_scaled"].values
    ppe = wdf["PPE_scaled"].values

    if include_cfo_lead:
        if "CFO_lead1_pred_scaled" not in wdf.columns:
            raise ValueError("CFO_lead1_pred_scaled is required when include_cfo_lead=True")
        cfo_lead_fixed = wdf["CFO_lead1_pred_scaled"].values

    coords = {
        "firm": window_firms,
        "sector": window_sectors,
        "obs": np.arange(len(wdf)),
    }

    with pm.Model(coords=coords) as model:
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

        b_lag = pm.Normal("beta_CFO_lag1", mu=0, sigma=0.3)
        b_cur = pm.Normal("beta_CFO_curr", mu=0, sigma=0.3)
        b_rev = pm.Normal("beta_dREV", mu=0, sigma=0.3)
        b_ppe = pm.Normal("beta_PPE", mu=0, sigma=0.3)

        if include_cfo_lead:
            b_lead = pm.Normal("beta_CFO_lead1", mu=0, sigma=0.3)

        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        mu_wca = (
            alpha_firm[firm_idx]
            + b_lag * cfo_lag1
            + b_cur * cfo_curr
            + b_rev * drev
            + b_ppe * ppe
        )

        if include_cfo_lead:
            mu_wca = mu_wca + b_lead * cfo_lead_fixed

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
        "n_obs": int(len(wdf)),
        "sector_remap": sector_remap,
        "include_cfo_lead": include_cfo_lead,
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
    model_name: str = "two_stage_ar1",
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
    cfo_draws: int = 1000,
    cfo_tune: int = 1500,
    cfo_prediction_mode: str = "mean",
    cfo_lead_mode: str = "best_external"
) -> dict:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = output_dir / "plots"
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyMC {pm.__version__} | ArviZ {az.__version__} | Model: {model_name}")
    
    cfo_lead_mode = cfo_lead_mode.lower()
    if cfo_lead_mode not in {"best_external", "none"}:
        raise ValueError("cfo_lead_mode must be one of: 'best_external', 'none'")

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

        n_obs = len(window_df)
        n_firms = window_df["Ticker"].nunique()
        years_in_window = sorted(int(y) for y in window_df["Year"].unique())

        print(f"Window years:    {years_in_window} ({n_obs} obs)")
        print(f"Firms in window: {n_firms}")

        # --------------------------------------------------
        # Stage 1: CFO_{t+1} handling
        # --------------------------------------------------
        if cfo_lead_mode == "best_external":
            try:
                cfo_info, cfo_trace = fit_external_cfo_ar1_model(
                    window_df=window_df,
                    random_seed=random_seed + 10_000 + int(port_year),
                    draws=cfo_draws,
                    tune=cfo_tune,
                    chains=n_chains,
                    target_accept=target_accept,
                )
            except Exception as e:
                print(f"ERROR fitting external CFO model: {e}")
                continue

            print(
                f"External CFO model fitted: "
                f"{cfo_info['n_ar1_obs']} observed transitions, "
                f"{cfo_info['n_predict']} rows predicted"
            )

            try:
                window_df_fixed = predict_cfo_lead_for_portfolio_rows(
                    window_df=window_df,
                    cfo_info=cfo_info,
                    cfo_trace=cfo_trace,
                    prediction_mode=cfo_prediction_mode,
                )
            except Exception as e:
                print(f"ERROR predicting CFO_t+1 externally: {e}")
                continue

            include_cfo_lead = True

        elif cfo_lead_mode == "none":
            print("Skipping CFO_{t+1}: cfo_lead_mode='none'")
            window_df_fixed = window_df.copy()
            include_cfo_lead = False

        else:
            raise ValueError(f"Unknown cfo_lead_mode: {cfo_lead_mode}")

        # --------------------------------------------------
        # Stage 2: accrual model
        # --------------------------------------------------
        try:
            model, trace_info = build_hb_accrual_model_fixed_lead(
                window_df_fixed,
                firm_sector=firm_sector,
                include_cfo_lead=include_cfo_lead,
            )
        except Exception as e:
            print(f"ERROR building accrual model: {e}")
            continue

        print(f"Accrual model built: {trace_info['n_obs']} obs")

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
            print(f"ERROR sampling accrual model: {e}")
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
        last_window_df = window_df_fixed.copy()

    print(f"\nDone! Estimated {len(all_results)} portfolio years.")

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

    sigma_merged = data.merge(
        sigma_summary,
        on=["Ticker", "Year"],
        how="inner",
        validate="1:1",
    ).copy()

    sigma_merged["sigma_acc"] = sigma_merged["sigma_mean"]

    merged_output_path = output_dir / "uncertainty_firm_year.csv"
    sigma_merged.to_csv(merged_output_path, index=False)
    print(f"Saved merged firm-year output: {merged_output_path}")

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
        "cfo_draws": cfo_draws,
        "cfo_tune": cfo_tune,
        "cfo_prediction_mode": cfo_prediction_mode,
        "cfo_lead_mode": cfo_lead_mode,
        "n_portfolio_years_completed": len(all_results),
        "n_sigma_rows": int(len(sigma_summary)),
        "full_posterior_parquet": str(full_post_path) if full_post_path is not None else None,
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
    parser = argparse.ArgumentParser(description="Run two-stage HB uncertainty model.")
    parser.add_argument("--input_csv", type=str, required=True, help="Prepared firm-year panel for Step 2.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HB outputs.")
    parser.add_argument("--model_name", type=str, default="two_stage_ar1")
    parser.add_argument("--year_start", type=int, default=2009)
    parser.add_argument("--year_end", type=int, default=2025)
    parser.add_argument("--n_draws", type=int, default=2000)
    parser.add_argument("--n_tune", type=int, default=4000)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.95)
    parser.add_argument("--min_train_years", type=int, default=3)
    parser.add_argument("--max_train_years", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--cfo_draws", type=int, default=1000)
    parser.add_argument("--cfo_tune", type=int, default=1500)
    parser.add_argument("--cfo_prediction_mode", type=str, default="mean", choices=["mean", "draw"])
    parser.add_argument(
        "--cfo_lead_mode",
        type=str,
        default="best_external",
        choices=["best_external", "none"],
        help="How to handle CFO_{t+1} in the accrual model.",
    )
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
        cfo_draws=args.cfo_draws,
        cfo_tune=args.cfo_tune,
        cfo_prediction_mode=args.cfo_prediction_mode,
        cfo_lead_mode=args.cfo_lead_mode,
    )