"""
Shared utilities for Step 2 hierarchical Bayesian accrual models.

Imported by:
  - hb_accruals_baseline.ipynb   (Model 1)
  - hb_accruals_ar1.ipynb        (Model 2)
  - hb_accruals_comparison.ipynb (post-hoc comparison)

Contains: data loading/merging, WCA computation, regressor construction,
per-year winsorization, index assignment, estimation window builder,
and diagnostic helpers.

Does NOT contain: model definitions (those live in each notebook,
because the model is the only thing that differs between specifications).
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================
WINSOR_COLS = ["WCA_scaled", "CFO_lag1_scaled", "CFO_scaled",
               "dREV_scaled", "PPE_scaled"]
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99

BASE_COLS_USABLE = ["WCA_scaled", "CFO_lag1_scaled", "CFO_scaled",
                    "dREV_scaled", "PPE_scaled", "AvgAT"]

# ============================================================
# DATA LOADING
# ============================================================
def load_and_merge(acc_dir: Path, prof_dir: Path,
                   year_min: int = 2005, year_max: int = 2025) -> pd.DataFrame:
    """Load per-firm CSVs from both source folders and merge on
    filename + Year + Ticker. Returns a single panel DataFrame."""
    acc_frames = []
    for fpath in sorted(glob.glob(os.path.join(acc_dir, "*.csv"))):
        df = pd.read_csv(fpath)
        df["_source_file"] = os.path.basename(fpath)
        acc_frames.append(df)
    acc_raw = pd.concat(acc_frames, ignore_index=True)

    prof_frames = []
    for fpath in sorted(glob.glob(os.path.join(prof_dir, "*.csv"))):
        df = pd.read_csv(fpath)
        df["_source_file"] = os.path.basename(fpath)
        cols = [c for c in ["Year", "Ticker", "REVT", "_source_file"] if c in df.columns]
        prof_frames.append(df[cols])
    prof_raw = pd.concat(prof_frames, ignore_index=True)

    data = acc_raw.merge(prof_raw, on=["_source_file", "Year", "Ticker"], how="left")
    data = data[(data["Year"] >= year_min) & (data["Year"] <= year_max)]
    return data.reset_index(drop=True)

# ============================================================
# GAP INTERPOLATION
# ============================================================
def _interpolate_single_year_gaps(
    series: pd.Series,
    max_rel_change: float = 5.0,
    col_name: str = "",
) -> tuple[pd.Series, int, int]:
    """
    Fill single-year interior gaps in a monotonically-indexed series using
    the mean of the adjacent values. Does NOT interpolate:
      - multi-year gaps (2+ consecutive NaN)
      - leading/trailing NaN (edges)
      - gaps where the two neighbours differ by more than max_rel_change
        (signals structural break, not a reporting gap)

    Returns (filled_series, n_filled, n_skipped_large_change).
    """
    out = series.copy()
    is_nan = out.isna().values
    vals = out.values
    n = len(vals)
    n_filled = 0
    n_skipped = 0

    for i in range(1, n - 1):
        if not is_nan[i]:
            continue
        # Neighbours must be observed
        if is_nan[i - 1] or is_nan[i + 1]:
            continue
        prev_val, next_val = vals[i - 1], vals[i + 1]
        # Sanity bound: don't interpolate across a huge change
        denom = max(abs(prev_val), abs(next_val), 1e-9)
        rel_change = abs(next_val - prev_val) / denom
        if rel_change > max_rel_change:
            n_skipped += 1
            continue
        out.iloc[i] = (prev_val + next_val) / 2
        n_filled += 1

    return out, n_filled, n_skipped

# ============================================================
# WCA AND REGRESSOR CONSTRUCTION
# ============================================================
def compute_wca(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute working capital accruals and scaled version.

    Pre-processing steps before differencing:
      1. Drop leading firm-year rows where CFO (OANCF) is missing — the
         firm's reporting history effectively begins once cash-flow data
         is available.
      2. Interpolate single-year interior gaps in PPEGT using the mean of
         adjacent values (with a sanity bound on relative change).

    Adds: d_ACT, d_CHE, d_LCT, d_STD, d_TXP, WCA, AT_lag, AvgAT, WCA_scaled.
    """
    data = data.sort_values(["Ticker", "Year"]).reset_index(drop=True)

    # --- Step 1: drop leading CFO-missing rows per firm ---------------
    # CFO is required as regressor, lag, and lead; rows before the first
    # observed CFO contribute no usable information and their inclusion
    # risks creating spurious "first differences" for d_ACT etc.
    n_before = len(data)
    data["_has_cfo"] = data["OANCF"].notna()
    data["_cum_cfo"] = data.groupby("Ticker")["_has_cfo"].cumsum()
    data = data[data["_cum_cfo"] > 0].copy()
    data = data.drop(columns=["_has_cfo", "_cum_cfo"])
    n_dropped = n_before - len(data)
    if verbose and n_dropped > 0:
        print(f"[compute_wca] Dropped {n_dropped} leading rows "
              f"with missing CFO across {data['Ticker'].nunique()} firms")

    # --- Step 2: interpolate single-year interior PPEGT gaps ----------
    total_filled = 0
    total_skipped = 0
    n_firms_touched = 0
    for tkr, grp in data.groupby("Ticker"):
        filled, n_fill, n_skip = _interpolate_single_year_gaps(
            grp["PPEGT"], max_rel_change=5.0, col_name="PPEGT"
        )
        if n_fill > 0:
            data.loc[grp.index, "PPEGT"] = filled.values
            n_firms_touched += 1
        total_filled += n_fill
        total_skipped += n_skip
    if verbose and (total_filled > 0 or total_skipped > 0):
        print(f"[compute_wca] PPEGT: interpolated {total_filled} interior "
              f"gaps across {n_firms_touched} firms; "
              f"skipped {total_skipped} gaps with >500% neighbour change "
              f"(likely structural breaks)")

    # --- Economic-zero fills for short-term debt and taxes payable ----
    data["STD"] = data["STD"].fillna(0)
    data["TXP"] = data["TXP"].fillna(0)

    # --- Year-over-year changes within firm ---------------------------
    for col in ["ACT", "CHE", "LCT", "STD", "TXP"]:
        data[f"d_{col}"] = data.groupby("Ticker")[col].diff()

    # --- Working capital accruals -------------------------------------
    data["WCA"] = (
        (data["d_ACT"] - data["d_CHE"])
        - (data["d_LCT"] - data["d_STD"] - data["d_TXP"])
    )

    data["AT_lag"] = data.groupby("Ticker")["AT"].shift(1)
    data["AvgAT"]  = (data["AT"] + data["AT_lag"]) / 2
    data["WCA_scaled"] = data["WCA"] / data["AvgAT"]

    return data

def build_regressors(data: pd.DataFrame, include_lead: bool = False) -> pd.DataFrame:
    """Construct McNichols regressors, scaled by average assets.

    Parameters
    ----------
    data : DataFrame
        Output of compute_wca().
    include_lead : bool, default False
        If True, also construct CFO_lead1 and CFO_lead1_scaled (for AR(1) model).
    """
    data["CFO"] = data["OANCF"]
    data["CFO_lag1"] = data.groupby("Ticker")["CFO"].shift(1)
    data["dREV"] = data.groupby("Ticker")["REVT"].diff()

    data["CFO_lag1_scaled"] = data["CFO_lag1"] / data["AvgAT"]
    data["CFO_scaled"]      = data["CFO"]      / data["AvgAT"]
    data["dREV_scaled"]     = data["dREV"]     / data["AvgAT"]
    data["PPE_scaled"]      = data["PPEGT"]    / data["AvgAT"]

    if include_lead:
        data["CFO_lead1"] = data.groupby("Ticker")["CFO"].shift(-1)
        data["CFO_lead1_scaled"] = data["CFO_lead1"] / data["AvgAT"]

    return data

# ============================================================
# PER-YEAR WINSORIZATION
# ============================================================
def winsorize_by_year(df: pd.DataFrame, cols: list[str],
                      low_q: float = WINSOR_LOW,
                      high_q: float = WINSOR_HIGH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip each column to its [low_q, high_q] quantiles within each Year.

    Returns (clipped_df, bounds_diag_df) where bounds_diag_df has
    columns [Year, variable, low, high, n_nonnull].
    """
    bounds_records = []
    out = df.copy()
    for col in cols:
        bounds = (out.groupby("Year")[col].quantile([low_q, high_q])
                     .unstack().rename(columns={low_q: "low", high_q: "high"}))
        low_map  = out["Year"].map(bounds["low"])
        high_map = out["Year"].map(bounds["high"])
        out[col] = out[col].clip(lower=low_map, upper=high_map)

        b = bounds.copy()
        b["variable"] = col
        b["n_nonnull"] = out.groupby("Year")[col].count()
        bounds_records.append(b.reset_index())
    diag = pd.concat(bounds_records, ignore_index=True)
    return out, diag

# ============================================================
# INDEX ASSIGNMENT
# ============================================================
def assign_indices(data: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, pd.Series]:
    """Assign integer indices for firm and sector. Returns updated data,
    firm_map (ticker -> int), sector_map (name -> int), firm_sector (Series)."""
    sector_map = {s: i for i, s in enumerate(sorted(data["Sector"].dropna().unique()))}
    data["sector_idx"] = data["Sector"].map(sector_map)

    firm_map = {t: i for i, t in enumerate(sorted(data["Ticker"].unique()))}
    data["firm_idx"] = data["Ticker"].map(firm_map)

    firm_sector = (
        data.dropna(subset=["sector_idx"])
            .groupby("firm_idx")["sector_idx"].first().astype(int)
            .sort_index()
    )

    return data, firm_map, sector_map, firm_sector

def mark_usable(data: pd.DataFrame,
                base_cols: list[str] = None) -> pd.DataFrame:
    """Mark rows as usable (all required columns non-null).
    NOTE: CFO_lead1 is NOT required here — AR(1) model handles it as latent."""
    cols = base_cols or BASE_COLS_USABLE
    data["usable"] = data[cols].notna().all(axis=1)
    return data

# ============================================================
# ESTIMATION WINDOW
# ============================================================
def build_estimation_window(data: pd.DataFrame, portfolio_year: int,
                            min_train_years: int = 3,
                            max_train_years: int = 5,
                            min_firm_obs: int = 3,
                            include_portfolio_year: bool = False,
                            verbose: bool = True) -> pd.DataFrame | None:
    """Build the estimation sample for one portfolio year.

    Parameters
    ----------
    include_portfolio_year : bool
        If False (default, baseline model): return ONLY training rows
            (year in [t-max_train_years, t-1]). Use this for models that
            do not use portfolio-year data in the likelihood.
        If True (AR(1) model): also include portfolio-year (year t) rows,
            tagged with is_portfolio_year=True. These rows feed the
            accrual likelihood via a latent CFO_{t+1}.
    """
    train_start = portfolio_year - max_train_years
    train_end   = portfolio_year - 1

    train_mask = ((data["Year"] >= train_start) &
                  (data["Year"] <= train_end) &
                  data["usable"])
    train_df = data.loc[train_mask].copy()

    if len(train_df) == 0 or train_df["Year"].nunique() < min_train_years:
        return None

    # Filter firms with insufficient training observations
    firm_obs_counts = train_df.groupby("Ticker").size()
    firms_ok = firm_obs_counts[firm_obs_counts >= min_firm_obs].index

    n_before = train_df["Ticker"].nunique()
    train_df = train_df[train_df["Ticker"].isin(firms_ok)]
    n_after = train_df["Ticker"].nunique()

    if verbose and n_before > n_after:
        print(f"  Firms removed (<{min_firm_obs} training obs): "
              f"{n_before - n_after} / {n_before}")

    if len(train_df) == 0:
        return None

    train_df["is_portfolio_year"] = False

    if not include_portfolio_year:
        return train_df.reset_index(drop=True)

    port_mask = ((data["Year"] == portfolio_year) & data["usable"])
    port_df = data.loc[port_mask].copy()
    port_df = port_df[port_df["Ticker"].isin(firms_ok)]
    port_df["is_portfolio_year"] = True

    return pd.concat([train_df, port_df], ignore_index=True)

# ============================================================
# DIAGNOSTICS HELPERS
# ============================================================
def summarize_convergence(trace, var_name: str = "sigma_firm") -> dict:
    """Compute R-hat, bulk ESS, tail ESS, and divergence count."""
    import arviz as az
    n_divergent = int(trace.sample_stats["diverging"].values.sum())
    rhat = az.rhat(trace, var_names=[var_name])[var_name]
    ess_bulk = az.ess(trace, var_names=[var_name], method="bulk")[var_name]
    ess_tail = az.ess(trace, var_names=[var_name], method="tail")[var_name]
    return {
        "n_divergent":   n_divergent,
        "max_rhat":      float(rhat.max()),
        "min_ess_bulk":  float(ess_bulk.min()),
        "min_ess_tail":  float(ess_tail.min()),
    }

def extract_sigma_posteriors(trace, trace_info: dict) -> dict[int, np.ndarray]:
    """Flatten sigma_firm posterior across chains/draws,
    return {orig_firm_idx: 1D array of draws}."""
    sigma_samples = trace.posterior["sigma_firm"].values
    sigma_samples = sigma_samples.reshape(-1, sigma_samples.shape[-1])
    return {
        orig: sigma_samples[:, w]
        for w, orig in enumerate(trace_info["window_firms"])
    }

def build_sigma_summary(all_results: dict, firm_map: dict) -> pd.DataFrame:
    """Convert {port_year: {firm_idx: draws}} into a tidy summary table."""
    firm_map_rev = {v: k for k, v in firm_map.items()}
    rows = []
    for port_year, year_results in sorted(all_results.items()):
        for firm_idx, draws in year_results.items():
            rows.append({
                "Year": port_year,
                "Ticker": firm_map_rev.get(firm_idx, f"firm_{firm_idx}"),
                "firm_idx": firm_idx,
                "sigma_mean":   np.mean(draws),
                "sigma_median": np.median(draws),
                "sigma_std":    np.std(draws),
                "sigma_q05":    np.percentile(draws, 5),
                "sigma_q95":    np.percentile(draws, 95),
                "n_draws":      len(draws),
            })
    return pd.DataFrame(rows)

