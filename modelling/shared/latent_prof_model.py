# latent_prof_model.py
#
# Produces three output directories in one run:
#   <output_dir>/HB/       — standard EB shrinkage toward mean  (original behaviour)
#   <output_dir>/HB_cap/   — shrinkage capped at theta_obs      (never lifts signal)
#   <output_dir>/HB_down/  — always-downward shrinkage          (always penalises)

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


# --------------------------------------------------
# Defaults
# --------------------------------------------------

DEFAULT_FORMATION_YEAR_MIN = 2010
DEFAULT_FORMATION_YEAR_MAX = 2025

DEFAULT_UNCERTAINTY_METHOD = "auto"   # "auto", "OLS", "HB"
DEFAULT_SIGMA_COL = None

DEFAULT_NOISE_SHARE_OF_PROF_VAR = 0.12

DEFAULT_WINSORIZE_PROF = False
DEFAULT_WINSORIZE_SIGMA = False
DEFAULT_WINSOR_LOWER = 0.01
DEFAULT_WINSOR_UPPER = 0.99

DEFAULT_MIN_FIRMS_PER_YEAR = 20
DEFAULT_MIN_TAU2 = 1e-8
DEFAULT_MIN_POST_VAR = 1e-12

DEFAULT_USE_FULL_PROPAGATION = False
DEFAULT_HB_FULL_POSTERIOR_PARQUET = None
DEFAULT_N_SIGMA_DRAWS = None
DEFAULT_CHECKPOINT_EVERY_DRAWS = 25

# Variants produced in a single run — name: shrinkage_method
VARIANTS = {
    "HB":      None,          # standard — identical to original code
    "HB_cap":  "cap",         # shrinkage capped at theta_obs
    "HB_down": "down_only",   # always-downward shrinkage
}


# --------------------------------------------------
# Candidate sigma columns from Step 2
# --------------------------------------------------

OLS_SIGMA_CANDIDATES = [
    "sigma_acc", "sigma_ols", "sigma_hat", "rmse", "rmse_acc", "sigma_acc_abs",
]

HB_SIGMA_CANDIDATES = [
    "sigma_post_mean", "sigma_acc_post_mean", "sigma_hb", "sigma_hb_mean",
    "posterior_mean_sigma", "sigma_mean", "sigma_acc",
]

REQUIRED_BASE_COLUMNS = ["Ticker", "Year", "PROF", "MarketCap"]


# --------------------------------------------------
# Shrinkage logic
# --------------------------------------------------

def apply_shrinkage(
    theta_obs: np.ndarray,
    lambda_i: np.ndarray,
    mu_t: float,
    shrinkage_method: Optional[str],
) -> np.ndarray:
    """
    Compute theta_post_mean with an optional shrinkage adjustment.

    None (HB)        — standard EB: lambda*theta_obs + (1-lambda)*mu_t
    "cap" (HB_cap)   — standard EB capped at theta_obs; never lifts a signal above observed
    "down_only" (HB_down) — theta_obs - (1-lambda)*|theta_obs - mu_t|; always downward
    """
    baseline = lambda_i * theta_obs + (1.0 - lambda_i) * mu_t

    if shrinkage_method is None:
        return baseline
    elif shrinkage_method == "cap":
        return np.minimum(baseline, theta_obs)
    elif shrinkage_method == "down_only":
        return theta_obs - (1.0 - lambda_i) * np.abs(theta_obs - mu_t)
    else:
        raise ValueError(
            f"Unknown shrinkage_method='{shrinkage_method}'. "
            "Valid options: None (HB), 'cap' (HB_cap), 'down_only' (HB_down)."
        )


# --------------------------------------------------
# Validation / sigma selection
# --------------------------------------------------

def validate_base_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def resolve_sigma_column(
    df: pd.DataFrame,
    uncertainty_method: str = DEFAULT_UNCERTAINTY_METHOD,
    sigma_col: Optional[str] = DEFAULT_SIGMA_COL,
) -> str:
    if sigma_col is not None:
        if sigma_col not in df.columns:
            raise ValueError(
                f"sigma_col='{sigma_col}' was requested, but this column is not in the input file."
            )
        return sigma_col

    method = uncertainty_method.upper()
    if method == "OLS":
        candidates = OLS_SIGMA_CANDIDATES
    elif method == "HB":
        candidates = HB_SIGMA_CANDIDATES
    elif method == "AUTO":
        candidates = HB_SIGMA_CANDIDATES + OLS_SIGMA_CANDIDATES
    else:
        raise ValueError("uncertainty_method must be one of: auto, OLS, HB")

    for c in candidates:
        if c in df.columns:
            return c

    sigma_like = [c for c in df.columns if "sigma" in c.lower() or "rmse" in c.lower()]
    raise ValueError(
        "Could not resolve a sigma column from the Step 2 input. "
        f"uncertainty_method='{uncertainty_method}', "
        f"available sigma-like columns={sigma_like}. "
        "Pass --sigma_col explicitly."
    )


def clean_input(df: pd.DataFrame, sigma_col: str) -> pd.DataFrame:
    df = df.copy()

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["PROF"] = pd.to_numeric(df["PROF"], errors="coerce")
    df[sigma_col] = pd.to_numeric(df[sigma_col], errors="coerce")
    df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")

    if "FormationYear" in df.columns:
        df["FormationYear"] = pd.to_numeric(df["FormationYear"], errors="coerce").astype("Int64")
    else:
        df["FormationYear"] = df["Year"] + 1

    df = df.replace([np.inf, -np.inf], np.nan)

    dupes = df.duplicated(subset=["Ticker", "Year"], keep=False)
    if dupes.any():
        raise ValueError(
            "Found duplicate Ticker-Year rows in input. "
            "latent_prof_model expects one row per firm-year."
        )

    df = df.dropna(
        subset=["Ticker", "Year", "FormationYear", "PROF", sigma_col, "MarketCap"]
    ).copy()

    df["Year"] = df["Year"].astype(int)
    df["FormationYear"] = df["FormationYear"].astype(int)
    df["sigma_acc"] = df[sigma_col]

    return df


# --------------------------------------------------
# Generic helpers
# --------------------------------------------------

def winsorize_series(
    s: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    if s.notna().sum() == 0:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def winsorize_array(
    x: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return x.copy()
    lo = np.quantile(x[valid], lower)
    hi = np.quantile(x[valid], upper)
    out = x.copy()
    out[valid] = np.clip(out[valid], lo, hi)
    return out


def reorder_columns(df: pd.DataFrame, sigma_source_col: str) -> pd.DataFrame:
    first_cols = [
        "Ticker", "Year", "FormationYear", "PROF", "PROF_w",
        "sigma_acc", sigma_source_col, "sigma_raw", "MarketCap",
        "theta_obs", "theta_post_mean", "theta_post_sd", "theta_post_sd_between",
        "p_q5", "p_q5_sd_mc",
        "p_median", "p_median_sd_mc",                           # ← NEW
        "lambda_i", "mu_t", "var_obs_t", "obs_var_i", "tau2_t",
        "q5_cutoff_obs", "median_cutoff_obs", "n_sigma_draws_used",  # ← median_cutoff_obs NEW
    ]
    existing_first, seen = [], set()
    for c in first_cols:
        if c in df.columns and c not in seen:
            existing_first.append(c)
            seen.add(c)
    remaining = [c for c in df.columns if c not in seen]
    return df[existing_first + remaining].copy()


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def evenly_spaced_selection(items: List[str], n_select: Optional[int]) -> List[str]:
    if n_select is None or n_select >= len(items):
        return items.copy()
    idx = np.linspace(0, len(items) - 1, num=n_select, dtype=int)
    idx = np.unique(idx)
    if len(idx) < n_select:
        needed = n_select - len(idx)
        extra = [i for i in range(len(items)) if i not in set(idx)][:needed]
        idx = np.sort(np.concatenate([idx, np.array(extra, dtype=int)]))
    return [items[i] for i in idx[:n_select]]


# --------------------------------------------------
# Static panel prep
# --------------------------------------------------

def prepare_static_panel(
    df: pd.DataFrame,
    sigma_input_col: str,
    formation_year_min: int,
    formation_year_max: int,
    winsorize_prof: bool,
    winsorize_sigma: bool,
    winsor_lower: float,
    winsor_upper: float,
    min_firms_per_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, dict]]:
    df = df.copy()

    df = df[
        (df["FormationYear"] >= formation_year_min)
        & (df["FormationYear"] <= formation_year_max)
    ].copy()

    if winsorize_prof:
        df["PROF_w"] = (
            df.groupby("FormationYear", group_keys=False)["PROF"]
            .apply(lambda s: winsorize_series(s, winsor_lower, winsor_upper))
        )
    else:
        df["PROF_w"] = df["PROF"]

    if winsorize_sigma:
        df["sigma_raw"] = (
            df.groupby("FormationYear", group_keys=False)["sigma_acc"]
            .apply(lambda s: winsorize_series(s, winsor_lower, winsor_upper))
        )
    else:
        df["sigma_raw"] = df["sigma_acc"]

    df = df[df["sigma_raw"].notna() & (df["sigma_raw"] > 0)].copy()
    df = df.sort_values(["FormationYear", "Ticker"]).reset_index(drop=True)
    df["theta_obs"] = df["PROF_w"].astype(float)

    valid_years, year_rows = [], []

    for fy, sub in df.groupby("FormationYear", sort=True):
        if len(sub) < min_firms_per_year:
            print(f"Skipping FormationYear={fy}: only {len(sub)} firms")
            continue

        theta_obs = sub["theta_obs"].astype(float)
        var_obs_t = theta_obs.var(ddof=1)

        if not np.isfinite(var_obs_t) or var_obs_t <= 0:
            print(f"Skipping FormationYear={fy}: invalid observed PROF variance")
            continue

        mu_t = theta_obs.mean()
        q5_cutoff_obs    = theta_obs.quantile(0.80)
        median_cutoff_obs = theta_obs.quantile(0.50)          # ← NEW
        median_sigma_raw = sub["sigma_raw"].median()

        valid_years.append(fy)
        year_rows.append({
            "FormationYear": int(fy),
            "n_firms": int(len(sub)),
            "mu_t": float(mu_t),
            "var_obs_t": float(var_obs_t),
            "q5_cutoff_obs": float(q5_cutoff_obs),
            "median_cutoff_obs": float(median_cutoff_obs),    # ← NEW
            "median_sigma_raw": float(median_sigma_raw),
            "sigma_input_col": sigma_input_col,
        })

    if not valid_years:
        raise ValueError(
            "No valid formation years available after cleaning/filtering. "
            "Check input panel and sample restrictions."
        )

    df = df[df["FormationYear"].isin(valid_years)].copy()
    df = df.sort_values(["FormationYear", "Ticker"]).reset_index(drop=True)

    static_year_df = pd.DataFrame(year_rows).sort_values("FormationYear").reset_index(drop=True)

    year_info: Dict[int, dict] = {}
    for fy, sub in df.groupby("FormationYear", sort=True):
        pos = sub.index.to_numpy()
        theta_obs = sub["theta_obs"].to_numpy(dtype=float)
        mu_t = float(theta_obs.mean())
        var_obs_t = float(theta_obs.var(ddof=1))
        q5_cutoff_obs     = float(np.quantile(theta_obs, 0.80))
        median_cutoff_obs = float(np.quantile(theta_obs, 0.50))  # ← NEW

        year_info[int(fy)] = {
            "positions": pos,
            "theta_obs": theta_obs,
            "mu_t": mu_t,
            "var_obs_t": var_obs_t,
            "q5_cutoff_obs": q5_cutoff_obs,
            "median_cutoff_obs": median_cutoff_obs,              # ← NEW
            "n_firms": int(len(sub)),
        }

    return df, static_year_df, year_info


# --------------------------------------------------
# Plug-in EB logic
# --------------------------------------------------

def run_empirical_bayes_by_year_plugin(
    df: pd.DataFrame,
    sigma_input_col: str,
    static_year_df: pd.DataFrame,
    year_info: Dict[int, dict],
    noise_share_of_prof_var: float = DEFAULT_NOISE_SHARE_OF_PROF_VAR,
    winsorize_sigma: bool = DEFAULT_WINSORIZE_SIGMA,
    winsor_lower: float = DEFAULT_WINSOR_LOWER,
    winsor_upper: float = DEFAULT_WINSOR_UPPER,
    min_tau2: float = DEFAULT_MIN_TAU2,
    min_post_var: float = DEFAULT_MIN_POST_VAR,
    shrinkage_method: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    frames: List[pd.DataFrame] = []
    year_results: List[Dict] = []

    for fy, info in year_info.items():
        sub = df[df["FormationYear"] == fy].copy()

        theta_obs         = info["theta_obs"]
        mu_t              = info["mu_t"]
        var_obs_t         = info["var_obs_t"]
        q5_cutoff_obs     = info["q5_cutoff_obs"]
        median_cutoff_obs = info["median_cutoff_obs"]          # ← NEW

        sigma_vec = sub["sigma_acc"].to_numpy(dtype=float)
        if winsorize_sigma:
            sigma_vec = winsorize_array(sigma_vec, winsor_lower, winsor_upper)

        sigma_valid = np.isfinite(sigma_vec) & (sigma_vec > 0)
        if sigma_valid.sum() < 2:
            print(f"Skipping FormationYear={fy}: invalid sigma vector")
            continue

        sigma_vec = sigma_vec[sigma_valid]
        theta_obs_valid = theta_obs[sigma_valid]

        sigma_median = float(np.median(sigma_vec))
        if not np.isfinite(sigma_median) or sigma_median <= 0:
            print(f"Skipping FormationYear={fy}: invalid median sigma")
            continue

        sigma_rel = sigma_vec / sigma_median
        obs_var_base = noise_share_of_prof_var * var_obs_t
        obs_var_i = obs_var_base * (sigma_rel ** 2)
        avg_obs_var_i = float(obs_var_i.mean())

        tau2_t = max(var_obs_t - avg_obs_var_i, min_tau2)
        lambda_i = tau2_t / (tau2_t + obs_var_i)

        post_var_i = (tau2_t * obs_var_i) / (tau2_t + obs_var_i)
        post_var_i = np.maximum(post_var_i, min_post_var)
        theta_post_sd = np.sqrt(post_var_i)

        theta_post_mean = apply_shrinkage(
            theta_obs=theta_obs_valid,
            lambda_i=lambda_i,
            mu_t=mu_t,
            shrinkage_method=shrinkage_method,
        )

        z     = (q5_cutoff_obs - theta_post_mean) / theta_post_sd
        p_q5  = 1.0 - norm.cdf(z)

        z_med    = (median_cutoff_obs - theta_post_mean) / theta_post_sd  # ← NEW
        p_median = 1.0 - norm.cdf(z_med)                                  # ← NEW

        out = sub.loc[sigma_valid].copy()
        out["mu_t"]               = mu_t
        out["var_obs_t"]          = var_obs_t
        out["obs_var_i"]          = obs_var_i
        out["tau2_t"]             = tau2_t
        out["lambda_i"]           = lambda_i
        out["theta_post_mean"]    = theta_post_mean
        out["theta_post_sd"]      = theta_post_sd
        out["theta_post_sd_between"] = 0.0
        out["q5_cutoff_obs"]      = q5_cutoff_obs
        out["median_cutoff_obs"]  = median_cutoff_obs   # ← NEW
        out["p_q5"]               = p_q5
        out["p_q5_sd_mc"]         = 0.0
        out["p_median"]           = p_median            # ← NEW
        out["p_median_sd_mc"]     = 0.0                 # ← NEW
        out["n_sigma_draws_used"] = 1

        frames.append(out)
        year_results.append({
            "FormationYear": int(fy),
            "n_firms": int(len(out)),
            "mu_t": float(mu_t),
            "var_obs_t": float(var_obs_t),
            "avg_obs_var_i": float(avg_obs_var_i),
            "tau2_t": float(tau2_t),
            "q5_cutoff_obs": float(q5_cutoff_obs),
            "median_cutoff_obs": float(median_cutoff_obs),  # ← NEW
            "median_sigma_raw": float(sigma_median),
            "sigma_input_col": sigma_input_col,
            "n_sigma_draws_used": 1,
            "tau2_floor_share": float(tau2_t <= min_tau2 + 1e-15),
        })

        print(
            f"FormationYear={fy}: n={len(out)}, "
            f"var_obs={var_obs_t:.6f}, avg_obs_var={avg_obs_var_i:.6f}, tau2={tau2_t:.6f}"
        )

    if not frames:
        raise ValueError("No valid formation years produced EB results in plug-in mode.")

    signals_df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["FormationYear", "Ticker"])
        .reset_index(drop=True)
    )
    year_summary_df = (
        pd.DataFrame(year_results)
        .sort_values("FormationYear")
        .reset_index(drop=True)
    )
    return signals_df, year_summary_df


# --------------------------------------------------
# HB full-posterior helpers
# --------------------------------------------------

def get_parquet_column_names(parquet_path: Path) -> List[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        return list(pq.ParquetFile(parquet_path).schema.names)
    except Exception:
        tmp = pd.read_parquet(parquet_path)
        return list(tmp.columns)


def load_hb_full_posteriors(
    parquet_path: Path,
    n_sigma_draws: Optional[int] = DEFAULT_N_SIGMA_DRAWS,
) -> Tuple[pd.DataFrame, List[str], int]:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"HB full posterior parquet not found: {parquet_path}")

    all_cols = get_parquet_column_names(parquet_path)
    missing = sorted({"Ticker", "Year"} - set(all_cols))
    if missing:
        raise ValueError(f"HB full posterior parquet missing required columns: {missing}")

    draw_cols_all = [c for c in all_cols if str(c).startswith("draw_")]
    if not draw_cols_all:
        raise ValueError(
            "HB full posterior parquet does not contain any draw columns like draw_0, draw_1, ..."
        )

    selected_draw_cols = evenly_spaced_selection(draw_cols_all, n_sigma_draws)
    read_cols = ["Ticker", "Year"] + selected_draw_cols

    df = pd.read_parquet(parquet_path, columns=read_cols)
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Ticker", "Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    if df.duplicated(subset=["Ticker", "Year"], keep=False).any():
        raise ValueError(
            "HB full posterior parquet has duplicate Ticker-Year rows. "
            "Expected exactly one row per firm-year."
        )

    for c in selected_draw_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, selected_draw_cols, len(draw_cols_all)


def save_progress(
    path: Path,
    mode: str,
    draws_completed: int,
    total_draws: int,
    elapsed_seconds: float,
    current_draw: Optional[str],
) -> None:
    pct = 0.0 if total_draws == 0 else draws_completed / total_draws
    eta_seconds = None
    if draws_completed > 0 and total_draws > draws_completed:
        avg_per_draw = elapsed_seconds / draws_completed
        eta_seconds = avg_per_draw * (total_draws - draws_completed)

    write_json(path, {
        "mode": mode,
        "draws_completed": int(draws_completed),
        "total_draws": int(total_draws),
        "pct_complete": float(pct),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_hms": format_seconds(elapsed_seconds),
        "eta_seconds": None if eta_seconds is None else float(eta_seconds),
        "eta_hms": None if eta_seconds is None else format_seconds(eta_seconds),
        "current_draw": current_draw,
        "updated_at_unix": time.time(),
    })


# --------------------------------------------------
# HB full propagation logic
# --------------------------------------------------

def run_empirical_bayes_full_propagation(
    df: pd.DataFrame,
    sigma_input_col: str,
    static_year_df: pd.DataFrame,
    year_info: Dict[int, dict],
    hb_full_posterior_parquet: str | Path,
    output_dir: str | Path,
    n_sigma_draws: Optional[int] = DEFAULT_N_SIGMA_DRAWS,
    checkpoint_every_draws: int = DEFAULT_CHECKPOINT_EVERY_DRAWS,
    noise_share_of_prof_var: float = DEFAULT_NOISE_SHARE_OF_PROF_VAR,
    winsorize_sigma: bool = DEFAULT_WINSORIZE_SIGMA,
    winsor_lower: float = DEFAULT_WINSOR_LOWER,
    winsor_upper: float = DEFAULT_WINSOR_UPPER,
    min_firms_per_year: int = DEFAULT_MIN_FIRMS_PER_YEAR,
    min_tau2: float = DEFAULT_MIN_TAU2,
    min_post_var: float = DEFAULT_MIN_POST_VAR,
    shrinkage_method: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    output_dir = Path(output_dir)
    progress_path = output_dir / "latent_prof_progress.json"

    hb_draw_df, selected_draw_cols, n_available_draws = load_hb_full_posteriors(
        parquet_path=Path(hb_full_posterior_parquet),
        n_sigma_draws=n_sigma_draws,
    )

    merged = df[["Ticker", "Year"]].merge(
        hb_draw_df, on=["Ticker", "Year"], how="left", validate="1:1",
    )

    if merged[selected_draw_cols].isna().any().any():
        n_missing_rows = int(merged[selected_draw_cols].isna().all(axis=1).sum())
        raise ValueError(
            "Some firm-year rows from the Step 3 base panel are missing in the HB full posterior parquet. "
            f"Completely missing rows: {n_missing_rows}"
        )

    draw_matrix = merged[selected_draw_cols].to_numpy(dtype=float)
    n_rows, total_draws = draw_matrix.shape

    count_i              = np.zeros(n_rows, dtype=np.int32)
    sum_theta_post       = np.zeros(n_rows, dtype=float)
    sum_theta_post_sq    = np.zeros(n_rows, dtype=float)
    sum_p_q5             = np.zeros(n_rows, dtype=float)
    sum_p_q5_sq          = np.zeros(n_rows, dtype=float)
    sum_p_median         = np.zeros(n_rows, dtype=float)   # ← NEW
    sum_p_median_sq      = np.zeros(n_rows, dtype=float)   # ← NEW
    sum_lambda           = np.zeros(n_rows, dtype=float)
    sum_obs_var          = np.zeros(n_rows, dtype=float)
    sum_post_var         = np.zeros(n_rows, dtype=float)

    year_order  = static_year_df["FormationYear"].tolist()
    year_to_idx = {fy: i for i, fy in enumerate(year_order)}

    year_sum_avg_obs_var  = np.zeros(len(year_order), dtype=float)
    year_sum_tau2         = np.zeros(len(year_order), dtype=float)
    year_sum_median_sigma = np.zeros(len(year_order), dtype=float)
    year_draw_count       = np.zeros(len(year_order), dtype=np.int32)
    year_tau2_floor_count = np.zeros(len(year_order), dtype=np.int32)

    t0 = time.time()
    print(
        f"[full propagation] Loaded {n_available_draws} available HB sigma draws. "
        f"Using {total_draws} draw(s)."
    )

    save_progress(progress_path, "full_propagation", 0, total_draws, 0.0, None)

    for j, draw_col in enumerate(selected_draw_cols):
        sigma_draw_all = draw_matrix[:, j]

        for fy, info in year_info.items():
            year_idx          = year_to_idx[fy]
            pos               = info["positions"]
            theta_obs         = info["theta_obs"]
            median_cutoff_obs = info["median_cutoff_obs"]    # ← NEW

            sigma_sub = sigma_draw_all[pos].astype(float)
            if winsorize_sigma:
                sigma_sub = winsorize_array(sigma_sub, winsor_lower, winsor_upper)

            valid = np.isfinite(sigma_sub) & (sigma_sub > 0)
            if valid.sum() < min_firms_per_year:
                continue

            pos_v       = pos[valid]
            theta_obs_v = theta_obs[valid]
            sigma_v     = sigma_sub[valid]

            mu_t      = float(theta_obs_v.mean())
            var_obs_t = float(theta_obs_v.var(ddof=1))
            if not np.isfinite(var_obs_t) or var_obs_t <= 0:
                continue

            q5_cutoff_obs = float(np.quantile(theta_obs_v, 0.80))
            sigma_median  = float(np.median(sigma_v))
            if not np.isfinite(sigma_median) or sigma_median <= 0:
                continue

            sigma_rel     = sigma_v / sigma_median
            obs_var_base  = noise_share_of_prof_var * var_obs_t
            obs_var_i     = obs_var_base * (sigma_rel ** 2)
            avg_obs_var_i = float(obs_var_i.mean())

            tau2_t   = max(var_obs_t - avg_obs_var_i, min_tau2)
            lambda_i = tau2_t / (tau2_t + obs_var_i)

            post_var_i = np.maximum(
                (tau2_t * obs_var_i) / (tau2_t + obs_var_i), min_post_var
            )
            post_sd_i = np.sqrt(post_var_i)

            theta_post_mean = apply_shrinkage(
                theta_obs=theta_obs_v,
                lambda_i=lambda_i,
                mu_t=mu_t,
                shrinkage_method=shrinkage_method,
            )

            z    = (q5_cutoff_obs - theta_post_mean) / post_sd_i
            p_q5 = 1.0 - norm.cdf(z)

            z_med    = (median_cutoff_obs - theta_post_mean) / post_sd_i  # ← NEW
            p_median = 1.0 - norm.cdf(z_med)                              # ← NEW

            count_i[pos_v]              += 1
            sum_theta_post[pos_v]       += theta_post_mean
            sum_theta_post_sq[pos_v]    += theta_post_mean ** 2
            sum_p_q5[pos_v]             += p_q5
            sum_p_q5_sq[pos_v]          += p_q5 ** 2
            sum_p_median[pos_v]         += p_median       # ← NEW
            sum_p_median_sq[pos_v]      += p_median ** 2  # ← NEW
            sum_lambda[pos_v]           += lambda_i
            sum_obs_var[pos_v]          += obs_var_i
            sum_post_var[pos_v]         += post_var_i

            year_sum_avg_obs_var[year_idx]  += avg_obs_var_i
            year_sum_tau2[year_idx]         += tau2_t
            year_sum_median_sigma[year_idx] += sigma_median
            year_draw_count[year_idx]       += 1
            year_tau2_floor_count[year_idx] += int(tau2_t <= min_tau2 + 1e-15)

        draws_completed = j + 1
        if (
            draws_completed == 1
            or draws_completed == total_draws
            or draws_completed % max(1, checkpoint_every_draws) == 0
        ):
            elapsed   = time.time() - t0
            pct       = 100.0 * draws_completed / total_draws
            remaining = (elapsed / draws_completed) * (total_draws - draws_completed)
            print(
                f"[full propagation] draw {draws_completed}/{total_draws} "
                f"({pct:.1f}%) | elapsed={format_seconds(elapsed)} "
                f"| eta={format_seconds(remaining)} | current={draw_col}"
            )
            save_progress(
                progress_path, "full_propagation",
                draws_completed, total_draws, elapsed, draw_col,
            )

    if (count_i == 0).any():
        n_bad = int((count_i == 0).sum())
        raise ValueError(
            f"Full propagation completed, but {n_bad} firm-year rows received zero usable sigma draws."
        )

    out = df.copy()

    theta_post_mean_mc = sum_theta_post / count_i
    p_q5_mc            = sum_p_q5 / count_i
    p_median_mc        = sum_p_median / count_i              # ← NEW
    lambda_mc          = sum_lambda / count_i
    obs_var_mc         = sum_obs_var / count_i
    mean_post_var_mc   = sum_post_var / count_i

    theta_between_var  = np.maximum(sum_theta_post_sq / count_i - theta_post_mean_mc ** 2, 0.0)
    p_q5_between_var   = np.maximum(sum_p_q5_sq / count_i - p_q5_mc ** 2, 0.0)
    p_median_between_var = np.maximum(sum_p_median_sq / count_i - p_median_mc ** 2, 0.0)  # ← NEW
    theta_total_var    = np.maximum(mean_post_var_mc + theta_between_var, min_post_var)

    out["theta_post_mean"]       = theta_post_mean_mc
    out["theta_post_sd"]         = np.sqrt(theta_total_var)
    out["theta_post_sd_between"] = np.sqrt(theta_between_var)
    out["p_q5"]                  = p_q5_mc
    out["p_q5_sd_mc"]            = np.sqrt(p_q5_between_var)
    out["p_median"]              = p_median_mc                         # ← NEW
    out["p_median_sd_mc"]        = np.sqrt(p_median_between_var)       # ← NEW
    out["lambda_i"]              = lambda_mc
    out["obs_var_i"]             = obs_var_mc
    out["n_sigma_draws_used"]    = count_i.astype(int)

    out = out.merge(
        static_year_df[["FormationYear", "mu_t", "var_obs_t", "q5_cutoff_obs", "median_cutoff_obs"]],  # ← median_cutoff_obs NEW
        on="FormationYear", how="left", validate="many_to_one",
    )

    year_summary = static_year_df.copy()
    year_summary["avg_obs_var_i"] = np.where(
        year_draw_count > 0, year_sum_avg_obs_var / year_draw_count, np.nan
    )
    year_summary["tau2_t"] = np.where(
        year_draw_count > 0, year_sum_tau2 / year_draw_count, np.nan
    )
    year_summary["median_sigma_raw"] = np.where(
        year_draw_count > 0, year_sum_median_sigma / year_draw_count, np.nan
    )
    year_summary["n_sigma_draws_used"] = year_draw_count.astype(int)
    year_summary["tau2_floor_share"]   = np.where(
        year_draw_count > 0, year_tau2_floor_count / year_draw_count, np.nan
    )

    out = out.merge(
        year_summary[["FormationYear", "tau2_t"]],
        on="FormationYear", how="left", validate="many_to_one",
    )

    out          = out.sort_values(["FormationYear", "Ticker"]).reset_index(drop=True)
    year_summary = year_summary.sort_values("FormationYear").reset_index(drop=True)

    elapsed_total = time.time() - t0
    write_json(progress_path, {
        "mode": "full_propagation",
        "status": "completed",
        "draws_completed": int(total_draws),
        "total_draws": int(total_draws),
        "elapsed_seconds": float(elapsed_total),
        "elapsed_hms": format_seconds(elapsed_total),
        "n_rows_output": int(len(out)),
        "n_years_output": int(year_summary["FormationYear"].nunique()),
        "n_available_draws_in_parquet": int(n_available_draws),
        "n_selected_draws": int(total_draws),
        "checkpoint_every_draws": int(checkpoint_every_draws),
        "hb_full_posterior_parquet": str(hb_full_posterior_parquet),
    })

    return out, year_summary, {
        "progress_json": str(progress_path),
        "n_available_draws_in_parquet": int(n_available_draws),
        "n_selected_draws": int(total_draws),
        "selected_draw_columns": selected_draw_cols,
    }


# --------------------------------------------------
# Single-variant runner
# --------------------------------------------------

def _run_single_variant(
    variant_name: str,
    shrinkage_method: Optional[str],
    static_df: pd.DataFrame,
    static_year_df: pd.DataFrame,
    year_info: Dict[int, dict],
    sigma_input_col: str,
    output_dir: Path,
    use_full_propagation: bool,
    hb_full_posterior_parquet: Optional[str | Path],
    noise_share_of_prof_var: float,
    winsorize_sigma: bool,
    winsor_lower: float,
    winsor_upper: float,
    min_firms_per_year: int,
    n_sigma_draws: Optional[int],
    checkpoint_every_draws: int,
    uncertainty_method: str,
    formation_year_min: int,
    formation_year_max: int,
    winsorize_prof: bool,
) -> dict:
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Variant:  {variant_name}  (shrinkage_method={shrinkage_method})")
    print(f"Output:   {variant_dir}")
    print(f"{'=' * 60}")

    progress_json_path = variant_dir / "latent_prof_progress.json"
    extra_info = {
        "progress_json": str(progress_json_path),
        "n_available_draws_in_parquet": None,
        "n_selected_draws": None,
        "selected_draw_columns": None,
    }

    if use_full_propagation:
        signals_df, year_summary_df, extra_info = run_empirical_bayes_full_propagation(
            df=static_df,
            sigma_input_col=sigma_input_col,
            static_year_df=static_year_df,
            year_info=year_info,
            hb_full_posterior_parquet=hb_full_posterior_parquet,
            output_dir=variant_dir,
            n_sigma_draws=n_sigma_draws,
            checkpoint_every_draws=checkpoint_every_draws,
            noise_share_of_prof_var=noise_share_of_prof_var,
            winsorize_sigma=winsorize_sigma,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            min_firms_per_year=min_firms_per_year,
            shrinkage_method=shrinkage_method,
        )
    else:
        signals_df, year_summary_df = run_empirical_bayes_by_year_plugin(
            df=static_df,
            sigma_input_col=sigma_input_col,
            static_year_df=static_year_df,
            year_info=year_info,
            noise_share_of_prof_var=noise_share_of_prof_var,
            winsorize_sigma=winsorize_sigma,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            shrinkage_method=shrinkage_method,
        )
        write_json(progress_json_path, {
            "mode": "plugin",
            "status": "completed",
            "draws_completed": 1,
            "total_draws": 1,
            "elapsed_seconds": 0.0,
            "elapsed_hms": "00:00",
        })

    signals_df = reorder_columns(signals_df, sigma_source_col=sigma_input_col)

    firm_year_path      = variant_dir / "latent_prof_firm_year.csv"
    year_summary_path   = variant_dir / "latent_prof_year_summary.csv"
    config_path         = variant_dir / "latent_prof_config.json"
    selected_draws_path = variant_dir / "latent_prof_selected_draws.json"

    signals_df.to_csv(firm_year_path, index=False)
    year_summary_df.to_csv(year_summary_path, index=False)

    write_json(config_path, {
        "variant": variant_name,
        "shrinkage_method": str(shrinkage_method),
        "sigma_input_col": sigma_input_col,
        "uncertainty_method": uncertainty_method,
        "formation_year_min": formation_year_min,
        "formation_year_max": formation_year_max,
        "noise_share_of_prof_var": noise_share_of_prof_var,
        "winsorize_prof": winsorize_prof,
        "winsorize_sigma": winsorize_sigma,
        "winsor_lower": winsor_lower,
        "winsor_upper": winsor_upper,
        "min_firms_per_year": min_firms_per_year,
        "use_full_propagation": bool(use_full_propagation),
        "hb_full_posterior_parquet": (
            None if hb_full_posterior_parquet is None else str(hb_full_posterior_parquet)
        ),
        "n_sigma_draws_used": extra_info.get("n_selected_draws"),
        "n_rows_output": int(len(signals_df)),
        "n_years_output": int(year_summary_df["FormationYear"].nunique()),
    })
    write_json(selected_draws_path, {
        "selected_draw_columns": extra_info.get("selected_draw_columns"),
        "n_selected_draws": extra_info.get("n_selected_draws"),
        "n_available_draws_in_parquet": extra_info.get("n_available_draws_in_parquet"),
    })

    print(f"Saved: {firm_year_path}")
    print(f"Saved: {year_summary_path}")

    return {
        "variant": variant_name,
        "output_dir": str(variant_dir),
        "firm_year_csv": str(firm_year_path),
        "year_summary_csv": str(year_summary_path),
        "config_json": str(config_path),
        "progress_json": str(progress_json_path),
    }


# --------------------------------------------------
# Public pipeline function
# --------------------------------------------------

def run_latent_prof_model(
    input_csv: str | Path,
    output_dir: str | Path,
    uncertainty_method: str = DEFAULT_UNCERTAINTY_METHOD,
    sigma_col: Optional[str] = DEFAULT_SIGMA_COL,
    formation_year_min: int = DEFAULT_FORMATION_YEAR_MIN,
    formation_year_max: int = DEFAULT_FORMATION_YEAR_MAX,
    noise_share_of_prof_var: float = DEFAULT_NOISE_SHARE_OF_PROF_VAR,
    winsorize_prof: bool = DEFAULT_WINSORIZE_PROF,
    winsorize_sigma: bool = DEFAULT_WINSORIZE_SIGMA,
    winsor_lower: float = DEFAULT_WINSOR_LOWER,
    winsor_upper: float = DEFAULT_WINSOR_UPPER,
    min_firms_per_year: int = DEFAULT_MIN_FIRMS_PER_YEAR,
    use_full_propagation: bool = DEFAULT_USE_FULL_PROPAGATION,
    hb_full_posterior_parquet: Optional[str | Path] = DEFAULT_HB_FULL_POSTERIOR_PARQUET,
    n_sigma_draws: Optional[int] = DEFAULT_N_SIGMA_DRAWS,
    checkpoint_every_draws: int = DEFAULT_CHECKPOINT_EVERY_DRAWS,
) -> dict:
    """
    Step 3: Empirical Bayes latent PROF model.

    Runs all three shrinkage variants in one call:
        <output_dir>/HB/       latent_prof_firm_year.csv  — standard
        <output_dir>/HB_cap/   latent_prof_firm_year.csv  — capped at theta_obs
        <output_dir>/HB_down/  latent_prof_firm_year.csv  — always downward
    """
    input_csv  = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    validate_base_columns(df)

    sigma_input_col = resolve_sigma_column(
        df=df, uncertainty_method=uncertainty_method, sigma_col=sigma_col,
    )
    print(f"Using sigma column: {sigma_input_col} (uncertainty_method={uncertainty_method})")

    df = clean_input(df, sigma_col=sigma_input_col)

    static_df, static_year_df, year_info = prepare_static_panel(
        df=df,
        sigma_input_col=sigma_input_col,
        formation_year_min=formation_year_min,
        formation_year_max=formation_year_max,
        winsorize_prof=winsorize_prof,
        winsorize_sigma=winsorize_sigma,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        min_firms_per_year=min_firms_per_year,
    )
    print(f"Static panel shape: {static_df.shape}")

    if use_full_propagation:
        if str(uncertainty_method).upper() == "OLS":
            raise ValueError("use_full_propagation=True requires HB output, not OLS.")
        if hb_full_posterior_parquet is None:
            raise ValueError(
                "use_full_propagation=True requires --hb_full_posterior_parquet "
                "pointing to sigma_posteriors_full.parquet."
            )

    all_outputs = {}
    for variant_name, shrinkage_method in VARIANTS.items():
        result = _run_single_variant(
            variant_name=variant_name,
            shrinkage_method=shrinkage_method,
            static_df=static_df,
            static_year_df=static_year_df,
            year_info=year_info,
            sigma_input_col=sigma_input_col,
            output_dir=output_dir,
            use_full_propagation=use_full_propagation,
            hb_full_posterior_parquet=hb_full_posterior_parquet,
            noise_share_of_prof_var=noise_share_of_prof_var,
            winsorize_sigma=winsorize_sigma,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            min_firms_per_year=min_firms_per_year,
            n_sigma_draws=n_sigma_draws,
            checkpoint_every_draws=checkpoint_every_draws,
            uncertainty_method=uncertainty_method,
            formation_year_min=formation_year_min,
            formation_year_max=formation_year_max,
            winsorize_prof=winsorize_prof,
        )
        all_outputs[variant_name] = result

    print(f"\n{'=' * 60}")
    print("All variants complete.")
    for name, res in all_outputs.items():
        print(f"  {name:10s} → {res['firm_year_csv']}")
    print(f"{'=' * 60}")

    return all_outputs


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3 latent PROF model — produces HB, HB_cap and HB_down variants."
    )

    parser.add_argument("--input_csv",    type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument(
        "--uncertainty_method", type=str,
        default=DEFAULT_UNCERTAINTY_METHOD, choices=["auto", "OLS", "HB"],
    )
    parser.add_argument("--sigma_col",              type=str,   default=None)
    parser.add_argument("--formation_year_min",     type=int,   default=DEFAULT_FORMATION_YEAR_MIN)
    parser.add_argument("--formation_year_max",     type=int,   default=DEFAULT_FORMATION_YEAR_MAX)
    parser.add_argument("--noise_share_of_prof_var",type=float, default=DEFAULT_NOISE_SHARE_OF_PROF_VAR)
    parser.add_argument("--winsor_lower",           type=float, default=DEFAULT_WINSOR_LOWER)
    parser.add_argument("--winsor_upper",           type=float, default=DEFAULT_WINSOR_UPPER)
    parser.add_argument("--min_firms_per_year",     type=int,   default=DEFAULT_MIN_FIRMS_PER_YEAR)
    parser.add_argument("--use_full_propagation",   action="store_true")
    parser.add_argument("--hb_full_posterior_parquet", type=str, default=None)
    parser.add_argument("--n_sigma_draws",          type=int,   default=None)
    parser.add_argument(
        "--checkpoint_every_draws", type=int, default=DEFAULT_CHECKPOINT_EVERY_DRAWS,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_latent_prof_model(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        uncertainty_method=args.uncertainty_method,
        sigma_col=args.sigma_col,
        formation_year_min=args.formation_year_min,
        formation_year_max=args.formation_year_max,
        noise_share_of_prof_var=args.noise_share_of_prof_var,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        min_firms_per_year=args.min_firms_per_year,
        use_full_propagation=args.use_full_propagation,
        hb_full_posterior_parquet=args.hb_full_posterior_parquet,
        n_sigma_draws=args.n_sigma_draws,
        checkpoint_every_draws=args.checkpoint_every_draws,
    )