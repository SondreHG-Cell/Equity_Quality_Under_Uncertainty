# latent_prof_model.py

from __future__ import annotations

import argparse
import json
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

DEFAULT_NOISE_SHARE_OF_PROF_VAR = 0.10

DEFAULT_WINSORIZE_PROF = True
DEFAULT_WINSORIZE_SIGMA = True
DEFAULT_WINSOR_LOWER = 0.01
DEFAULT_WINSOR_UPPER = 0.99

DEFAULT_MIN_FIRMS_PER_YEAR = 20
DEFAULT_MIN_TAU2 = 1e-8
DEFAULT_MIN_POST_VAR = 1e-12


# --------------------------------------------------
# Candidate sigma columns from Step 2
# --------------------------------------------------

OLS_SIGMA_CANDIDATES = [
    "sigma_acc",
    "sigma_ols",
    "sigma_hat",
    "rmse",
    "rmse_acc",
    "sigma_acc_abs",
]

HB_SIGMA_CANDIDATES = [
    "sigma_post_mean",
    "sigma_acc_post_mean",
    "sigma_hb",
    "sigma_hb_mean",
    "posterior_mean_sigma",
    "sigma_mean",
    "sigma_acc",
]

REQUIRED_BASE_COLUMNS = [
    "Ticker",
    "Year",
    "PROF",
    "MarketCap",
]


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
    """
    Choose which Step 2 sigma column to use in Step 3.

    Priority:
    1. explicit sigma_col if supplied
    2. method-specific auto-detection
    3. generic fallback search
    """
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


def clean_input(
    df: pd.DataFrame,
    sigma_col: str,
) -> pd.DataFrame:
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

    # Standardize for downstream files
    df["sigma_acc"] = df[sigma_col]

    return df


# --------------------------------------------------
# Helpers
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


def reorder_columns(df: pd.DataFrame, sigma_source_col: str) -> pd.DataFrame:
    first_cols = [
        "Ticker",
        "Year",
        "FormationYear",
        "PROF",
        "PROF_w",
        "sigma_acc",
        sigma_source_col,
        "sigma_raw",
        "MarketCap",
        "theta_obs",
        "theta_post_mean",
        "theta_post_sd",
        "p_q5",
        "lambda_i",
        "mu_t",
        "var_obs_t",
        "obs_var_i",
        "tau2_t",
        "q5_cutoff_obs",
    ]
    existing_first = []
    seen = set()
    for c in first_cols:
        if c in df.columns and c not in seen:
            existing_first.append(c)
            seen.add(c)

    remaining = [c for c in df.columns if c not in seen]
    return df[existing_first + remaining].copy()


# --------------------------------------------------
# Core EB logic
# --------------------------------------------------

def run_empirical_bayes_by_year(
    df: pd.DataFrame,
    sigma_input_col: str,
    noise_share_of_prof_var: float = DEFAULT_NOISE_SHARE_OF_PROF_VAR,
    winsorize_prof: bool = DEFAULT_WINSORIZE_PROF,
    winsorize_sigma: bool = DEFAULT_WINSORIZE_SIGMA,
    winsor_lower: float = DEFAULT_WINSOR_LOWER,
    winsor_upper: float = DEFAULT_WINSOR_UPPER,
    min_firms_per_year: int = DEFAULT_MIN_FIRMS_PER_YEAR,
    min_tau2: float = DEFAULT_MIN_TAU2,
    min_post_var: float = DEFAULT_MIN_POST_VAR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Empirical Bayes shrinkage year by year.

    Uses a plug-in sigma estimate from Step 2, regardless of whether it came
    from OLS or HB. Internally, the chosen sigma column has already been
    standardized to df['sigma_acc'].
    """
    df = df.copy()

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

    df = df[df["sigma_raw"] > 0].copy()

    frames: List[pd.DataFrame] = []
    year_results: List[Dict] = []

    for fy, sub in df.groupby("FormationYear", sort=True):
        sub = sub.copy()

        if len(sub) < min_firms_per_year:
            print(f"Skipping FormationYear={fy}: only {len(sub)} firms")
            continue

        theta_obs = sub["PROF_w"].astype(float)

        mu_t = theta_obs.mean()
        var_obs = theta_obs.var(ddof=1)

        if not np.isfinite(var_obs) or var_obs <= 0:
            print(f"Skipping FormationYear={fy}: invalid observed PROF variance")
            continue

        sigma_median = sub["sigma_raw"].median()
        if not np.isfinite(sigma_median) or sigma_median <= 0:
            print(f"Skipping FormationYear={fy}: invalid median sigma")
            continue

        sigma_rel = sub["sigma_raw"] / sigma_median

        obs_var_base = noise_share_of_prof_var * var_obs
        obs_var_i = obs_var_base * (sigma_rel ** 2)

        avg_obs_var = obs_var_i.mean()
        tau2_t = max(var_obs - avg_obs_var, min_tau2)

        lambda_i = tau2_t / (tau2_t + obs_var_i)
        theta_post_mean = lambda_i * theta_obs + (1.0 - lambda_i) * mu_t

        post_var_i = (tau2_t * obs_var_i) / (tau2_t + obs_var_i)
        post_var_i = np.maximum(post_var_i, min_post_var)
        post_sd_i = np.sqrt(post_var_i)

        q5_cutoff = theta_obs.quantile(0.80)
        z = (q5_cutoff - theta_post_mean) / post_sd_i
        p_q5 = 1.0 - norm.cdf(z)

        sub["theta_obs"] = theta_obs
        sub["mu_t"] = mu_t
        sub["var_obs_t"] = var_obs
        sub["obs_var_i"] = obs_var_i
        sub["tau2_t"] = tau2_t
        sub["lambda_i"] = lambda_i
        sub["theta_post_mean"] = theta_post_mean
        sub["theta_post_sd"] = post_sd_i
        sub["q5_cutoff_obs"] = q5_cutoff
        sub["p_q5"] = p_q5

        frames.append(sub)

        year_results.append(
            {
                "FormationYear": int(fy),
                "n_firms": int(len(sub)),
                "mu_t": float(mu_t),
                "var_obs_t": float(var_obs),
                "avg_obs_var_i": float(avg_obs_var),
                "tau2_t": float(tau2_t),
                "q5_cutoff_obs": float(q5_cutoff),
                "median_sigma_raw": float(sigma_median),
                "sigma_input_col": sigma_input_col,
            }
        )

        print(
            f"FormationYear={fy}: n={len(sub)}, "
            f"var_obs={var_obs:.6f}, avg_obs_var={avg_obs_var:.6f}, tau2={tau2_t:.6f}"
        )

    if not frames:
        raise ValueError(
            "No valid formation years produced EB results. "
            "Check sample restrictions and sigma selection."
        )

    signals_eb_df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["FormationYear", "Ticker"])
        .reset_index(drop=True)
    )

    signals_year_df = (
        pd.DataFrame(year_results)
        .sort_values("FormationYear")
        .reset_index(drop=True)
    )

    return signals_eb_df, signals_year_df


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
) -> dict:
    """
    Step 3: Empirical Bayes latent PROF model.

    Supports Step 2 input from either:
    - OLS-based uncertainty model
    - HB-based uncertainty model

    The chosen Step 2 uncertainty column is standardized internally to:
        sigma_acc

    Main output for Step 4:
        latent_prof_firm_year.csv
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    validate_base_columns(df)
    sigma_input_col = resolve_sigma_column(
        df=df,
        uncertainty_method=uncertainty_method,
        sigma_col=sigma_col,
    )

    print(f"Using sigma column: {sigma_input_col} (uncertainty_method={uncertainty_method})")

    df = clean_input(df, sigma_col=sigma_input_col)

    df = df[
        (df["FormationYear"] >= formation_year_min)
        & (df["FormationYear"] <= formation_year_max)
    ].copy()

    print(f"Input shape after cleaning/filtering: {df.shape}")

    signals_eb_df, signals_year_df = run_empirical_bayes_by_year(
        df=df,
        sigma_input_col=sigma_input_col,
        noise_share_of_prof_var=noise_share_of_prof_var,
        winsorize_prof=winsorize_prof,
        winsorize_sigma=winsorize_sigma,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        min_firms_per_year=min_firms_per_year,
    )

    signals_eb_df = reorder_columns(signals_eb_df, sigma_source_col=sigma_input_col)

    firm_year_path = output_dir / "latent_prof_firm_year.csv"
    year_summary_path = output_dir / "latent_prof_year_summary.csv"
    config_path = output_dir / "latent_prof_config.json"

    signals_eb_df.to_csv(firm_year_path, index=False)
    signals_year_df.to_csv(year_summary_path, index=False)

    config = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "uncertainty_method": uncertainty_method,
        "sigma_input_col": sigma_input_col,
        "formation_year_min": formation_year_min,
        "formation_year_max": formation_year_max,
        "noise_share_of_prof_var": noise_share_of_prof_var,
        "winsorize_prof": winsorize_prof,
        "winsorize_sigma": winsorize_sigma,
        "winsor_lower": winsor_lower,
        "winsor_upper": winsor_upper,
        "min_firms_per_year": min_firms_per_year,
        "n_rows_output": int(len(signals_eb_df)),
        "n_years_output": int(signals_year_df["FormationYear"].nunique()) if not signals_year_df.empty else 0,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved firm-year output: {firm_year_path}")
    print(f"Saved year summary:     {year_summary_path}")
    print(f"Saved config:           {config_path}")

    return {
        "output_dir": str(output_dir),
        "firm_year_csv": str(firm_year_path),
        "year_summary_csv": str(year_summary_path),
        "config_json": str(config_path),
    }


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 3 latent PROF Empirical Bayes model.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to Step 2 output CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save Step 3 outputs.")
    parser.add_argument(
        "--uncertainty_method",
        type=str,
        default=DEFAULT_UNCERTAINTY_METHOD,
        choices=["auto", "OLS", "HB"],
        help="Which Step 2 uncertainty model produced the sigma input.",
    )
    parser.add_argument(
        "--sigma_col",
        type=str,
        default=None,
        help="Optional explicit sigma column name. Overrides auto-detection.",
    )
    parser.add_argument("--formation_year_min", type=int, default=DEFAULT_FORMATION_YEAR_MIN)
    parser.add_argument("--formation_year_max", type=int, default=DEFAULT_FORMATION_YEAR_MAX)
    parser.add_argument("--noise_share_of_prof_var", type=float, default=DEFAULT_NOISE_SHARE_OF_PROF_VAR)
    parser.add_argument("--winsor_lower", type=float, default=DEFAULT_WINSOR_LOWER)
    parser.add_argument("--winsor_upper", type=float, default=DEFAULT_WINSOR_UPPER)
    parser.add_argument("--min_firms_per_year", type=int, default=DEFAULT_MIN_FIRMS_PER_YEAR)
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
    )