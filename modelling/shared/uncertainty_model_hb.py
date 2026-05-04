# uncertainty_model_hb.py
# Default version drops CFO_{t+1} from the accrual equation to avoid look-ahead.
# Optional variants can use analyst CFO forecasts or an explicit external CFO
# forecast model where those inputs are available before portfolio formation.

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

DEFAULT_ANALYST_CFO_FORECAST_CSV = "modelling/analysis/cfo_forecast_complete_cases_no_gaps_until_2024.csv"
ANALYST_FORECAST_FILENAME = "cfo_forecast_complete_cases_no_gaps_until_2024.csv"


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if (p / "data").exists() and (p / "modelling").exists():
            return p
    return here.parents[1]


def _resolve_optional_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return _find_project_root() / path


def resolve_analyst_cfo_forecast_csv(path_like: str | Path | None) -> Path:
    if path_like is not None:
        path = _resolve_optional_path(path_like)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Analyst CFO forecast CSV not found: {path}")
        return path

    project_root = _find_project_root()
    candidates = [
        project_root / DEFAULT_ANALYST_CFO_FORECAST_CSV,
        project_root / ANALYST_FORECAST_FILENAME,
        Path.cwd() / ANALYST_FORECAST_FILENAME,
        Path(__file__).resolve().parent / ANALYST_FORECAST_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate analyst CFO forecast CSV. Searched:\n" + searched
    )


def load_analyst_cfo_forecasts(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Load analyst CFO forecasts and keep the latest valid forecast available
    before the portfolio-formation cutoff for each Ticker x FiscalYear.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    required = ["Ticker", "FiscalYear", "ForecastDate", "cfo_forecast"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Analyst CFO forecast CSV missing required columns: {missing}")

    df = raw.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["FiscalYear"] = pd.to_numeric(df["FiscalYear"], errors="coerce").astype("Int64")
    df["ForecastDate"] = pd.to_datetime(df["ForecastDate"], errors="coerce")
    df["cfo_forecast"] = pd.to_numeric(df["cfo_forecast"], errors="coerce")

    if "forecast_window_end" in df.columns:
        df["ForecastCutoffDate"] = pd.to_datetime(df["forecast_window_end"], errors="coerce")
    else:
        cutoff_year = df["FiscalYear"].astype("float").astype("Int64") + 1
        df["ForecastCutoffDate"] = pd.to_datetime(
            cutoff_year.astype(str) + "-06-30",
            errors="coerce",
        )

    if "forecast_window_start" in df.columns:
        df["ForecastWindowStart"] = pd.to_datetime(df["forecast_window_start"], errors="coerce")
    elif "AnnouncementDate" in df.columns:
        df["ForecastWindowStart"] = pd.to_datetime(df["AnnouncementDate"], errors="coerce")
    else:
        df["ForecastWindowStart"] = pd.NaT

    valid = (
        df["Ticker"].ne("")
        & df["FiscalYear"].notna()
        & df["ForecastDate"].notna()
        & df["ForecastCutoffDate"].notna()
        & df["cfo_forecast"].notna()
        & (df["ForecastDate"] <= df["ForecastCutoffDate"])
        & (df["ForecastWindowStart"].isna() | (df["ForecastDate"] >= df["ForecastWindowStart"]))
    )
    if "has_valid_cfo_forecast" in df.columns:
        valid &= pd.to_numeric(df["has_valid_cfo_forecast"], errors="coerce").fillna(0).astype(int).eq(1)

    df["forecast_timing_valid"] = valid
    valid_df = df.loc[valid].copy()

    # If multiple forecasts exist, use the latest one observable before the cutoff.
    valid_df = (
        valid_df.sort_values(["Ticker", "FiscalYear", "ForecastDate"])
        .drop_duplicates(["Ticker", "FiscalYear"], keep="last")
        .reset_index(drop=True)
    )
    valid_df["FiscalYear"] = valid_df["FiscalYear"].astype(int)

    diagnostics = {
        "forecast_csv": str(path),
        "rows_raw": int(len(raw)),
        "rows_valid_timing": int(len(valid_df)),
        "rows_invalid_or_missing": int(len(raw) - len(valid_df)),
        "unique_firms_valid": int(valid_df["Ticker"].nunique()),
        "first_fiscal_year_valid": int(valid_df["FiscalYear"].min()) if not valid_df.empty else None,
        "last_fiscal_year_valid": int(valid_df["FiscalYear"].max()) if not valid_df.empty else None,
    }

    keep_cols = [
        "Ticker",
        "FiscalYear",
        "ForecastDate",
        "ForecastCutoffDate",
        "ForecastWindowStart",
        "cfo_forecast",
        "forecast_timing_valid",
    ]
    return valid_df[keep_cols].copy(), diagnostics


def merge_analyst_cfo_forecasts(
    data: pd.DataFrame,
    forecast_csv: str | Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts, forecast_diag = load_analyst_cfo_forecasts(forecast_csv)
    before_rows = len(data)

    merged = data.merge(
        forecasts,
        left_on=["Ticker", "Year"],
        right_on=["Ticker", "FiscalYear"],
        how="left",
        validate="many_to_one",
    ).copy()
    merged["has_analyst_cfo_forecast"] = merged["cfo_forecast"].notna()
    merged["CFO_lead1_analyst_scaled"] = merged["cfo_forecast"] / merged["AvgAT"]

    coverage = (
        merged.groupby("Year", as_index=False)
        .agg(
            firm_years_before_merge=("Ticker", "size"),
            firms_before_merge=("Ticker", "nunique"),
            firm_years_with_analyst_forecast=("has_analyst_cfo_forecast", "sum"),
            firms_with_analyst_forecast=(
                "Ticker",
                lambda s: s[merged.loc[s.index, "has_analyst_cfo_forecast"]].nunique(),
            ),
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )

    missing_after_merge = int(merged["CFO_lead1_analyst_scaled"].isna().sum())
    diagnostics = {
        **forecast_diag,
        "firm_years_before_merging_analyst_forecasts": int(before_rows),
        "firm_years_after_merging_analyst_forecasts": int(len(merged)),
        "firms_in_hb_panel_before_merge": int(data["Ticker"].nunique()),
        "firms_in_analyst_forecast_csv": int(forecasts["Ticker"].nunique()),
        "firm_years_with_analyst_cfo_forecast_after_merge": int(merged["has_analyst_cfo_forecast"].sum()),
        "firms_with_analyst_cfo_forecast_after_merge": int(
            merged.loc[merged["has_analyst_cfo_forecast"], "Ticker"].nunique()
        ),
        "first_fiscal_year_in_analyst_csv": int(forecasts["FiscalYear"].min()) if not forecasts.empty else None,
        "last_fiscal_year_in_analyst_csv": int(forecasts["FiscalYear"].max()) if not forecasts.empty else None,
        "missing_analyst_cfo_forecasts_after_merge": missing_after_merge,
        "firm_years_without_analyst_cfo_forecast_after_merge": missing_after_merge,
    }

    diag_path = output_dir / "analyst_cfo_forecast_merge_diagnostics.json"
    coverage_path = output_dir / "analyst_cfo_forecast_yearly_coverage.csv"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    coverage.to_csv(coverage_path, index=False)

    diagnostics["diagnostics_json"] = str(diag_path)
    diagnostics["yearly_coverage_csv"] = str(coverage_path)
    return merged, diagnostics, coverage


def _canonical_cfo_source(source: str | None, cfo_lead_mode: str, use_analyst: bool) -> str:
    if source is not None:
        raw = source.lower().replace("-", "_")
    elif use_analyst:
        raw = "analyst"
    else:
        raw = cfo_lead_mode.lower().replace("-", "_")

    aliases = {
        "analystcfo": "analyst",
        "analyst_cfo": "analyst",
        "best_external": "external",
        "ar1": "external",
        "none": "none",
        "realized": "realized",
        "realised": "realized",
        "analyst": "analyst",
        "hybrid": "hybrid",
        "external": "external",
    }
    if raw not in aliases:
        raise ValueError(
            "cfo_t1_source must be one of realized, analyst, hybrid, external, or none."
        )
    return aliases[raw]


def apply_cfo_t1_source_to_window(
    window_df: pd.DataFrame,
    cfo_t1_source: str,
) -> tuple[pd.DataFrame, dict, bool]:
    """
    Set CFO_lead1_pred_scaled for realized/analyst/hybrid modes.
    Returns (window_df_fixed, diagnostics, include_cfo_lead).
    """
    wdf = window_df.copy()
    diagnostics = {
        "cfo_t1_source": cfo_t1_source,
        "window_rows_before_cfo_source_filter": int(len(wdf)),
        "rows_dropped_missing_analyst_forecast": 0,
        "rows_dropped_missing_realized_cfo_lead": 0,
    }

    if cfo_t1_source == "none":
        diagnostics["window_rows_after_cfo_source_filter"] = int(len(wdf))
        return wdf, diagnostics, False

    if cfo_t1_source == "realized":
        wdf["CFO_lead1_pred_scaled"] = wdf["CFO_lead1_scaled"]
        missing = wdf["CFO_lead1_pred_scaled"].isna()
        diagnostics["rows_dropped_missing_realized_cfo_lead"] = int(missing.sum())
        wdf = wdf.loc[~missing].copy()

    elif cfo_t1_source == "analyst":
        wdf["CFO_lead1_pred_scaled"] = wdf["CFO_lead1_analyst_scaled"]
        missing = wdf["CFO_lead1_pred_scaled"].isna()
        diagnostics["rows_dropped_missing_analyst_forecast"] = int(missing.sum())
        wdf = wdf.loc[~missing].copy()

    elif cfo_t1_source == "hybrid":
        is_port = wdf["is_portfolio_year"].astype(bool)
        wdf["CFO_lead1_pred_scaled"] = np.where(
            is_port,
            wdf["CFO_lead1_analyst_scaled"],
            wdf["CFO_lead1_scaled"],
        )
        missing_analyst = is_port & wdf["CFO_lead1_analyst_scaled"].isna()
        missing_realized = (~is_port) & wdf["CFO_lead1_scaled"].isna()
        diagnostics["rows_dropped_missing_analyst_forecast"] = int(missing_analyst.sum())
        diagnostics["rows_dropped_missing_realized_cfo_lead"] = int(missing_realized.sum())
        wdf = wdf.loc[~(missing_analyst | missing_realized)].copy()

    else:
        raise ValueError(f"Unsupported direct CFO source: {cfo_t1_source}")

    diagnostics["window_rows_after_cfo_source_filter"] = int(len(wdf))
    return wdf.reset_index(drop=True), diagnostics, True


def enforce_training_requirements_after_cfo_filter(
    window_df: pd.DataFrame,
    min_train_years: int,
    min_firm_obs: int = 3,
) -> tuple[pd.DataFrame | None, dict]:
    train_df = window_df.loc[~window_df["is_portfolio_year"].astype(bool)].copy()
    diagnostics = {
        "training_years_after_cfo_filter": int(train_df["Year"].nunique()),
        "training_rows_after_cfo_filter_before_firm_filter": int(len(train_df)),
        "firms_removed_after_cfo_filter_insufficient_training": 0,
        "rows_removed_after_cfo_filter_insufficient_training": 0,
    }

    if len(train_df) == 0 or train_df["Year"].nunique() < min_train_years:
        diagnostics["window_dropped_insufficient_training_history"] = True
        return None, diagnostics

    firm_obs_counts = train_df.groupby("Ticker").size()
    firms_ok = firm_obs_counts[firm_obs_counts >= min_firm_obs].index
    before_firms = window_df["Ticker"].nunique()
    filtered = window_df.loc[window_df["Ticker"].isin(firms_ok)].copy()
    after_firms = filtered["Ticker"].nunique()
    diagnostics["firms_removed_after_cfo_filter_insufficient_training"] = int(before_firms - after_firms)
    diagnostics["rows_removed_after_cfo_filter_insufficient_training"] = int(len(window_df) - len(filtered))

    if filtered.empty:
        diagnostics["window_dropped_insufficient_training_history"] = True
        return None, diagnostics

    diagnostics["window_dropped_insufficient_training_history"] = False
    return filtered.reset_index(drop=True), diagnostics


def keep_firms_with_portfolio_year_rows(
    window_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    is_port = window_df["is_portfolio_year"].astype(bool)
    portfolio_firms = set(window_df.loc[is_port, "Ticker"])
    before_rows = int(len(window_df))
    before_firms = int(window_df["Ticker"].nunique())

    if not portfolio_firms:
        return window_df.iloc[0:0].copy(), {
            "portfolio_firms_after_cfo_source_filter": 0,
            "rows_removed_no_portfolio_year_row_after_cfo_filter": before_rows,
            "firms_removed_no_portfolio_year_row_after_cfo_filter": before_firms,
        }

    filtered = window_df.loc[window_df["Ticker"].isin(portfolio_firms)].copy()
    return filtered.reset_index(drop=True), {
        "portfolio_firms_after_cfo_source_filter": int(len(portfolio_firms)),
        "rows_removed_no_portfolio_year_row_after_cfo_filter": int(before_rows - len(filtered)),
        "firms_removed_no_portfolio_year_row_after_cfo_filter": int(before_firms - filtered["Ticker"].nunique()),
    }


def _window_row_filter_from_csv(path: str | Path | None) -> dict[int, set[tuple[str, int]]] | None:
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return {}
    required = {"portfolio_year", "Ticker", "Year"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Window-row filter file missing required columns: {sorted(missing)}")

    filters: dict[int, set[tuple[str, int]]] = {}
    clean = df.dropna(subset=["portfolio_year", "Ticker", "Year"]).copy()
    clean["portfolio_year"] = clean["portfolio_year"].astype(int)
    clean["Year"] = clean["Year"].astype(int)
    for portfolio_year, group in clean.groupby("portfolio_year"):
        filters[int(portfolio_year)] = set(zip(group["Ticker"].astype(str), group["Year"].astype(int)))
    return filters


def _filter_window_to_reference_rows(
    window_df: pd.DataFrame,
    portfolio_year: int,
    row_filter_by_year: dict[int, set[tuple[str, int]]],
) -> tuple[pd.DataFrame, dict]:
    allowed = row_filter_by_year.get(int(portfolio_year), set())
    before_rows = int(len(window_df))
    before_firms = int(window_df["Ticker"].nunique())

    diagnostics = {
        "reference_sample_rows_available": int(len(allowed)),
        "window_rows_before_reference_sample_filter": before_rows,
        "window_firms_before_reference_sample_filter": before_firms,
        "window_rows_after_reference_sample_filter": 0,
        "window_firms_after_reference_sample_filter": 0,
        "rows_removed_by_reference_sample_filter": before_rows,
        "firms_removed_by_reference_sample_filter": before_firms,
    }

    if not allowed:
        return window_df.iloc[0:0].copy(), diagnostics

    keys = list(zip(window_df["Ticker"].astype(str), window_df["Year"].astype(int)))
    mask = pd.Series([key in allowed for key in keys], index=window_df.index)
    filtered = window_df.loc[mask].copy()

    diagnostics["window_rows_after_reference_sample_filter"] = int(len(filtered))
    diagnostics["window_firms_after_reference_sample_filter"] = int(filtered["Ticker"].nunique())
    diagnostics["rows_removed_by_reference_sample_filter"] = int(before_rows - len(filtered))
    diagnostics["firms_removed_by_reference_sample_filter"] = int(before_firms - filtered["Ticker"].nunique())
    return filtered.reset_index(drop=True), diagnostics


def _window_rows_used_frame(
    window_df: pd.DataFrame,
    portfolio_year: int,
    cfo_t1_source: str,
    specification_label: str,
    reference_sample_label: str | None,
) -> pd.DataFrame:
    cols = ["Ticker", "Year", "is_portfolio_year"]
    optional_cols = [
        "CFO_lead1_scaled",
        "CFO_lead1_analyst_scaled",
        "CFO_lead1_pred_scaled",
    ]
    out_cols = cols + [c for c in optional_cols if c in window_df.columns]
    out = window_df[out_cols].copy()
    out.insert(0, "portfolio_year", int(portfolio_year))
    out.insert(1, "specification_label", specification_label)
    out.insert(2, "cfo_t1_source", cfo_t1_source)
    out.insert(3, "reference_sample_label", reference_sample_label)
    return out


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
        # -------------------------
        # Market-level priors
        # -------------------------
        mu_0 = pm.Normal("mu_0", mu=0, sigma=0.1)

        omega = pm.HalfNormal("omega", sigma=0.05)
        tau = pm.HalfNormal("tau", sigma=0.05)
        sigma_0 = pm.HalfNormal("sigma_0", sigma=0.05)

        # -------------------------
        # Sector intercepts
        # -------------------------
        alpha_sector_raw = pm.Normal("alpha_sector_raw", mu=0, sigma=1, dims="sector")
        alpha_sector = pm.Deterministic(
            "alpha_sector",
            mu_0 + omega * alpha_sector_raw,
            dims="sector",
        )

        # -------------------------
        # Firm intercepts
        # -------------------------
        alpha_firm_raw = pm.Normal("alpha_firm_raw", mu=0, sigma=1, dims="firm")
        alpha_firm = pm.Deterministic(
            "alpha_firm",
            alpha_sector[firm_to_sector] + tau * alpha_firm_raw,
            dims="firm",
        )

        # -------------------------
        # Sector noise
        # -------------------------
        sigma_sector_raw = pm.HalfNormal("sigma_sector_raw", sigma=1, dims="sector")
        sigma_sector = pm.Deterministic(
            "sigma_sector",
            sigma_0 * sigma_sector_raw,
            dims="sector",
        )

        # -------------------------
        # Firm noise
        # -------------------------
        sigma_firm_raw = pm.HalfNormal("sigma_firm_raw", sigma=1, dims="firm")
        sigma_firm = pm.Deterministic(
            "sigma_firm",
            sigma_sector[firm_to_sector] * sigma_firm_raw,
            dims="firm",
        )

        # -------------------------
        # Shared slopes
        # -------------------------
        b_lag = pm.Normal("beta_CFO_lag1", mu=0, sigma=0.3)
        b_cur = pm.Normal("beta_CFO_curr", mu=0, sigma=0.3)
        b_rev = pm.Normal("beta_dREV", mu=0, sigma=0.3)
        b_ppe = pm.Normal("beta_PPE", mu=0, sigma=0.3)

        if include_cfo_lead:
            b_lead = pm.Normal("beta_CFO_lead1", mu=0, sigma=0.3)

        mu_wca = (
            alpha_firm[firm_idx]
            + b_lag * cfo_lag1
            + b_cur * cfo_curr
            + b_rev * drev
            + b_ppe * ppe
        )

        if include_cfo_lead:
            mu_wca = mu_wca + b_lead * cfo_lead_fixed

        mu_wca_expected = pm.Deterministic("mu_wca_expected", mu_wca, dims="obs")

        # -------------------------
        # Student-t likelihood
        # -------------------------
        nu_minus_two = pm.Exponential("nu_minus_two", lam=1 / 10)
        nu = pm.Deterministic("nu", 2 + nu_minus_two)

        pm.StudentT(
            "WCA_obs",
            nu=nu,
            mu=mu_wca_expected,
            sigma=sigma_firm[firm_idx],
            observed=y,
            dims="obs",
        )

        pm.Deterministic(
            "sigma_firm_sd",
            sigma_firm * pm.math.sqrt(nu / (nu - 2)),
            dims="firm",
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


def extract_expected_accrual_summary(
    trace: az.InferenceData,
    trace_info: dict,
    window_df: pd.DataFrame,
    portfolio_year: int,
    cfo_t1_source: str,
) -> pd.DataFrame:
    if "mu_wca_expected" not in trace.posterior:
        return pd.DataFrame()

    mu_samples = trace.posterior["mu_wca_expected"].values
    mu_samples = mu_samples.reshape(-1, mu_samples.shape[-1])
    wdf = window_df.reset_index(drop=True).copy()

    rows = []
    for obs_idx, row in wdf.iterrows():
        draws = mu_samples[:, obs_idx]
        rows.append(
            {
                "Year": int(portfolio_year),
                "Ticker": row["Ticker"],
                "firm_idx": int(row["firm_idx"]),
                "is_portfolio_year": bool(row["is_portfolio_year"]),
                "cfo_t1_source": cfo_t1_source,
                "expected_wca_mean": float(np.mean(draws)),
                "expected_wca_median": float(np.median(draws)),
                "expected_wca_std": float(np.std(draws)),
                "expected_wca_q05": float(np.percentile(draws, 5)),
                "expected_wca_q95": float(np.percentile(draws, 95)),
                "observed_wca_scaled": float(row["WCA_scaled"]),
                "cfo_t1_scaled_used": (
                    float(row["CFO_lead1_pred_scaled"])
                    if "CFO_lead1_pred_scaled" in row.index and pd.notna(row["CFO_lead1_pred_scaled"])
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _rename_sigma_summary_columns(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    rename = {
        "sigma_mean": f"{prefix}_mean",
        "sigma_median": f"{prefix}_median",
        "sigma_std": f"{prefix}_std",
        "sigma_q05": f"{prefix}_q05",
        "sigma_q95": f"{prefix}_q95",
        "n_draws": f"{prefix}_n_draws",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


# =============================================================================
# Main pipeline function
# =============================================================================

def _run_uncertainty_model_hb_single(
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
    cfo_lead_mode: str = "none",
    cfo_t1_source: str | None = None,
    use_analyst_cfo_forecast: bool = False,
    analyst_cfo_forecast_csv: str | Path | None = None,
    specification_label: str | None = None,
    window_row_filter_by_year: dict[int, set[tuple[str, int]]] | str | Path | None = None,
    window_row_filter_label: str | None = None,
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
    cfo_t1_source = _canonical_cfo_source(
        cfo_t1_source,
        cfo_lead_mode=cfo_lead_mode,
        use_analyst=use_analyst_cfo_forecast,
    )
    specification_label = specification_label or cfo_t1_source
    if not isinstance(window_row_filter_by_year, dict):
        window_row_filter_by_year = _window_row_filter_from_csv(window_row_filter_by_year)
    if cfo_t1_source in {"analyst", "hybrid"}:
        analyst_cfo_forecast_csv = resolve_analyst_cfo_forecast_csv(analyst_cfo_forecast_csv)

    data = pd.read_csv(input_csv)

    data = compute_wca(data)
    data = build_regressors(data, include_lead=True)
    data = mark_usable(data)
    data, firm_map, sector_map, firm_sector = assign_indices(data)

    analyst_merge_diagnostics = None
    analyst_yearly_coverage = None
    if cfo_t1_source in {"analyst", "hybrid"}:
        data, analyst_merge_diagnostics, analyst_yearly_coverage = merge_analyst_cfo_forecasts(
            data=data,
            forecast_csv=analyst_cfo_forecast_csv,
            output_dir=output_dir,
        )

    print(
        f"Panel: {len(data)} firm-years, {data['Ticker'].nunique()} firms, "
        f"{data['Year'].min()}–{data['Year'].max()}"
    )
    print(f"Usable: {data['usable'].sum()}")
    print(f"CFO_lead1 non-null: {data['CFO_lead1_scaled'].notna().sum()}")
    print(f"CFO_t+1 source: {cfo_t1_source}")
    if analyst_merge_diagnostics is not None:
        print(
            "Analyst CFO forecasts: "
            f"{analyst_merge_diagnostics['firms_in_analyst_forecast_csv']} firms, "
            f"{analyst_merge_diagnostics['first_fiscal_year_in_analyst_csv']}–"
            f"{analyst_merge_diagnostics['last_fiscal_year_in_analyst_csv']}; "
            f"missing after merge={analyst_merge_diagnostics['missing_analyst_cfo_forecasts_after_merge']}"
        )

    portfolio_years_to_run = sorted(
        y for y in data["Year"].unique() if year_start <= y <= year_end
    )

    all_results = {}
    all_results_scale = {}
    expected_accrual_frames = []
    window_rows_used_frames = []
    window_diagnostics = []
    last_model = None
    last_trace = None
    last_window_df = None

    for port_year in portfolio_years_to_run:
        checkpoint_path = checkpoint_dir / f"hb_checkpoint_{specification_label}_{port_year}_sigma_sd.pkl"

        if checkpoint_path.exists():
            print(f"Loading checkpoint for {port_year}")
            with open(checkpoint_path, "rb") as f:
                checkpoint_payload = pickle.load(f)
            if (
                isinstance(checkpoint_payload, dict)
                and "sigma_sd" in checkpoint_payload
                and "sigma_scale" in checkpoint_payload
            ):
                all_results[port_year] = checkpoint_payload["sigma_sd"][port_year]
                all_results_scale[port_year] = checkpoint_payload["sigma_scale"][port_year]
            else:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} does not contain sigma_sd/sigma_scale payload. "
                    "Remove old checkpoints or rerun in a fresh output directory."
                )
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
        year_diag = {
            "portfolio_year": int(port_year),
            "cfo_t1_source": cfo_t1_source,
            "specification_label": specification_label,
            "reference_sample_label": window_row_filter_label,
            "window_rows_initial": int(len(window_df)),
            "window_firms_initial": int(window_df["Ticker"].nunique()),
            "window_training_years_initial": int(
                window_df.loc[~window_df["is_portfolio_year"].astype(bool), "Year"].nunique()
            ),
        }

        if window_row_filter_by_year is not None:
            window_df, reference_diag = _filter_window_to_reference_rows(
                window_df=window_df,
                portfolio_year=int(port_year),
                row_filter_by_year=window_row_filter_by_year,
            )
            year_diag.update(reference_diag)
            print(
                "Reference sample filter: "
                f"{reference_diag['window_rows_before_reference_sample_filter']} -> "
                f"{reference_diag['window_rows_after_reference_sample_filter']} rows"
            )
            if window_df.empty:
                year_diag["skipped"] = True
                year_diag["skip_reason"] = "no_reference_sample_rows"
                window_diagnostics.append(year_diag)
                print("SKIPPED — no rows in matched reference sample")
                continue

        if cfo_t1_source == "external":
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

            year_diag.update(
                {
                    "rows_dropped_missing_analyst_forecast": 0,
                    "rows_dropped_missing_realized_cfo_lead": 0,
                    "window_rows_after_cfo_source_filter": int(len(window_df_fixed)),
                    "n_ar1_observed_transitions": int(cfo_info["n_ar1_obs"]),
                    "n_ar1_predicted_rows": int(cfo_info["n_predict"]),
                }
            )

        elif cfo_t1_source in {"realized", "analyst", "hybrid", "none"}:
            if cfo_t1_source == "none":
                print("Skipping CFO_{t+1}: cfo_t1_source='none'")
            else:
                print(f"Using CFO_t+1 source: {cfo_t1_source}")

            window_df_fixed, source_diag, include_cfo_lead = apply_cfo_t1_source_to_window(
                window_df=window_df,
                cfo_t1_source=cfo_t1_source,
            )
            year_diag.update(source_diag)

        else:
            raise ValueError(f"Unknown cfo_t1_source: {cfo_t1_source}")

        window_df_fixed, portfolio_firm_diag = keep_firms_with_portfolio_year_rows(window_df_fixed)
        year_diag.update(portfolio_firm_diag)
        if window_df_fixed.empty:
            year_diag["skipped"] = True
            year_diag["skip_reason"] = "no_portfolio_year_rows_after_filters"
            year_diag["portfolio_year_rows_final"] = 0
            window_diagnostics.append(year_diag)
            print("SKIPPED — no portfolio-year rows after CFO/sample filters")
            continue

        window_df_fixed, training_diag = enforce_training_requirements_after_cfo_filter(
            window_df=window_df_fixed,
            min_train_years=min_train_years,
            min_firm_obs=3,
        )
        year_diag.update(training_diag)

        if window_df_fixed is None:
            year_diag["skipped"] = True
            year_diag["skip_reason"] = "insufficient_training_history_after_cfo_source_filter"
            window_diagnostics.append(year_diag)
            print("SKIPPED — insufficient training data after CFO_t+1 source filter")
            continue

        portfolio_year_rows_final = int(window_df_fixed["is_portfolio_year"].astype(bool).sum())
        if portfolio_year_rows_final == 0:
            year_diag["skipped"] = True
            year_diag["skip_reason"] = "no_portfolio_year_rows_after_filters"
            year_diag["portfolio_year_rows_final"] = 0
            window_diagnostics.append(year_diag)
            print("SKIPPED — no portfolio-year rows after CFO/sample filters")
            continue

        year_diag["skipped"] = False
        year_diag["window_rows_final"] = int(len(window_df_fixed))
        year_diag["window_firms_final"] = int(window_df_fixed["Ticker"].nunique())
        year_diag["portfolio_year_rows_final"] = portfolio_year_rows_final
        window_diagnostics.append(year_diag)

        # --------------------------------------------------
        # Stage 2: accrual model
        # --------------------------------------------------
        try:
            model, trace_info = build_hb_accrual_model_fixed_lead(
                window_df_fixed,
                firm_sector_map=firm_sector,
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

        sigma_sd_conv = summarize_convergence(trace, var_name="sigma_firm_sd")
        sigma_scale_conv = summarize_convergence(trace, var_name="sigma_firm")
        alpha_conv = summarize_convergence(trace, var_name="alpha_firm")

        n_divergent = sigma_sd_conv["n_divergent"]
        rhat_sigma_sd = sigma_sd_conv["max_rhat"]
        ess_sigma_sd_bulk = sigma_sd_conv["min_ess_bulk"]
        ess_sigma_sd_tail = sigma_sd_conv["min_ess_tail"]

        rhat_sigma_scale = sigma_scale_conv["max_rhat"]
        ess_sigma_scale_bulk = sigma_scale_conv["min_ess_bulk"]
        ess_sigma_scale_tail = sigma_scale_conv["min_ess_tail"]

        rhat_alpha = alpha_conv["max_rhat"]
        ess_alpha_bulk = alpha_conv["min_ess_bulk"]

        print(f"Divergences:          {n_divergent}")
        print(
            "σ_firm_sd  R̂ / ESS(b) / ESS(t):  "
            f"{rhat_sigma_sd:.3f} / {ess_sigma_sd_bulk:.0f} / {ess_sigma_sd_tail:.0f}"
        )
        print(
            "σ_firm scale R̂ / ESS(b) / ESS(t): "
            f"{rhat_sigma_scale:.3f} / {ess_sigma_scale_bulk:.0f} / {ess_sigma_scale_tail:.0f}"
        )
        print(f"α_firm  R̂ / ESS(b):            {rhat_alpha:.3f} / {ess_alpha_bulk:.0f}")

        max_rhat = max(rhat_sigma_sd, rhat_sigma_scale, rhat_alpha)
        if n_divergent > 0:
            print(f"⚠ {n_divergent} divergences — consider raising target_accept")
        if max_rhat > 1.05:
            print("✗ R̂ > 1.05 — DO NOT USE these results, chains did not converge")
        elif max_rhat > 1.01:
            print("⚠ R̂ > 1.01 — investigate convergence before trusting")
        else:
            print("✓ Convergence good")
        if min(
            ess_sigma_sd_bulk,
            ess_sigma_sd_tail,
            ess_sigma_scale_bulk,
            ess_sigma_scale_tail,
            ess_alpha_bulk,
        ) < 400:
            print("⚠ ESS < 400 for some parameter — credible intervals will be noisy")

        year_results_all = extract_sigma_posteriors(
            trace,
            trace_info,
            var_name="sigma_firm_sd",
        )
        year_results_scale_all = extract_sigma_posteriors(
            trace,
            trace_info,
            var_name="sigma_firm",
        )
        portfolio_firm_indices = set(
            window_df_fixed.loc[
                window_df_fixed["is_portfolio_year"].astype(bool),
                "firm_idx",
            ].astype(int)
        )
        year_results = {
            firm_idx: draws
            for firm_idx, draws in year_results_all.items()
            if int(firm_idx) in portfolio_firm_indices
        }
        year_results_scale = {
            firm_idx: draws
            for firm_idx, draws in year_results_scale_all.items()
            if int(firm_idx) in portfolio_firm_indices
        }
        all_results[port_year] = year_results
        all_results_scale[port_year] = year_results_scale

        expected_accruals = extract_expected_accrual_summary(
            trace=trace,
            trace_info=trace_info,
            window_df=window_df_fixed,
            portfolio_year=int(port_year),
            cfo_t1_source=cfo_t1_source,
        )
        if not expected_accruals.empty:
            expected_accrual_frames.append(expected_accruals)

        window_rows_used_frames.append(
            _window_rows_used_frame(
                window_df=window_df_fixed,
                portfolio_year=int(port_year),
                cfo_t1_source=cfo_t1_source,
                specification_label=specification_label,
                reference_sample_label=window_row_filter_label,
            )
        )

        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "sigma_measure": "student_t_residual_sd",
                    "sigma_scale_measure": "student_t_scale",
                    "sigma_sd": {port_year: year_results},
                    "sigma_scale": {port_year: year_results_scale},
                },
                f,
            )
        print(f"Checkpoint saved: {checkpoint_path.name}")

        last_model = model
        last_trace = trace
        last_window_df = window_df_fixed.copy()

    print(f"\nDone! Estimated {len(all_results)} portfolio years.")

    all_results_path = output_dir / "hb_all_results.pkl"
    with open(all_results_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Consolidated results saved to {all_results_path}")

    all_results_scale_path = output_dir / "hb_all_results_sigma_scale.pkl"
    with open(all_results_scale_path, "wb") as f:
        pickle.dump(all_results_scale, f)
    print(f"Consolidated scale audit results saved to {all_results_scale_path}")

    sigma_summary = build_sigma_summary(all_results, firm_map)
    sigma_summary["sigma_measure"] = "student_t_residual_sd"

    sigma_scale_summary = build_sigma_summary(all_results_scale, firm_map)
    if not sigma_scale_summary.empty:
        sigma_scale_summary = _rename_sigma_summary_columns(
            sigma_scale_summary,
            prefix="sigma_scale",
        )
        sigma_summary = sigma_summary.merge(
            sigma_scale_summary,
            on=["Year", "Ticker", "firm_idx"],
            how="left",
            validate="1:1",
        )

    sigma_summary_path = output_dir / "sigma_posteriors_summary.csv"
    sigma_summary.to_csv(sigma_summary_path, index=False)

    print(f"Saved summary: {sigma_summary_path}")
    if not sigma_summary.empty:
        print(
            f"{len(sigma_summary)} firm-year estimates, "
            f"{sigma_summary['Ticker'].nunique()} unique firms, "
            f"{sigma_summary['Year'].min()}–{sigma_summary['Year'].max()}"
        )
        print("\nPosterior mean σ_i distribution (Student-t residual SD):")
        print(sigma_summary["sigma_mean"].describe().round(4).to_string())
        if "sigma_scale_mean" in sigma_summary.columns:
            print("\nPosterior mean σ_i distribution (Student-t scale audit):")
            print(sigma_summary["sigma_scale_mean"].describe().round(4).to_string())

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
        print(
            "Saved full posteriors: "
            f"{full_post_path} (Student-t residual SD used by Step 3)"
        )
        print(f"Shape: {sigma_full.shape} ({sigma_full.shape[1] - 3} draws per firm-year)")

    window_diag_df = pd.DataFrame(window_diagnostics)
    window_diag_path = output_dir / "hb_window_diagnostics.csv"
    window_diag_df.to_csv(window_diag_path, index=False)
    print(f"Saved window diagnostics: {window_diag_path}")

    window_rows_used_path = output_dir / "hb_window_rows_used.csv"
    if window_rows_used_frames:
        window_rows_used_df = pd.concat(window_rows_used_frames, ignore_index=True)
        window_rows_used_df.to_csv(window_rows_used_path, index=False)
        print(f"Saved window rows used: {window_rows_used_path}")
    elif not window_rows_used_path.exists():
        pd.DataFrame(
            columns=[
                "portfolio_year",
                "specification_label",
                "cfo_t1_source",
                "reference_sample_label",
                "Ticker",
                "Year",
                "is_portfolio_year",
            ]
        ).to_csv(window_rows_used_path, index=False)
        print(f"Saved empty window rows used: {window_rows_used_path}")

    expected_accruals_path = output_dir / "expected_accruals_summary.csv"
    if expected_accrual_frames:
        expected_accruals_df = pd.concat(expected_accrual_frames, ignore_index=True)
        expected_accruals_df.to_csv(expected_accruals_path, index=False)
        print(f"Saved expected accrual summary: {expected_accruals_path}")
    else:
        expected_accruals_df = pd.DataFrame()
        expected_accruals_df.to_csv(expected_accruals_path, index=False)
        print(f"Saved empty expected accrual summary: {expected_accruals_path}")

    sigma_merged = data.merge(
        sigma_summary,
        on=["Ticker", "Year"],
        how="inner",
        validate="1:1",
    ).copy()

    sigma_merged["sigma_acc"] = sigma_merged["sigma_mean"]
    sigma_merged["sigma_acc_measure"] = "student_t_residual_sd"

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
        "cfo_t1_source": cfo_t1_source,
        "use_analyst_cfo_forecast": bool(use_analyst_cfo_forecast),
        "analyst_cfo_forecast_csv": str(analyst_cfo_forecast_csv) if analyst_cfo_forecast_csv is not None else None,
        "specification_label": specification_label,
        "window_row_filter_label": window_row_filter_label,
        "analyst_merge_diagnostics": analyst_merge_diagnostics,
        "rows_dropped_missing_analyst_forecast_across_windows": (
            int(window_diag_df.get("rows_dropped_missing_analyst_forecast", pd.Series(dtype=float)).sum())
            if not window_diag_df.empty else 0
        ),
        "rows_dropped_missing_realized_cfo_lead_across_windows": (
            int(window_diag_df.get("rows_dropped_missing_realized_cfo_lead", pd.Series(dtype=float)).sum())
            if not window_diag_df.empty else 0
        ),
        "rows_removed_insufficient_training_after_cfo_filter_across_windows": (
            int(window_diag_df.get("rows_removed_after_cfo_filter_insufficient_training", pd.Series(dtype=float)).sum())
            if not window_diag_df.empty else 0
        ),
        "n_portfolio_years_completed": len(all_results),
        "n_sigma_rows": int(len(sigma_summary)),
        "sigma_acc_measure": "student_t_residual_sd",
        "sigma_scale_audit_columns": bool("sigma_scale_mean" in sigma_summary.columns),
        "full_posterior_parquet": str(full_post_path) if full_post_path is not None else None,
        "scale_audit_results_pkl": str(all_results_scale_path),
        "window_diagnostics_csv": str(window_diag_path),
        "window_rows_used_csv": str(window_rows_used_path),
        "expected_accruals_summary_csv": str(expected_accruals_path),
    }

    config_path = output_dir / "uncertainty_model_hb_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "firm_year_csv": str(merged_output_path),
        "sigma_summary_csv": str(sigma_summary_path),
        "all_results_pkl": str(all_results_path),
        "scale_audit_results_pkl": str(all_results_scale_path),
        "full_posterior_parquet": str(full_post_path) if full_post_path is not None else None,
        "window_diagnostics_csv": str(window_diag_path),
        "window_rows_used_csv": str(window_rows_used_path),
        "expected_accruals_summary_csv": str(expected_accruals_path),
        "analyst_cfo_forecast_yearly_coverage_csv": (
            analyst_merge_diagnostics.get("yearly_coverage_csv")
            if analyst_merge_diagnostics is not None else None
        ),
        "analyst_cfo_forecast_merge_diagnostics_json": (
            analyst_merge_diagnostics.get("diagnostics_json")
            if analyst_merge_diagnostics is not None else None
        ),
        "config_json": str(config_path),
        "plots_dir": str(plot_dir) if save_plots else None,
        "ppc_plot_png": ppc_plot_path,
    }


def _read_expected_accruals(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df[df["is_portfolio_year"].astype(bool)].copy()
    return df


def create_hb_specification_comparison(
    baseline_result: dict,
    analyst_result: dict,
    output_dir: Path,
    baseline_label: str = "no_cfo_lead",
    analyst_label: str = "analyst_cfo",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_fy = pd.read_csv(baseline_result["firm_year_csv"])
    analyst_fy = pd.read_csv(analyst_result["firm_year_csv"])
    baseline_exp = _read_expected_accruals(baseline_result.get("expected_accruals_summary_csv"))
    analyst_exp = _read_expected_accruals(analyst_result.get("expected_accruals_summary_csv"))

    if not baseline_exp.empty:
        baseline_exp = baseline_exp.rename(columns={"expected_wca_mean": "expected_wca_baseline"})
    if not analyst_exp.empty:
        analyst_exp = analyst_exp.rename(columns={"expected_wca_mean": "expected_wca_analyst_cfo"})

    baseline_counts = baseline_fy.groupby("Year").agg(
        baseline_firms=("Ticker", "nunique"),
        baseline_firm_years=("Ticker", "size"),
    )
    analyst_counts = analyst_fy.groupby("Year").agg(
        analyst_cfo_firms=("Ticker", "nunique"),
        analyst_cfo_firm_years=("Ticker", "size"),
    )

    years = sorted(set(baseline_counts.index).union(set(analyst_counts.index)))
    yearly = pd.DataFrame(index=years)
    yearly = yearly.join(baseline_counts, how="left").join(analyst_counts, how="left")

    overlap_rows = []
    for year in years:
        b_firms = set(baseline_fy.loc[baseline_fy["Year"] == year, "Ticker"])
        a_firms = set(analyst_fy.loc[analyst_fy["Year"] == year, "Ticker"])
        overlap_rows.append({"Year": year, "overlap_firms": len(b_firms & a_firms)})
    yearly = yearly.reset_index(names="Year").merge(pd.DataFrame(overlap_rows), on="Year", how="left")

    if not baseline_exp.empty:
        b_mean = baseline_exp.groupby("Year")["expected_wca_baseline"].mean().rename("mean_expected_wca_baseline")
        yearly = yearly.merge(b_mean.reset_index(), on="Year", how="left")
    if not analyst_exp.empty:
        a_mean = analyst_exp.groupby("Year")["expected_wca_analyst_cfo"].mean().rename("mean_expected_wca_analyst_cfo")
        yearly = yearly.merge(a_mean.reset_index(), on="Year", how="left")

    if not baseline_exp.empty and not analyst_exp.empty:
        overlap_exp = baseline_exp[["Year", "Ticker", "expected_wca_baseline"]].merge(
            analyst_exp[["Year", "Ticker", "expected_wca_analyst_cfo"]],
            on=["Year", "Ticker"],
            how="inner",
        )
        overlap_exp["expected_wca_difference_analyst_minus_baseline"] = (
            overlap_exp["expected_wca_analyst_cfo"] - overlap_exp["expected_wca_baseline"]
        )
        yearly_stats = (
            overlap_exp.groupby("Year")
            .agg(
                expected_wca_overlap_obs=("Ticker", "size"),
                expected_wca_overlap_corr=(
                    "expected_wca_baseline",
                    lambda s: s.corr(
                        overlap_exp.loc[s.index, "expected_wca_analyst_cfo"]
                    ),
                ),
                mean_expected_wca_difference_analyst_minus_baseline=(
                    "expected_wca_difference_analyst_minus_baseline",
                    "mean",
                ),
            )
            .reset_index()
        )
        yearly = yearly.merge(yearly_stats, on="Year", how="left")
    else:
        overlap_exp = pd.DataFrame()

    yearly.insert(1, "baseline_model", baseline_label)
    yearly.insert(2, "comparison_model", analyst_label)

    comparison_name = f"hb_{baseline_label}_vs_{analyst_label}"
    yearly_path = output_dir / f"{comparison_name}_by_year.csv"
    yearly.to_csv(yearly_path, index=False)

    def _distribution_rows(label: str, df: pd.DataFrame, value_col: str, sample: str) -> dict:
        if df.empty or value_col not in df:
            return {
                "model": label,
                "sample": sample,
                "n_firm_years": 0,
                "n_firms": 0,
            }
        x = pd.to_numeric(df[value_col], errors="coerce").dropna()
        return {
            "model": label,
            "sample": sample,
            "n_firm_years": int(len(df)),
            "n_firms": int(df["Ticker"].nunique()),
            "mean_expected_wca": float(x.mean()) if len(x) else np.nan,
            "median_expected_wca": float(x.median()) if len(x) else np.nan,
            "std_expected_wca": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "p05_expected_wca": float(x.quantile(0.05)) if len(x) else np.nan,
            "p25_expected_wca": float(x.quantile(0.25)) if len(x) else np.nan,
            "p75_expected_wca": float(x.quantile(0.75)) if len(x) else np.nan,
            "p95_expected_wca": float(x.quantile(0.95)) if len(x) else np.nan,
        }

    overall_rows = [
        _distribution_rows(baseline_label, baseline_exp, "expected_wca_baseline", "full_available_sample"),
        _distribution_rows(analyst_label, analyst_exp, "expected_wca_analyst_cfo", "full_available_sample"),
    ]
    if not overlap_exp.empty:
        overall_rows.extend(
            [
                _distribution_rows(
                    baseline_label,
                    overlap_exp.rename(columns={"expected_wca_baseline": "expected_wca"}),
                    "expected_wca",
                    "overlapping_firm_year_sample",
                ),
                _distribution_rows(
                    analyst_label,
                    overlap_exp.rename(columns={"expected_wca_analyst_cfo": "expected_wca"}),
                    "expected_wca",
                    "overlapping_firm_year_sample",
                ),
                {
                    "model": f"{baseline_label}_vs_{analyst_label}",
                    "sample": "overlapping_firm_year_sample",
                    "n_firm_years": int(len(overlap_exp)),
                    "n_firms": int(overlap_exp["Ticker"].nunique()),
                    "correlation_expected_wca": float(
                        overlap_exp["expected_wca_baseline"].corr(overlap_exp["expected_wca_analyst_cfo"])
                    ),
                    "mean_difference_analyst_minus_baseline": float(
                        overlap_exp["expected_wca_difference_analyst_minus_baseline"].mean()
                    ),
                    "median_difference_analyst_minus_baseline": float(
                        overlap_exp["expected_wca_difference_analyst_minus_baseline"].median()
                    ),
                    "std_difference_analyst_minus_baseline": float(
                        overlap_exp["expected_wca_difference_analyst_minus_baseline"].std(ddof=1)
                    ),
                },
            ]
        )

    overall = pd.DataFrame(overall_rows)
    overall_path = output_dir / f"{comparison_name}_overall_summary.csv"
    overall.to_csv(overall_path, index=False)

    overlap_path = output_dir / f"{comparison_name}_overlap_firm_years.csv"
    overlap_exp.to_csv(overlap_path, index=False)

    return {
        "comparison_by_year_csv": str(yearly_path),
        "comparison_overall_summary_csv": str(overall_path),
        "comparison_overlap_firm_years_csv": str(overlap_path),
    }


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
    cfo_lead_mode: str = "none",
    use_analyst_cfo_forecast: bool = False,
    cfo_t1_source: str | None = None,
    analyst_cfo_forecast_csv: str | Path | None = None,
    run_model_specification: str = "baseline",
) -> dict:
    spec = run_model_specification.lower().replace("-", "_")
    if spec == "analystcfo":
        spec = "analyst_cfo"
    if spec not in {"baseline", "analyst_cfo", "both"}:
        raise ValueError("run_model_specification must be 'baseline', 'analyst_cfo', or 'both'.")

    common_kwargs = {
        "input_csv": input_csv,
        "model_name": model_name,
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
        "analyst_cfo_forecast_csv": analyst_cfo_forecast_csv,
    }

    output_dir = Path(output_dir)
    if spec == "baseline":
        source = cfo_t1_source
        if source is None and use_analyst_cfo_forecast:
            source = "hybrid"
        source_canon = _canonical_cfo_source(
            source,
            cfo_lead_mode=cfo_lead_mode,
            use_analyst=use_analyst_cfo_forecast,
        )
        label = "baseline"
        if source_canon == "hybrid":
            label = "analyst_cfo_hybrid"
        elif source_canon == "analyst":
            label = "analyst_cfo"
        return _run_uncertainty_model_hb_single(
            output_dir=output_dir,
            cfo_t1_source=source_canon,
            use_analyst_cfo_forecast=use_analyst_cfo_forecast,
            specification_label=label,
            **common_kwargs,
        )

    if spec == "analyst_cfo":
        source = cfo_t1_source or "hybrid"
        source_canon = _canonical_cfo_source(source, cfo_lead_mode=cfo_lead_mode, use_analyst=True)
        if source_canon not in {"analyst", "hybrid"}:
            raise ValueError(
                "run_model_specification='analyst_cfo' requires cfo_t1_source='hybrid', "
                "cfo_t1_source='analyst', or no explicit cfo_t1_source."
            )
        return _run_uncertainty_model_hb_single(
            output_dir=output_dir,
            cfo_t1_source=source_canon,
            use_analyst_cfo_forecast=True,
            specification_label="analyst_cfo_hybrid" if source_canon == "hybrid" else "analyst_cfo",
            **common_kwargs,
        )

    baseline_dir = output_dir / "hb_results_no_cfo_lead_matched_sample"
    analyst_dir = output_dir / "hb_results_analyst_cfo"
    comparison_dir = output_dir / "hb_results_comparison"

    analyst_result = _run_uncertainty_model_hb_single(
        output_dir=analyst_dir,
        cfo_t1_source="hybrid",
        use_analyst_cfo_forecast=True,
        specification_label="analyst_cfo_hybrid",
        **common_kwargs,
    )
    baseline_result = _run_uncertainty_model_hb_single(
        output_dir=baseline_dir,
        cfo_t1_source="none",
        use_analyst_cfo_forecast=False,
        specification_label="no_cfo_lead_matched_sample",
        window_row_filter_by_year=analyst_result.get("window_rows_used_csv"),
        window_row_filter_label="analyst_cfo_hybrid_window_rows",
        **common_kwargs,
    )
    comparison_result = create_hb_specification_comparison(
        baseline_result=baseline_result,
        analyst_result=analyst_result,
        output_dir=comparison_dir,
        baseline_label="no_cfo_lead_matched_sample",
        analyst_label="analyst_cfo_hybrid",
    )

    return {
        "output_dir": str(output_dir),
        "firm_year_csv": analyst_result["firm_year_csv"],
        "full_posterior_parquet": analyst_result.get("full_posterior_parquet"),
        "sigma_summary_csv": analyst_result.get("sigma_summary_csv"),
        "all_results_pkl": analyst_result.get("all_results_pkl"),
        "config_json": analyst_result.get("config_json"),
        "selected_specification_for_downstream": "analyst_cfo_hybrid",
        "baseline": baseline_result,
        "no_cfo_lead_matched_sample": baseline_result,
        "analyst_cfo": analyst_result,
        "analyst_cfo_hybrid": analyst_result,
        "comparison": comparison_result,
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
        default="none",
        choices=["best_external", "none"],
        help=(
            "Legacy CFO_{t+1} handling when --cfo_t1_source is unset. "
            "Default 'none' matches the no-look-ahead main specification."
        ),
    )
    parser.add_argument(
        "--cfo_t1_source",
        type=str,
        default=None,
        choices=["realized", "realised", "analyst", "analyst_cfo", "hybrid", "external", "none"],
        help=(
            "Source for CFO_{t+1}: none, analyst, hybrid, external, or explicit realized. "
            "Use realized only for diagnostics/backtests because it can create look-ahead bias "
            "in portfolio-year rows. Hybrid uses realized CFO_{t+1} in training rows and "
            "analyst forecasts in portfolio-year rows."
        ),
    )
    parser.add_argument(
        "--use_analyst_cfo_forecast",
        action="store_true",
        help="Use analyst CFO forecasts for CFO_{t+1}; defaults to hybrid source handling.",
    )
    parser.add_argument(
        "--analyst_cfo_forecast_csv",
        type=str,
        default=None,
        help=(
            "Path to cfo_forecast_complete_cases_no_gaps_until_2024.csv. "
            "Defaults to modelling/analysis/cfo_forecast_complete_cases_no_gaps_until_2024.csv."
        ),
    )
    parser.add_argument(
        "--run_model_specification",
        type=str,
        default="baseline",
        choices=["baseline", "analyst_cfo", "analystcfo", "both"],
        help=(
            "Run baseline only, analyst-CFO only, or both with comparison tables. "
            "'analyst_cfo' defaults to hybrid CFO handling; 'both' compares analyst-CFO "
            "hybrid with a no-lead HB model matched to the analyst-CFO estimation sample."
        ),
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
        cfo_t1_source=args.cfo_t1_source,
        use_analyst_cfo_forecast=args.use_analyst_cfo_forecast,
        analyst_cfo_forecast_csv=args.analyst_cfo_forecast_csv,
        run_model_specification=args.run_model_specification,
    )
