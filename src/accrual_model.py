"""
accrual_model.py
----------------
Constructs working capital accruals (WCA) and estimates the McNichols (2002)
accrual regression with a rolling window.

Key design decisions (per methodology script):
- WCA = (dCA - dCash) - (dCL - dSTD - dTP)
- McNichols regression: WCA/A = alpha + b1*CFO_{t-1} + b2*CFO_t + b3*CFO_{t+1}
                                      + b4*dREV + b5*PPE + epsilon
- CFO_{t+1} = NA for portfolio year t (no look-ahead bias)
- Rolling window: expanding 3-5 years, then 5-year rolling
- OLS residuals used as robustness check (Robustness 1)
- RMSE = sqrt(mean(epsilon^2)) — not std — as noise measure
"""

import logging
import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1: Construct WCA
# ---------------------------------------------------------------------------

def build_wca(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct working capital accruals per firm-year.

    WCA = (delta_CA - delta_Cash) - (delta_CL - delta_STD - delta_TP)

    All components scaled by lagged total assets (AT_{t-1}).

    Parameters
    ----------
    data : merged DataFrame with columns ACT, CHE, LCT, STD, TXP, AT

    Returns
    -------
    DataFrame with additional columns:
        WCA       : unscaled working capital accruals
        AT_lag    : lagged total assets (scaling denominator)
        WCA_scaled: WCA / AT_lag
        CFO_scaled: OANCF / AT_lag (operating cash flow, scaled)
        dREV_scaled: change in revenue / AT_lag
        PPE_scaled : PPEGT / AT_lag
    """
    df = data.copy().sort_values(["Ticker", "Year"])

    # Year-over-year changes within each firm
    grp = df.groupby("Ticker")

    df["dCA"]   = grp["ACT"].diff()
    df["dCash"] = grp["CHE"].diff()
    df["dCL"]   = grp["LCT"].diff()
    df["dSTD"]  = grp["STD"].diff()
    df["dTP"]   = grp["TXP"].diff()

    # WCA (unscaled)
    df["WCA"] = (df["dCA"] - df["dCash"]) - (df["dCL"] - df["dSTD"] - df["dTP"])

    # Lagged total assets (scaling denominator)
    df["AT_lag"] = grp["AT"].shift(1)

    # Drop rows where AT_lag is missing or zero (first year per firm, or bad data)
    n_before = len(df)
    df = df[df["AT_lag"].notna() & (df["AT_lag"] > 0)]
    n_after = len(df)
    if n_before - n_after > 0:
        log.info("WCA construction: dropped %d rows (missing or zero AT_lag).",
                 n_before - n_after)

    # Scaled variables
    df["WCA_scaled"]  = df["WCA"]   / df["AT_lag"]
    df["CFO_scaled"]  = df["OANCF"] / df["AT_lag"]
    df["PPE_scaled"]  = df["PPEGT"] / df["AT_lag"]

    # Change in revenue scaled — use REVT if available
    if "REVT" in df.columns:
        df["dREV"]        = grp["REVT"].diff().reindex(df.index)
        df["dREV_scaled"] = df["dREV"] / df["AT_lag"]
    else:
        log.warning("REVT not found — dREV_scaled will be zero-filled.")
        df["dREV_scaled"] = 0.0

    log.info("WCA construction complete: %d firm-years.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: OLS McNichols regression (single window)
# ---------------------------------------------------------------------------

def _run_ols_mcnichols(train_df: pd.DataFrame,
                        portfolio_row: pd.Series) -> dict:
    """
    Run McNichols OLS on training data, return residual for portfolio year.

    Training years (t-4 to t-1): full McNichols with CFO_{t+1} observed.
    Portfolio year t: CFO_{t+1} = NA — excluded from RHS.

    Returns dict with keys: residual, rmse, n_train, beta_names
    """
    # Build lagged/lead CFO within the training window
    train = train_df.copy().sort_values("Year")
    train["CFO_lag1"] = train["CFO_scaled"].shift(1)
    train["CFO_lead1"] = train["CFO_scaled"].shift(-1)
    train = train.dropna(subset=["WCA_scaled", "CFO_lag1", "CFO_scaled",
                                  "CFO_lead1", "dREV_scaled", "PPE_scaled"])

    if len(train) < 5:
        return None

    X_train = train[["CFO_lag1", "CFO_scaled", "CFO_lead1",
                      "dREV_scaled", "PPE_scaled"]]
    X_train = add_constant(X_train, has_constant="add")
    y_train = train["WCA_scaled"]

    try:
        model   = OLS(y_train, X_train).fit()
    except Exception as e:
        log.debug("OLS failed: %s", e)
        return None

    # In-sample RMSE (note: RMSE not std — captures magnitude + variability)
    resid_train = model.resid
    rmse        = np.sqrt(np.mean(resid_train ** 2))

    # Predict for portfolio year (CFO_{t+1} = NA, so set to 0 as placeholder
    # but mark as excluded — residual absorbs the missing info intentionally)
    row = portfolio_row.copy()

    # Retrieve lagged CFO from training data
    cfo_lag1 = train["CFO_scaled"].iloc[-1] if len(train) > 0 else np.nan

    X_port = np.array([[1.0,
                         cfo_lag1,
                         row["CFO_scaled"],
                         0.0,   # CFO_{t+1} = NA treated as marginalised
                         row["dREV_scaled"],
                         row["PPE_scaled"]]])

    predicted       = X_port @ model.params.values
    residual        = row["WCA_scaled"] - predicted[0]

    return {
        "residual"    : residual,
        "rmse_train"  : rmse,
        "n_train"     : len(train),
        "predicted"   : predicted[0],
    }


# ---------------------------------------------------------------------------
# Step 3: Rolling OLS RMSE (Robustness 1)
# ---------------------------------------------------------------------------

def compute_ols_rmse(data: pd.DataFrame,
                     min_train_years: int = 3,
                     max_train_years: int = 5) -> pd.DataFrame:
    """
    Compute OLS-based accrual noise measure (Robustness 1).

    For each firm-year (portfolio year t):
    - Training window: expanding 3–5 years, then rolling 5-year
    - Run McNichols OLS on training years
    - sigma_OLS = RMSE of in-sample residuals
    - Also stores the out-of-sample residual for year t

    Parameters
    ----------
    data             : output of build_wca()
    min_train_years  : minimum years required to estimate (default 3)
    max_train_years  : rolling window size once warm-up complete (default 5)

    Returns
    -------
    DataFrame with columns: Ticker, Year, sigma_OLS, residual_t, n_train
    """
    records = []

    for ticker, firm_df in data.groupby("Ticker"):
        firm_df = firm_df.sort_values("Year").reset_index(drop=True)
        years   = firm_df["Year"].tolist()

        for i, port_year in enumerate(years):
            # Portfolio year index i — need at least min_train_years before it
            if i < min_train_years:
                continue

            # Training window: expanding up to max_train_years
            train_start = max(0, i - max_train_years)
            train_idx   = list(range(train_start, i))
            train_df    = firm_df.iloc[train_idx].copy()
            port_row    = firm_df.iloc[i]

            result = _run_ols_mcnichols(train_df, port_row)
            if result is None:
                continue

            records.append({
                "Ticker"     : ticker,
                "Year"       : port_year,
                "sigma_OLS"  : result["rmse_train"],
                "residual_t" : result["residual"],
                "n_train"    : result["n_train"],
            })

    out = pd.DataFrame(records)
    log.info("OLS RMSE computed: %d firm-years.", len(out))
    return out
