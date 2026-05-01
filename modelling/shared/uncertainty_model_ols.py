# uncertainty_model_ols.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =============================================================================
# Helpers
# =============================================================================

REQUIRED_COLUMNS = [
    "Ticker",
    "Year",
    "CompanyName",
    "Industry",
    "Sector",
    "ACT",
    "CHE",
    "LCT",
    "STD",
    "TXP",
    "PPEGT",
    "AT",
    "OANCF",
    "REVT",
    "PROF",
    "MarketCap",
]


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    num_cols = [
        "ACT",
        "CHE",
        "LCT",
        "STD",
        "TXP",
        "PPEGT",
        "AT",
        "OANCF",
        "REVT",
        "PROF",
        "MarketCap",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Ticker", "Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    dupes = df.duplicated(subset=["Ticker", "Year"], keep=False)
    if dupes.any():
        raise ValueError(
            "Found duplicate Ticker-Year rows in input. "
            "uncertainty_model_ols expects one row per firm-year."
        )

    df = df.sort_values(["Ticker", "Year"]).reset_index(drop=True)
    return df


def expanding_then_rolling_std(
    x: pd.Series,
    rolling_window: int = 5,
    min_periods_start: int = 4,
) -> pd.Series:
    x = x.astype(float)
    out = []

    for i in range(len(x)):
        hist = x.iloc[: i + 1].dropna()

        if len(hist) < min_periods_start:
            out.append(np.nan)
        elif len(hist) < rolling_window:
            out.append(hist.std(ddof=1))
        else:
            out.append(hist.iloc[-rolling_window:].std(ddof=1))

    return pd.Series(out, index=x.index)


def expanding_then_rolling_mean_abs(
    x: pd.Series,
    rolling_window: int = 5,
    min_periods_start: int = 4,
) -> pd.Series:
    x = x.astype(float)
    out = []

    for i in range(len(x)):
        hist = x.iloc[: i + 1].dropna().abs()

        if len(hist) < min_periods_start:
            out.append(np.nan)
        elif len(hist) < rolling_window:
            out.append(hist.mean())
        else:
            out.append(hist.iloc[-rolling_window:].mean())

    return pd.Series(out, index=x.index)


# =============================================================================
# Core OLS logic
# =============================================================================

def run_uncertainty_model_ols(
    input_csv: str | Path,
    output_dir: str | Path,
    min_obs_per_year: int = 20,
    rolling_window: int = 5,
    min_periods_start: int = 4,
    sigma_history_start_year: int | None = 2004,
) -> dict:
    """
    Step 2 OLS uncertainty model.

    Logic kept aligned with the original notebook:
    - real-time DD / McNichols-style OLS
    - WITHOUT CFO_t+1
    - WITH dREV
    - SCALED BY average total assets: (AT_t + AT_t-1) / 2

    Expected input_csv
    ------------------
    Prepared extraction panel containing the required accounting fields,
    plus PROF and MarketCap so the output is ready for Step 3.

    Returns
    -------
    dict of saved output paths for run_main.py
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load prepared panel
    # --------------------------------------------------
    panel = pd.read_csv(input_csv)
    validate_input_columns(panel)
    panel = clean_input(panel)

    print(f"Loaded prepared panel shape: {panel.shape}")

    # --------------------------------------------------
    # 2. Basic cleaning exactly as in notebook
    # --------------------------------------------------
    # Assumption: missing STD and TXP treated as zero
    panel["STD"] = panel["STD"].fillna(0.0)
    panel["TXP"] = panel["TXP"].fillna(0.0)

    # --------------------------------------------------
    # 3. Construct DD / McNichols variables
    # --------------------------------------------------
    g = panel.groupby("Ticker", group_keys=False)

    panel["AT_lag1"] = g["AT"].shift(1)
    panel["AvgAT"] = (panel["AT"] + panel["AT_lag1"]) / 2

    panel["dACT"] = g["ACT"].diff()
    panel["dCHE"] = g["CHE"].diff()
    panel["dLCT"] = g["LCT"].diff()
    panel["dSTD"] = g["STD"].diff()
    panel["dTXP"] = g["TXP"].diff()
    panel["dREV"] = g["REVT"].diff()

    panel["WCA"] = (
        (panel["dACT"] - panel["dCHE"])
        - (panel["dLCT"] - panel["dSTD"] - panel["dTXP"])
    )

    panel["CFO_t-1"] = g["OANCF"].shift(1)
    panel["CFO_t"] = panel["OANCF"]

    panel["WCA_scaled"] = panel["WCA"] / panel["AvgAT"]
    panel["CFO_t-1_scaled"] = panel["CFO_t-1"] / panel["AvgAT"]
    panel["CFO_t_scaled"] = panel["CFO_t"] / panel["AvgAT"]
    panel["dREV_scaled"] = panel["dREV"] / panel["AvgAT"]
    panel["PPE_scaled"] = panel["PPEGT"] / panel["AvgAT"]

    # --------------------------------------------------
    # 4. Keep regression sample
    # --------------------------------------------------
    reg_cols = [
        "Ticker",
        "Year",
        "CompanyName",
        "Industry",
        "Sector",
        "WCA_scaled",
        "CFO_t-1_scaled",
        "CFO_t_scaled",
        "dREV_scaled",
        "PPE_scaled",
    ]

    reg_df = panel[reg_cols].copy()
    reg_df = reg_df.replace([np.inf, -np.inf], np.nan)

    print("\nMissing values before regression drop:")
    print(
        reg_df[
            ["WCA_scaled", "CFO_t-1_scaled", "CFO_t_scaled", "dREV_scaled", "PPE_scaled"]
        ]
        .isna()
        .sum()
    )

    reg_df = reg_df.dropna(
        subset=["WCA_scaled", "CFO_t-1_scaled", "CFO_t_scaled", "dREV_scaled", "PPE_scaled"]
    ).reset_index(drop=True)

    print(f"\nRegression sample shape after dropna: {reg_df.shape}")

    # --------------------------------------------------
    # 5. Cross-sectional OLS by year
    # --------------------------------------------------
    yearly_results = []
    residual_frames = []

    for year, sub in reg_df.groupby("Year"):
        sub = sub.copy()

        if len(sub) < min_obs_per_year:
            print(f"Skipping {year}: only {len(sub)} observations")
            continue

        y = sub["WCA_scaled"]
        X = sub[["CFO_t-1_scaled", "CFO_t_scaled", "dREV_scaled", "PPE_scaled"]]
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit(cov_type="HC3")

            yearly_results.append(
                {
                    "Year": int(year),
                    "n_obs": int(len(sub)),
                    "r2": model.rsquared,
                    "adj_r2": model.rsquared_adj,
                    "coef_const": model.params.get("const", np.nan),
                    "coef_CFO_t-1": model.params.get("CFO_t-1_scaled", np.nan),
                    "coef_CFO_t": model.params.get("CFO_t_scaled", np.nan),
                    "coef_dREV": model.params.get("dREV_scaled", np.nan),
                    "coef_PPE": model.params.get("PPE_scaled", np.nan),
                    "t_CFO_t-1": model.tvalues.get("CFO_t-1_scaled", np.nan),
                    "t_CFO_t": model.tvalues.get("CFO_t_scaled", np.nan),
                    "t_dREV": model.tvalues.get("dREV_scaled", np.nan),
                    "t_PPE": model.tvalues.get("PPE_scaled", np.nan),
                    "p_CFO_t-1": model.pvalues.get("CFO_t-1_scaled", np.nan),
                    "p_CFO_t": model.pvalues.get("CFO_t_scaled", np.nan),
                    "p_dREV": model.pvalues.get("dREV_scaled", np.nan),
                    "p_PPE": model.pvalues.get("PPE_scaled", np.nan),
                }
            )

            sub["WCA_fitted"] = model.predict(X)
            sub["dd_resid"] = y - sub["WCA_fitted"]
            residual_frames.append(sub)

        except Exception as e:
            print(f"Skipping {year} due to regression error: {e}")

    yearly_results_df = (
        pd.DataFrame(yearly_results)
        .sort_values("Year")
        .reset_index(drop=True)
    )

    if residual_frames:
        resid_df = (
            pd.concat(residual_frames, ignore_index=True)
            .sort_values(["Ticker", "Year"])
            .reset_index(drop=True)
        )
    else:
        raise ValueError("No yearly regressions were estimated successfully.")

    print("\nYearly regression summary preview:")
    print(yearly_results_df.head())

    print("\nResidual sample preview:")
    print(resid_df.head())

    # --------------------------------------------------
    # 6. Construct firm-year accounting noise.
    #    Use the same historical anchor as the HB run: for the first
    #    2009 portfolio year, the five-year training window begins in 2004.
    # --------------------------------------------------
    if sigma_history_start_year is not None and sigma_history_start_year <= 0:
        sigma_history_start_year = None

    sigma_source = resid_df["dd_resid"].copy()
    if sigma_history_start_year is not None:
        sigma_source = sigma_source.where(resid_df["Year"] >= sigma_history_start_year)
        print(f"\nOLS sigma history starts in {sigma_history_start_year}.")

    sigma_input = resid_df[["Ticker", "Year"]].copy()
    sigma_input["dd_resid_for_sigma"] = sigma_source

    resid_df["sigma_acc"] = (
        sigma_input.groupby("Ticker", group_keys=False)["dd_resid_for_sigma"]
        .apply(
            expanding_then_rolling_std,
            rolling_window=rolling_window,
            min_periods_start=min_periods_start,
        )
    )

    resid_df["sigma_acc_abs"] = (
        sigma_input.groupby("Ticker", group_keys=False)["dd_resid_for_sigma"]
        .apply(
            expanding_then_rolling_mean_abs,
            rolling_window=rolling_window,
            min_periods_start=min_periods_start,
        )
    )

    print("\nCoverage of accounting-noise measures:")
    print(resid_df[["sigma_acc", "sigma_acc_abs"]].notna().sum())

    print("\nPreview with sigma:")
    print(
        resid_df[
            ["Ticker", "Year", "WCA_scaled", "WCA_fitted", "dd_resid", "sigma_acc", "sigma_acc_abs"]
        ].head(15)
    )

    # --------------------------------------------------
    # 7. Merge residual/sigma output back to prepared panel
    #    so Step 3 gets PROF + MarketCap + sigma in one file
    # --------------------------------------------------
    keep_from_resid = [
        "Ticker",
        "Year",
        "WCA_scaled",
        "CFO_t-1_scaled",
        "CFO_t_scaled",
        "dREV_scaled",
        "PPE_scaled",
        "WCA_fitted",
        "dd_resid",
        "sigma_acc",
        "sigma_acc_abs",
    ]

    uncertainty_panel = panel.merge(
        resid_df[keep_from_resid],
        on=["Ticker", "Year"],
        how="left",
        validate="1:1",
    ).copy()

    # --------------------------------------------------
    # 8. Save outputs
    # --------------------------------------------------
    yearly_summary_path = output_dir / "yearly_ols_summary_no_lead_avgat.csv"
    residuals_path = output_dir / "firm_year_residuals_and_sigma_no_lead_avgat.csv"
    merged_output_path = output_dir / "uncertainty_firm_year.csv"
    config_path = output_dir / "uncertainty_model_ols_config.json"

    yearly_results_df.to_csv(yearly_summary_path, index=False)
    resid_df.to_csv(residuals_path, index=False)
    uncertainty_panel.to_csv(merged_output_path, index=False)

    config = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "min_obs_per_year": min_obs_per_year,
        "rolling_window": rolling_window,
        "min_periods_start": min_periods_start,
        "sigma_history_start_year": sigma_history_start_year,
        "n_rows_input": int(len(panel)),
        "n_rows_reg_sample": int(len(reg_df)),
        "n_rows_residual_output": int(len(resid_df)),
        "n_rows_merged_output": int(len(uncertainty_panel)),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved outputs to: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "firm_year_csv": str(merged_output_path),
        "residuals_csv": str(residuals_path),
        "yearly_summary_csv": str(yearly_summary_path),
        "config_json": str(config_path),
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OLS uncertainty model.")
    parser.add_argument("--input_csv", type=str, required=True, help="Prepared firm-year panel for Step 2.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save OLS outputs.")
    parser.add_argument("--min_obs_per_year", type=int, default=20)
    parser.add_argument("--rolling_window", type=int, default=5)
    parser.add_argument("--min_periods_start", type=int, default=4)
    parser.add_argument(
        "--sigma_history_start_year",
        type=int,
        default=2004,
        help=(
            "First residual year allowed to enter rolling sigma history. "
            "Use 0 or a negative value to use all available history."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_uncertainty_model_ols(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        min_obs_per_year=args.min_obs_per_year,
        rolling_window=args.rolling_window,
        min_periods_start=args.min_periods_start,
        sigma_history_start_year=args.sigma_history_start_year,
    )
