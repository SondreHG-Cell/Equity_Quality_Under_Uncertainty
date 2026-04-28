# portfolio_formation.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import argparse
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "Ticker",
    "FormationYear",
    "theta_obs",
    "theta_post_mean",
    "p_q5",
    "p_median",       # ← NEW
    "sigma_acc",
    "MarketCap",
]

METHOD_SPECS = {
    "Method1_Raw": {
        "signal_col": "theta_obs",
        "method_label": "Raw signal",
    },
    "Method2_PostMean": {
        "signal_col": "theta_post_mean",
        "method_label": "Posterior mean",
    },
    "Method3_ProbQ5": {
        "signal_col": "p_q5",
        "method_label": "Probabilistic Q5",
    },
    "Method4_ProbMedian": {             # ← NEW
        "signal_col": "p_median",
        "method_label": "P(>Median)",
    },
}


# --------------------------------------------------
# Validation / cleaning
# --------------------------------------------------

def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning before portfolio formation.
    Keeps only rows with:
    - non-missing Ticker / FormationYear
    - positive MarketCap
    """
    df = df.copy()

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["FormationYear"] = pd.to_numeric(df["FormationYear"], errors="coerce").astype("Int64")
    df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")

    for col in ["theta_obs", "theta_post_mean", "p_q5", "p_median", "sigma_acc"]:  # ← p_median NEW
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Ticker"].notna() & (df["Ticker"] != "")]
    df = df[df["FormationYear"].notna()]
    df = df[df["MarketCap"].notna() & (df["MarketCap"] > 0)]

    # drop duplicate firm-year rows if any
    dupes = df.duplicated(subset=["Ticker", "FormationYear"], keep=False)
    if dupes.any():
        raise ValueError(
            "Found duplicate Ticker-FormationYear rows in input. "
            "Make sure latent_prof_model outputs one row per firm-year."
        )

    df["FormationYear"] = df["FormationYear"].astype(int)

    return df


# --------------------------------------------------
# Portfolio assignment helpers
# --------------------------------------------------

def assign_quantile_portfolios(
    sub: pd.DataFrame,
    signal_col: str,
    n_portfolios: int,
) -> pd.DataFrame:
    """
    Assigns portfolios Q1...Qn within one FormationYear.

    Logic:
    - sort ascending by signal (and Ticker for stable tie-breaking)
    - split into n almost-equal groups
    - lowest signal -> Q1, highest signal -> Qn

    Returns the same dataframe with:
    - PortfolioNum
    - Portfolio
    """
    sub = sub.copy()

    valid = sub[signal_col].notna()
    n_valid = int(valid.sum())

    sub["PortfolioNum"] = pd.Series(pd.NA, index=sub.index, dtype="Int64")
    sub["Portfolio"] = pd.Series(pd.NA, index=sub.index, dtype="object")

    if n_valid < n_portfolios:
        return sub

    ranked = (
        sub.loc[valid, ["Ticker", signal_col]]
        .sort_values([signal_col, "Ticker"], ascending=[True, True])
        .copy()
    )

    ranked["_rank"] = np.arange(n_valid)

    # qcut on rank avoids duplicate-edge problems when signal has many ties
    ranked["PortfolioNum"] = pd.qcut(
        ranked["_rank"],
        q=n_portfolios,
        labels=range(1, n_portfolios + 1),
    ).astype(int)

    ranked["Portfolio"] = "Q" + ranked["PortfolioNum"].astype(str)

    sub.loc[ranked.index, "PortfolioNum"] = ranked["PortfolioNum"].astype("Int64")
    sub.loc[ranked.index, "Portfolio"] = ranked["Portfolio"].values

    return sub

# --------------------------------------------------
# Main formation logic
# --------------------------------------------------

def form_portfolios_for_method(
    df: pd.DataFrame,
    method_name: str,
    signal_col: str,
    n_portfolios: int = 5,
) -> pd.DataFrame:
    """
    Returns long-format portfolio assignments for one method.
    """
    pieces = []

    for year, sub in df.groupby("FormationYear", sort=True):
        sub = sub.copy()
        sub = assign_quantile_portfolios(
            sub=sub,
            signal_col=signal_col,
            n_portfolios=n_portfolios,
        )

        sub["Method"] = method_name
        sub["SignalUsed"] = signal_col
        pieces.append(sub)

    out = pd.concat(pieces, axis=0, ignore_index=True)

    keep_cols = [
        "Ticker",
        "FormationYear",
        "theta_obs",
        "theta_post_mean",
        "p_q5",
        "p_median",       # ← NEW
        "sigma_acc",
        "MarketCap",
        "Method",
        "SignalUsed",
        "PortfolioNum",
        "Portfolio",
    ]

    return out[keep_cols].copy()


def build_long_output(
    df: pd.DataFrame,
    n_portfolios: int = 5,
) -> pd.DataFrame:
    """
    Builds one long file with all four sorting methods.
    """
    method_frames = []

    for method_name, spec in METHOD_SPECS.items():
        method_df = form_portfolios_for_method(
            df=df,
            method_name=method_name,
            signal_col=spec["signal_col"],
            n_portfolios=n_portfolios,
        )
        method_frames.append(method_df)

    long_df = pd.concat(method_frames, axis=0, ignore_index=True)

    long_df["PortfolioNum"] = long_df["PortfolioNum"].astype("Int64")
    long_df = long_df.sort_values(["FormationYear", "Method", "PortfolioNum", "Ticker"]).reset_index(drop=True)

    return long_df


def build_wide_output(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the long file into one row per firm-year, with separate
    portfolio assignment and weight columns for each method.
    """
    base_cols = [
        "Ticker",
        "FormationYear",
        "theta_obs",
        "theta_post_mean",
        "p_q5",
        "p_median",       # ← NEW
        "sigma_acc",
        "MarketCap",
    ]

    wide = (
        long_df[base_cols]
        .drop_duplicates(subset=["Ticker", "FormationYear"])
        .copy()
    )

    for method_name in METHOD_SPECS.keys():
        sub = long_df[long_df["Method"] == method_name].copy()

        sub = sub.rename(
            columns={
                "PortfolioNum": f"{method_name}_PortfolioNum",
                "Portfolio": f"{method_name}_Portfolio",
            }
        )

        keep = [
            "Ticker",
            "FormationYear",
            f"{method_name}_PortfolioNum",
            f"{method_name}_Portfolio",
        ]

        wide = wide.merge(
            sub[keep],
            on=["Ticker", "FormationYear"],
            how="left",
        )

    wide = wide.sort_values(["FormationYear", "Ticker"]).reset_index(drop=True)
    return wide


def build_summary_output(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary file by FormationYear x Method x Portfolio.
    """
    valid = long_df[long_df["PortfolioNum"].notna()].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "FormationYear",
                "Method",
                "PortfolioNum",
                "Portfolio",
                "n_firms",
                "total_marketcap",
                "avg_theta_obs",
                "avg_theta_post_mean",
                "avg_p_q5",
                "avg_p_median",       # ← NEW
                "avg_sigma_acc",
            ]
        )

    summary = (
        valid.groupby(["FormationYear", "Method", "PortfolioNum", "Portfolio"], as_index=False)
        .agg(
            n_firms=("Ticker", "nunique"),
            total_marketcap=("MarketCap", "sum"),
            avg_theta_obs=("theta_obs", "mean"),
            avg_theta_post_mean=("theta_post_mean", "mean"),
            avg_p_q5=("p_q5", "mean"),
            avg_p_median=("p_median", "mean"),    # ← NEW
            avg_sigma_acc=("sigma_acc", "mean"),
        )
        .sort_values(["FormationYear", "Method", "PortfolioNum"])
        .reset_index(drop=True)
    )

    return summary


# --------------------------------------------------
# Public run function
# --------------------------------------------------

def run_portfolio_formation(
    input_csv: str | Path,
    output_dir: str | Path,
    n_portfolios: int = 5,
) -> dict:
    """
    Main function to be called from run_main.py.

    Returns:
        dict with explicit output paths
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    validate_input_columns(df)
    df = clean_input(df)

    long_df = build_long_output(df, n_portfolios=n_portfolios)
    wide_df = build_wide_output(long_df)
    summary_df = build_summary_output(long_df)

    long_path = output_dir / "portfolio_assignments_long.csv"
    wide_path = output_dir / "portfolio_assignments_wide.csv"
    summary_path = output_dir / "portfolio_formation_summary.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved long output:    {long_path}")
    print(f"Saved wide output:    {wide_path}")
    print(f"Saved summary output: {summary_path}")

    return {
        "output_dir": str(output_dir),
        "portfolio_assignments_long_csv": str(long_path),
        "portfolio_assignments_wide_csv": str(wide_path),
        "portfolio_summary_csv": str(summary_path),
    }


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Form portfolios from latent PROF model output.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to latent_prof_model output CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save portfolio formation results.",
    )
    parser.add_argument(
        "--n_portfolios",
        type=int,
        default=5,
        help="Number of portfolios to form (default=5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_portfolio_formation(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        n_portfolios=args.n_portfolios,
    )