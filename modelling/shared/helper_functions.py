# helper_functions.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import re


# =============================================================================
# Public API
# =============================================================================

METHOD_ORDER = ["Method1_Raw", "Method2_PostMean", "Method3_ProbQ5"]


def load_factor_data(factors_csv: str | Path) -> pd.DataFrame:
    """
    Load and standardise factor data.

    Expected output columns:
        MKT, SMB, HML, RMW, CMA, MOM, RF

    Notes
    -----
    - MKT should be the excess market return (Rm - Rf).
    - RF should be the monthly risk-free rate.
    - Factor inputs from Ken French are often in percent, so this function
      converts them to decimals if needed.
    - Index is converted to month-end datetime.
    """
    factors_csv = Path(factors_csv)
    df = pd.read_csv(factors_csv).copy()

    df = _standardise_date_column(df)

    # Flexible column mapping
    col_map = {}
    for c in df.columns:
        c_clean = str(c).strip().replace("-", "_").replace(" ", "").upper()

        if c_clean in {"MKT_RF", "MKTRF", "MKT"}:
            col_map[c] = "MKT"
        elif c_clean == "SMB":
            col_map[c] = "SMB"
        elif c_clean == "HML":
            col_map[c] = "HML"
        elif c_clean == "RMW":
            col_map[c] = "RMW"
        elif c_clean == "CMA":
            col_map[c] = "CMA"
        elif c_clean in {"MOM", "WML"}:
            col_map[c] = "MOM"
        elif c_clean in {"RF", "R_F"}:
            col_map[c] = "RF"

    df = df.rename(columns=col_map)

    required = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Factor file is missing required columns after renaming: {missing}"
        )

    keep = ["Date"] + required
    df = df[keep].copy()

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ken French data is often in percent form
    for c in required:
        if _looks_like_percent(df[c]):
            df[c] = df[c] / 100.0

    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
    df = df.set_index("Date")

    return df


def build_monthly_portfolio_returns(
    assignments: pd.DataFrame,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    n_portfolios: int = 5,
) -> Dict[str, object]:
    """
    Build monthly portfolio return series from firm-level assignments.

    Parameters
    ----------
    assignments : pd.DataFrame
        Expected to be the long output from portfolio_formation.py with columns:
            Ticker, FormationYear, Method, PortfolioNum, Portfolio, FormationWeight
        plus signal columns if available.
    returns : pd.DataFrame
        Monthly stock return data. Must contain:
            Ticker, Date, Return
        Flexible column names are supported.
    factors : pd.DataFrame
        Standardised factor data from load_factor_data(), indexed by Date.
    n_portfolios : int
        Number of portfolios formed in Step 4.

    Returns
    -------
    dict with:
        returns_wide     : DataFrame with MultiIndex columns (Method, Portfolio)
        ls_returns       : dict {method: Series}
        q5_returns       : dict {method: Series}
        rf               : Series
        monthly_holdings : long DataFrame of firm-month portfolio membership
    """
    assignments = _standardise_assignments(assignments)
    returns = _standardise_returns_input(returns)

    merged = assignments.merge(
        returns,
        on=["Ticker", "FormationYear"],
        how="left",
        validate="many_to_many",
    )

    # Keep only months that belong to the relevant holding year by construction
    merged = merged[merged["Date"].notna()].copy()

    # Month-specific re-normalised weights among firms with available returns
    monthly_portfolio = (
        merged.groupby(
            ["Date", "FormationYear", "Method", "PortfolioNum", "Portfolio"],
            as_index=False,
        )
        .apply(_compute_monthly_portfolio_return)
        .reset_index(drop=True)
    )

    returns_wide = (
        monthly_portfolio
        .pivot(index="Date", columns=["Method", "Portfolio"], values="PortfolioReturn")
        .sort_index()
    )

    returns_wide = _add_long_short_columns(returns_wide, n_portfolios=n_portfolios)
    returns_wide = _order_return_columns(returns_wide, n_portfolios=n_portfolios)

    rf = factors["RF"].reindex(returns_wide.index)

    qn_label = f"Q{n_portfolios}"
    ls_returns = {}
    q5_returns = {}

    for method in returns_wide.columns.get_level_values(0).unique():
        if (method, "LS") in returns_wide.columns:
            ls_returns[method] = returns_wide[(method, "LS")].copy()
        if (method, qn_label) in returns_wide.columns:
            q5_returns[method] = returns_wide[(method, qn_label)].copy()

    monthly_holdings = merged.copy()
    monthly_holdings = monthly_holdings.sort_values(
        ["Date", "Method", "PortfolioNum", "Ticker"]
    ).reset_index(drop=True)

    return {
        "returns_wide": returns_wide,
        "ls_returns": ls_returns,
        "q5_returns": q5_returns,
        "rf": rf,
        "monthly_holdings": monthly_holdings,
    }


def build_probabilistic_targets(
    assignments: pd.DataFrame,
    returns: pd.DataFrame,
    n_portfolios: int = 5,
    min_months: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build out-of-sample probabilistic targets for Method 3.

    Definition
    ----------
    y_true = 1 if the firm's realised holding-period return places it in the
             top portfolio (Qn) within the FormationYear cross-section
    y_prob = predicted probability p_q5 from Method 3

    Holding-period return
    ---------------------
    Compounded from monthly stock returns over the portfolio holding window:
        July of FormationYear through June of FormationYear + 1

    Parameters
    ----------
    assignments : pd.DataFrame
        Long assignment output from portfolio_formation.py.
    returns : pd.DataFrame
        Monthly stock return data.
    n_portfolios : int
        Number of portfolios used in the sort.
    min_months : int
        Minimum number of monthly return observations required to form a
        realised holding-period return.

    Returns
    -------
    y_true, y_prob : np.ndarray, np.ndarray
    """
    assignments = _standardise_assignments(assignments)
    returns = _standardise_returns_input(returns)

    if "p_q5" not in assignments.columns:
        raise ValueError("Assignments file must contain 'p_q5' to evaluate Method 3.")

    method3 = assignments.loc[assignments["Method"] == "Method3_ProbQ5"].copy()

    method3 = (
        method3[["Ticker", "FormationYear", "p_q5"]]
        .drop_duplicates(subset=["Ticker", "FormationYear"])
        .copy()
    )

    realised = _build_realised_holding_period_returns(
        returns=returns,
        min_months=min_months,
    )

    realised = _assign_realised_quantiles(
        realised=realised,
        value_col="HoldingPeriodReturn",
        n_portfolios=n_portfolios,
    )

    eval_df = method3.merge(
        realised,
        on=["Ticker", "FormationYear"],
        how="inner",
        validate="one_to_one",
    )

    qn_label = f"Q{n_portfolios}"
    eval_df["y_true"] = (eval_df["RealisedPortfolio"] == qn_label).astype(int)
    eval_df["y_prob"] = pd.to_numeric(eval_df["p_q5"], errors="coerce")

    eval_df = eval_df.dropna(subset=["y_true", "y_prob"]).copy()

    return eval_df["y_true"].to_numpy(), eval_df["y_prob"].to_numpy()


# =============================================================================
# Input standardisation
# =============================================================================

def _standardise_assignments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "Ticker",
        "FormationYear",
        "Method",
        "PortfolioNum",
        "Portfolio",
        "FormationWeight",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Assignments file is missing required columns: {missing}")

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["FormationYear"] = pd.to_numeric(df["FormationYear"], errors="coerce").astype("Int64")
    df["PortfolioNum"] = pd.to_numeric(df["PortfolioNum"], errors="coerce").astype("Int64")
    df["FormationWeight"] = pd.to_numeric(df["FormationWeight"], errors="coerce")

    if "p_q5" in df.columns:
        df["p_q5"] = pd.to_numeric(df["p_q5"], errors="coerce")

    df = df.dropna(subset=["Ticker", "FormationYear", "Method", "PortfolioNum", "FormationWeight"]).copy()
    df["FormationYear"] = df["FormationYear"].astype(int)

    return df


def _standardise_returns_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise monthly stock data into long monthly RETURNS format.

    Supports two input formats:

    1) Long returns format:
        Ticker, Date, Return

    2) Wide monthly price format:
        Ticker, 2004-01, 2004-02, 2004-03, ...

    For wide monthly prices, this function:
    - melts to long format
    - parses month columns as dates
    - computes simple monthly returns by ticker:
          Return_t = Price_t / Price_{t-1} - 1
    """
    df = df.copy()

    # --------------------------------------------------
    # Case 1: already long format with returns
    # --------------------------------------------------
    rename_map = {}
    for c in df.columns:
        c_clean = str(c).strip().lower()

        if c_clean == "ticker":
            rename_map[c] = "Ticker"
        elif c_clean in {"date", "month", "monthend"}:
            rename_map[c] = "Date"
        elif c_clean in {"return", "ret", "monthly_return", "monthlyret"}:
            rename_map[c] = "Return"

    df_long = df.rename(columns=rename_map).copy()

    if {"Ticker", "Date", "Return"}.issubset(df_long.columns):
        df_long["Ticker"] = df_long["Ticker"].astype(str).str.strip()
        df_long["Date"] = _parse_date_series(df_long["Date"])
        df_long["Date"] = df_long["Date"] + pd.offsets.MonthEnd(0)
        df_long["Return"] = pd.to_numeric(df_long["Return"], errors="coerce")

        df_long["FormationYear"] = np.where(
            df_long["Date"].dt.month >= 7,
            df_long["Date"].dt.year,
            df_long["Date"].dt.year - 1,
        )

        df_long = df_long.dropna(subset=["Ticker", "Date", "Return"]).copy()
        df_long["FormationYear"] = df_long["FormationYear"].astype(int)

        return df_long.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # --------------------------------------------------
    # Case 2: wide monthly price matrix
    # --------------------------------------------------
    if "Ticker" not in df.columns:
        ticker_candidates = [c for c in df.columns if str(c).strip().lower() == "ticker"]
        if not ticker_candidates:
            raise ValueError(
                "Could not find 'Ticker' column in stock price file."
            )
        df = df.rename(columns={ticker_candidates[0]: "Ticker"})

    month_cols = [c for c in df.columns if _is_month_column(c)]

    if not month_cols:
        raise ValueError(
            "Could not detect monthly date columns like '2004-01' in the stock file."
        )

    price_long = df.melt(
        id_vars="Ticker",
        value_vars=month_cols,
        var_name="Date",
        value_name="Price",
    ).copy()

    price_long["Ticker"] = price_long["Ticker"].astype(str).str.strip()
    price_long["Date"] = pd.to_datetime(price_long["Date"], format="%Y-%m", errors="coerce")
    price_long["Date"] = price_long["Date"] + pd.offsets.MonthEnd(0)
    price_long["Price"] = pd.to_numeric(price_long["Price"], errors="coerce")

    price_long = price_long.dropna(subset=["Ticker", "Date"]).copy()
    price_long = price_long.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Compute simple monthly return from prices
    price_long["Return"] = price_long.groupby("Ticker")["Price"].pct_change()

    # Drop rows where current or previous price is missing implicitly
    price_long = price_long.dropna(subset=["Return"]).copy()

    price_long["FormationYear"] = np.where(
        price_long["Date"].dt.month >= 7,
        price_long["Date"].dt.year,
        price_long["Date"].dt.year - 1,
    )
    price_long["FormationYear"] = price_long["FormationYear"].astype(int)

    return (
        price_long[["Ticker", "Date", "Return", "FormationYear"]]
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )


def _standardise_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_col = None
    for c in df.columns:
        c_clean = str(c).strip().lower()
        if c_clean in {"date", "month", "yyyymm"}:
            date_col = c
            break

    if date_col is None:
        raise ValueError("Could not find a date column in the factor file.")

    df = df.rename(columns={date_col: "Date"})
    df["Date"] = _parse_date_series(df["Date"])
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)

    return df


def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Handles common monthly date formats such as:
    - YYYY-MM-DD
    - YYYY/MM/DD
    - YYYYMM
    - YYYYMMDD
    """
    s = s.copy()

    # Try direct datetime parse first
    parsed = pd.to_datetime(s, errors="coerce")

    # Handle numeric-like YYYYMM or YYYYMMDD where direct parse may fail
    needs_fix = parsed.isna() & s.notna()
    if needs_fix.any():
        raw = s.loc[needs_fix].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

        parsed_fix = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

        mask_yyyymm = raw.str.fullmatch(r"\d{6}")
        if mask_yyyymm.any():
            parsed_fix.loc[mask_yyyymm] = pd.to_datetime(raw.loc[mask_yyyymm] + "01", format="%Y%m%d", errors="coerce")

        mask_yyyymmdd = raw.str.fullmatch(r"\d{8}")
        if mask_yyyymmdd.any():
            parsed_fix.loc[mask_yyyymmdd] = pd.to_datetime(raw.loc[mask_yyyymmdd], format="%Y%m%d", errors="coerce")

        parsed.loc[needs_fix] = parsed_fix

    return parsed


def _looks_like_percent(s: pd.Series) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return False
    return s.abs().quantile(0.90) > 1.0

def _is_month_column(col) -> bool:
    col = str(col).strip()
    return bool(re.fullmatch(r"\d{4}-\d{2}", col))

# =============================================================================
# Monthly portfolio returns
# =============================================================================

def _compute_monthly_portfolio_return(group: pd.DataFrame) -> pd.Series:
    """
    Compute one monthly portfolio return using formation weights, re-normalised
    over firms with available returns in that month.
    """
    valid = group["Return"].notna() & group["FormationWeight"].notna()
    sub = group.loc[valid, ["FormationWeight", "Return"]].copy()

    if sub.empty:
        return pd.Series({
            "PortfolioReturn": np.nan,
            "n_firms_total": group["Ticker"].nunique(),
            "n_firms_used": 0,
            "weight_sum_used": 0.0,
        })

    weight_sum = sub["FormationWeight"].sum()
    if weight_sum <= 0:
        return pd.Series({
            "PortfolioReturn": np.nan,
            "n_firms_total": group["Ticker"].nunique(),
            "n_firms_used": len(sub),
            "weight_sum_used": float(weight_sum),
        })

    w = sub["FormationWeight"] / weight_sum
    port_ret = np.sum(w * sub["Return"])

    return pd.Series({
        "PortfolioReturn": float(port_ret),
        "n_firms_total": group["Ticker"].nunique(),
        "n_firms_used": len(sub),
        "weight_sum_used": float(weight_sum),
    })


def _add_long_short_columns(
    returns_wide: pd.DataFrame,
    n_portfolios: int,
) -> pd.DataFrame:
    returns_wide = returns_wide.copy()

    q1 = "Q1"
    qn = f"Q{n_portfolios}"

    methods = returns_wide.columns.get_level_values(0).unique()

    for method in methods:
        if (method, q1) in returns_wide.columns and (method, qn) in returns_wide.columns:
            returns_wide[(method, "LS")] = returns_wide[(method, qn)] - returns_wide[(method, q1)]

    return returns_wide


def _order_return_columns(
    returns_wide: pd.DataFrame,
    n_portfolios: int,
) -> pd.DataFrame:
    portfolio_order = [f"Q{i}" for i in range(1, n_portfolios + 1)] + ["LS"]

    existing_methods = list(returns_wide.columns.get_level_values(0).unique())
    method_order = [m for m in METHOD_ORDER if m in existing_methods] + [
        m for m in existing_methods if m not in METHOD_ORDER
    ]

    ordered_cols = []
    for method in method_order:
        for p in portfolio_order:
            if (method, p) in returns_wide.columns:
                ordered_cols.append((method, p))

    return returns_wide.loc[:, ordered_cols]


# =============================================================================
# Probabilistic targets
# =============================================================================

def _build_realised_holding_period_returns(
    returns: pd.DataFrame,
    min_months: int = 9,
) -> pd.DataFrame:
    """
    Build realised firm-level holding period returns for each FormationYear.

    Since returns already have FormationYear derived from the month, grouping by
    (Ticker, FormationYear) compounds July-to-June returns automatically.
    """
    out = (
        returns.groupby(["Ticker", "FormationYear"], as_index=False)
        .agg(
            n_months=("Return", "count"),
            HoldingPeriodReturn=("Return", lambda x: (1 + x).prod() - 1),
        )
    )

    out = out[out["n_months"] >= min_months].copy()
    return out


def _assign_realised_quantiles(
    realised: pd.DataFrame,
    value_col: str,
    n_portfolios: int,
) -> pd.DataFrame:
    """
    Assign realised holding-period returns into Q1..Qn cross-sectionally by
    FormationYear.
    """
    realised = realised.copy()
    realised["RealisedPortfolioNum"] = pd.Series(pd.NA, index=realised.index, dtype="Int64")
    realised["RealisedPortfolio"] = pd.Series(pd.NA, index=realised.index, dtype="object")

    pieces = []

    for year, sub in realised.groupby("FormationYear", sort=True):
        sub = sub.copy()
        valid = sub[value_col].notna()
        n_valid = int(valid.sum())

        if n_valid < n_portfolios:
            pieces.append(sub)
            continue

        ranked = (
            sub.loc[valid, ["Ticker", value_col]]
            .sort_values([value_col, "Ticker"], ascending=[True, True])
            .copy()
        )

        ranked["_rank"] = np.arange(n_valid)
        ranked["RealisedPortfolioNum"] = pd.qcut(
            ranked["_rank"],
            q=n_portfolios,
            labels=range(1, n_portfolios + 1),
        ).astype(int)
        ranked["RealisedPortfolio"] = "Q" + ranked["RealisedPortfolioNum"].astype(str)

        sub.loc[ranked.index, "RealisedPortfolioNum"] = ranked["RealisedPortfolioNum"].astype("Int64")
        sub.loc[ranked.index, "RealisedPortfolio"] = ranked["RealisedPortfolio"].values

        pieces.append(sub)

    out = pd.concat(pieces, axis=0, ignore_index=True)
    return out