# build_factor_csv.py

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


MAX_MONTHLY_DIVIDEND_YIELD = 1.0


# =============================================================================
# Path helpers
# =============================================================================

def find_project_root() -> Path:
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()

    for p in [here] + list(here.parents):
        if (
            (p / "data" / "processed_data_lseg").exists()
            and (p / "results" / "extraction_static").exists()
        ):
            return p

    raise FileNotFoundError(
        "Could not find project root containing data/processed_data_lseg and results/extraction_static."
    )


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return project_root / p


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# Generic helpers
# =============================================================================

def parse_month_series(s: pd.Series) -> pd.Series:
    """
    Parse common monthly date formats and convert to month-end timestamps.
    Supports formats like:
      - YYYY-MM
      - YYYY-MM-DD
      - YYYYMM
    """
    s = s.astype(str).str.strip()

    parsed = pd.to_datetime(s, errors="coerce")

    needs_fix = parsed.isna() & s.notna()
    if needs_fix.any():
        raw = s.loc[needs_fix]

        parsed_fix = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

        mask_yyyymm = raw.str.fullmatch(r"\d{6}")
        if mask_yyyymm.any():
            parsed_fix.loc[mask_yyyymm] = pd.to_datetime(
                raw.loc[mask_yyyymm] + "01",
                format="%Y%m%d",
                errors="coerce",
            )

        mask_yyyy_mm = raw.str.fullmatch(r"\d{4}-\d{2}")
        if mask_yyyy_mm.any():
            parsed_fix.loc[mask_yyyy_mm] = pd.to_datetime(
                raw.loc[mask_yyyy_mm] + "-01",
                format="%Y-%m-%d",
                errors="coerce",
            )

        parsed.loc[needs_fix] = parsed_fix

    return parsed + pd.offsets.MonthEnd(0)


def safe_weighted_average(x: pd.DataFrame, value_col: str, weight_col: str) -> float:
    sub = x[[value_col, weight_col]].dropna()
    sub = sub[sub[weight_col] > 0]

    if sub.empty:
        return np.nan

    w_sum = sub[weight_col].sum()
    if w_sum <= 0:
        return np.nan

    return float(np.sum(sub[value_col] * sub[weight_col]) / w_sum)


def assign_size_bucket(s: pd.Series) -> pd.Series:
    """
    Median split: Small / Big
    """
    out = pd.Series(index=s.index, dtype="object")
    valid = s.dropna()

    if valid.empty:
        return out

    med = valid.median()
    out.loc[valid.index] = np.where(valid <= med, "S", "B")
    return out


def assign_tercile_bucket(
    s: pd.Series,
    low_label: str,
    mid_label: str,
    high_label: str,
    low_q: float = 0.30,
    high_q: float = 0.70,
) -> pd.Series:
    """
    30/40/30 split. Uses quantiles when possible; falls back to rank-based splits
    if quantiles collapse because of ties.
    """
    out = pd.Series(index=s.index, dtype="object")
    valid = s.dropna()

    if len(valid) < 3:
        return out

    q_low = valid.quantile(low_q)
    q_high = valid.quantile(high_q)

    if pd.isna(q_low) or pd.isna(q_high) or q_low >= q_high:
        pct_rank = valid.rank(method="first", pct=True)
        out.loc[valid.index] = np.where(
            pct_rank <= low_q,
            low_label,
            np.where(pct_rank >= high_q, high_label, mid_label),
        )
        return out

    out.loc[valid.index] = np.where(
        valid <= q_low,
        low_label,
        np.where(valid >= q_high, high_label, mid_label),
    )
    return out


def get_col(df: pd.DataFrame, key1: str, key2: str) -> pd.Series:
    if (key1, key2) in df.columns:
        return df[(key1, key2)]
    return pd.Series(index=df.index, dtype=float)


# =============================================================================
# Input loaders
# =============================================================================

def load_prepared_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["Ticker", "Year", "BE", "AT", "PROF"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prepared_step2_input.csv is missing required columns: {missing}")

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["BE"] = pd.to_numeric(df["BE"], errors="coerce")
    df["AT"] = pd.to_numeric(df["AT"], errors="coerce")
    df["PROF"] = pd.to_numeric(df["PROF"], errors="coerce")

    df = df.dropna(subset=["Ticker", "Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    df = (
        df.sort_values(["Ticker", "Year"])
        .drop_duplicates(subset=["Ticker", "Year"], keep="first")
        .reset_index(drop=True)
    )

    # Investment = annual total asset growth from t-1 to t
    df["INV"] = df.groupby("Ticker")["AT"].pct_change()

    # Accounting year t is used for sort year t+1 (June t+1 -> July t+1 to June t+2 style)
    # For a standard July-June holding period, sort_year Y uses accounting year Y-1.
    df["SortYear"] = df["Year"] + 1

    return df


def load_market_cap_monthly(path: Path) -> pd.DataFrame:
    """
    Load wide monthly market cap file of the form:

    Ticker,2003-03,2004-01,2004-02,...
    AAB.CO,...

    and convert it to long format:
    Ticker, Date, MarketCap, LagMarketCap
    """
    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        raise ValueError("market_cap_monthly.csv must contain a 'Ticker' column.")

    month_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(c).strip())]
    if not month_cols:
        raise ValueError(
            "Could not detect monthly columns like '2010-01' in market_cap_monthly.csv."
        )

    long_df = df.melt(
        id_vars="Ticker",
        value_vars=month_cols,
        var_name="Month",
        value_name="MarketCap",
    ).copy()

    long_df["Ticker"] = long_df["Ticker"].astype(str).str.strip()
    long_df["Date"] = parse_month_series(long_df["Month"])
    long_df["MarketCap"] = pd.to_numeric(long_df["MarketCap"], errors="coerce")

    long_df = (
        long_df.dropna(subset=["Ticker", "Date", "MarketCap"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(subset=["Ticker", "Date"], keep="first")
        .reset_index(drop=True)
    )

    long_df["LagMarketCap"] = long_df.groupby("Ticker")["MarketCap"].shift(1)

    return long_df[["Ticker", "Date", "MarketCap", "LagMarketCap"]]


def load_nibor(path: Path, method: str = "simple") -> pd.DataFrame:
    """
    Load NIBOR file of the form:

    date,nibor_1m
    2010-01,2.0365

    Assumes nibor_1m is an annual percentage rate and converts it to a
    monthly decimal RF.
    """
    df = pd.read_csv(path)

    required = ["date", "nibor_1m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"nibor_monthly.csv is missing required columns: {missing}")

    df["Date"] = parse_month_series(df["date"])
    df["nibor_1m"] = pd.to_numeric(df["nibor_1m"], errors="coerce")

    df = df.dropna(subset=["Date", "nibor_1m"]).copy()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first")

    if method == "simple":
        df["RF"] = df["nibor_1m"] / 100.0 / 12.0
    elif method == "compound":
        df["RF"] = (1.0 + df["nibor_1m"] / 100.0) ** (1.0 / 12.0) - 1.0
    else:
        raise ValueError("method must be 'simple' or 'compound'")

    return df[["Date", "RF"]].reset_index(drop=True)


def load_monthly_dividends(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        raise ValueError("dividends_monthly_nok.csv must contain a 'Ticker' column.")

    month_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(c).strip())]
    if not month_cols:
        raise ValueError("Could not detect monthly dividend columns like '2010-01' in dividends_monthly_nok.csv.")

    long_df = df.melt(
        id_vars="Ticker",
        value_vars=month_cols,
        var_name="Month",
        value_name="Dividend",
    ).copy()

    long_df["Ticker"] = long_df["Ticker"].astype(str).str.strip()
    long_df["Date"] = parse_month_series(long_df["Month"])
    long_df["Dividend"] = pd.to_numeric(long_df["Dividend"], errors="coerce")

    long_df = (
        long_df.dropna(subset=["Ticker", "Date"])
        .groupby(["Ticker", "Date"], as_index=False)["Dividend"]
        .sum(min_count=1)
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    return long_df


def load_prices_and_build_returns(
    path: Path,
    dividends_csv: Path | None = Path("data/processed_data_lseg/dividends_monthly_nok.csv"),
) -> pd.DataFrame:
    if dividends_csv is not None:
        project_root = find_project_root()
        dividends_csv = resolve_path(dividends_csv, project_root)

    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        raise ValueError("all_stock_prices.csv must contain a 'Ticker' column.")

    month_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(c).strip())]

    if not month_cols:
        raise ValueError("Could not detect monthly price columns like '2010-01' in all_stock_prices.csv.")

    long_df = df.melt(
        id_vars="Ticker",
        value_vars=month_cols,
        var_name="Month",
        value_name="Price",
    ).copy()

    long_df["Ticker"] = long_df["Ticker"].astype(str).str.strip()
    long_df["Date"] = parse_month_series(long_df["Month"])
    long_df["Price"] = pd.to_numeric(long_df["Price"], errors="coerce")

    long_df = (
        long_df.dropna(subset=["Ticker", "Date"])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    if dividends_csv is not None:
        dividends_df = load_monthly_dividends(dividends_csv)
        long_df = long_df.merge(
            dividends_df,
            on=["Ticker", "Date"],
            how="left",
            validate="1:1",
        )
    else:
        long_df["Dividend"] = 0.0

    long_df["Dividend"] = long_df["Dividend"].fillna(0.0)
    long_df["DividendRaw"] = long_df["Dividend"]
    long_df["LagPrice"] = long_df.groupby("Ticker")["Price"].shift(1)
    long_df["PriceReturn"] = long_df["Price"] / long_df["LagPrice"] - 1.0
    long_df["DividendYieldRaw"] = long_df["DividendRaw"] / long_df["LagPrice"]

    dividend_sanity_mask = (
        long_df["DividendRaw"].gt(0)
        & long_df["LagPrice"].gt(0)
        & long_df["DividendYieldRaw"].gt(MAX_MONTHLY_DIVIDEND_YIELD)
    )
    long_df["DividendSanityFlag"] = "ok"
    long_df.loc[
        dividend_sanity_mask,
        "DividendSanityFlag",
    ] = f"dividend_yield_gt_{MAX_MONTHLY_DIVIDEND_YIELD:g}_excluded"
    long_df.loc[dividend_sanity_mask, "Dividend"] = 0.0

    long_df["DividendYield"] = long_df["Dividend"] / long_df["LagPrice"]
    long_df["Return"] = (long_df["Price"] + long_df["Dividend"]) / long_df["LagPrice"] - 1.0

    valid_return = (
        long_df["Price"].notna()
        & long_df["LagPrice"].notna()
        & (long_df["LagPrice"] > 0)
        & long_df["Return"].notna()
    )
    returns_df = long_df.loc[valid_return].copy()
    return returns_df[
        [
            "Ticker",
            "Date",
            "Return",
            "Price",
            "LagPrice",
            "Dividend",
            "DividendRaw",
            "PriceReturn",
            "DividendYield",
            "DividendYieldRaw",
            "DividendSanityFlag",
        ]
    ].reset_index(drop=True)


# =============================================================================
# Annual sort inputs
# =============================================================================

def build_annual_sort_inputs(
    prepared_df: pd.DataFrame,
    mcap_df: pd.DataFrame,
) -> pd.DataFrame:
    acc = prepared_df[["Ticker", "Year", "SortYear", "BE", "AT", "PROF", "INV"]].copy()

    # December market cap of year t-1 for BM used in sort year t
    dec_mcap = mcap_df[mcap_df["Date"].dt.month == 12].copy()
    dec_mcap["SortYear"] = dec_mcap["Date"].dt.year + 1
    dec_mcap = dec_mcap.rename(columns={"MarketCap": "DecMarketCap"})
    dec_mcap = dec_mcap[["Ticker", "SortYear", "DecMarketCap"]]

    # June market cap of sort year t for size split
    june_mcap = mcap_df[mcap_df["Date"].dt.month == 6].copy()
    june_mcap["SortYear"] = june_mcap["Date"].dt.year
    june_mcap = june_mcap.rename(columns={"MarketCap": "JuneMarketCap"})
    june_mcap = june_mcap[["Ticker", "SortYear", "JuneMarketCap"]]

    annual = acc.merge(dec_mcap, on=["Ticker", "SortYear"], how="left")
    annual = annual.merge(june_mcap, on=["Ticker", "SortYear"], how="left")

    annual["BM"] = np.where(
        (annual["BE"] > 0) & (annual["DecMarketCap"] > 0),
        annual["BE"] / annual["DecMarketCap"],
        np.nan,
    )

    return annual.sort_values(["Ticker", "SortYear"]).reset_index(drop=True)


def build_annual_memberships(annual_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    bm_memberships = []
    prof_memberships = []
    inv_memberships = []

    for sort_year, sub in annual_df.groupby("SortYear", sort=True):
        sub = sub.copy()

        # Base size universe: positive June market cap
        size_universe = sub[sub["JuneMarketCap"].notna() & (sub["JuneMarketCap"] > 0)].copy()
        if size_universe.empty:
            continue

        size_universe["Size"] = assign_size_bucket(size_universe["JuneMarketCap"])

        # HML memberships
        bm_sub = size_universe[size_universe["BM"].notna() & (size_universe["BM"] > 0)].copy()
        if not bm_sub.empty:
            bm_sub["Char"] = assign_tercile_bucket(
                bm_sub["BM"],
                low_label="L",
                mid_label="N",
                high_label="H",
            )
            bm_memberships.append(bm_sub[["Ticker", "SortYear", "Size", "Char"]].copy())

        # RMW memberships
        prof_sub = size_universe[size_universe["PROF"].notna()].copy()
        if not prof_sub.empty:
            prof_sub["Char"] = assign_tercile_bucket(
                prof_sub["PROF"],
                low_label="W",
                mid_label="N",
                high_label="R",
            )
            prof_memberships.append(prof_sub[["Ticker", "SortYear", "Size", "Char"]].copy())

        # CMA memberships
        inv_sub = size_universe[size_universe["INV"].notna()].copy()
        if not inv_sub.empty:
            inv_sub["Char"] = assign_tercile_bucket(
                inv_sub["INV"],
                low_label="C",   # conservative = low asset growth
                mid_label="N",
                high_label="A",  # aggressive = high asset growth
            )
            inv_memberships.append(inv_sub[["Ticker", "SortYear", "Size", "Char"]].copy())

    out = {
        "BM": pd.concat(bm_memberships, ignore_index=True) if bm_memberships else pd.DataFrame(columns=["Ticker", "SortYear", "Size", "Char"]),
        "PROF": pd.concat(prof_memberships, ignore_index=True) if prof_memberships else pd.DataFrame(columns=["Ticker", "SortYear", "Size", "Char"]),
        "INV": pd.concat(inv_memberships, ignore_index=True) if inv_memberships else pd.DataFrame(columns=["Ticker", "SortYear", "Size", "Char"]),
    }
    return out


# =============================================================================
# Monthly base panel
# =============================================================================

def build_monthly_base_panel(
    returns_df: pd.DataFrame,
    mcap_df: pd.DataFrame,
    analysis_tickers: pd.Index,
) -> pd.DataFrame:
    monthly = returns_df.merge(
        mcap_df[["Ticker", "Date", "MarketCap", "LagMarketCap"]],
        on=["Ticker", "Date"],
        how="left",
        validate="1:1",
    ).copy()

    monthly = monthly[monthly["Ticker"].isin(set(analysis_tickers))].copy()
    monthly = monthly.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Sort year for July-June holding periods
    monthly["SortYear"] = np.where(
        monthly["Date"].dt.month >= 7,
        monthly["Date"].dt.year,
        monthly["Date"].dt.year - 1,
    )

    # Momentum signal: cumulative return from t-12 to t-2 (11 months), skipping t-1
    monthly["MOM_signal"] = (
        monthly.groupby("Ticker")["Return"]
        .transform(lambda s: (1.0 + s).shift(2).rolling(11).apply(np.prod, raw=True) - 1.0)
    )

    return monthly


# =============================================================================
# Factor builders
# =============================================================================

def build_market_return(monthly_df: pd.DataFrame) -> pd.Series:
    valid = monthly_df[
        monthly_df["Return"].notna()
        & monthly_df["LagMarketCap"].notna()
        & (monthly_df["LagMarketCap"] > 0)
    ].copy()

    market_ret = (
        valid.groupby("Date", as_index=True)
        .apply(lambda x: safe_weighted_average(x, "Return", "LagMarketCap"))
        .rename("RM")
    )

    return market_ret


def build_2x3_portfolio_returns(
    monthly_df: pd.DataFrame,
    membership_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = monthly_df.merge(
        membership_df,
        on=["Ticker", "SortYear"],
        how="inner",
        validate="many_to_one",
    ).copy()

    merged = merged[
        merged["Return"].notna()
        & merged["LagMarketCap"].notna()
        & (merged["LagMarketCap"] > 0)
        & merged["Size"].notna()
        & merged["Char"].notna()
    ].copy()

    if merged.empty:
        return pd.DataFrame()

    port_rets = (
        merged.groupby(["Date", "Size", "Char"], as_index=False)
        .apply(lambda x: pd.Series({"VWRet": safe_weighted_average(x, "Return", "LagMarketCap")}))
        .reset_index(drop=True)
    )

    wide = port_rets.pivot(index="Date", columns=["Size", "Char"], values="VWRet").sort_index()
    return wide


def build_hml_and_smb_from_bm(bm_wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    hml = 0.5 * (get_col(bm_wide, "S", "H") + get_col(bm_wide, "B", "H")) - \
          0.5 * (get_col(bm_wide, "S", "L") + get_col(bm_wide, "B", "L"))
    hml.name = "HML"

    smb_bm = (
        (get_col(bm_wide, "S", "L") + get_col(bm_wide, "S", "N") + get_col(bm_wide, "S", "H")) / 3.0
        - (get_col(bm_wide, "B", "L") + get_col(bm_wide, "B", "N") + get_col(bm_wide, "B", "H")) / 3.0
    )
    smb_bm.name = "SMB_BM"

    return hml, smb_bm


def build_rmw_and_smb_from_prof(prof_wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rmw = 0.5 * (get_col(prof_wide, "S", "R") + get_col(prof_wide, "B", "R")) - \
          0.5 * (get_col(prof_wide, "S", "W") + get_col(prof_wide, "B", "W"))
    rmw.name = "RMW"

    smb_prof = (
        (get_col(prof_wide, "S", "W") + get_col(prof_wide, "S", "N") + get_col(prof_wide, "S", "R")) / 3.0
        - (get_col(prof_wide, "B", "W") + get_col(prof_wide, "B", "N") + get_col(prof_wide, "B", "R")) / 3.0
    )
    smb_prof.name = "SMB_PROF"

    return rmw, smb_prof


def build_cma_and_smb_from_inv(inv_wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cma = 0.5 * (get_col(inv_wide, "S", "C") + get_col(inv_wide, "B", "C")) - \
          0.5 * (get_col(inv_wide, "S", "A") + get_col(inv_wide, "B", "A"))
    cma.name = "CMA"

    smb_inv = (
        (get_col(inv_wide, "S", "C") + get_col(inv_wide, "S", "N") + get_col(inv_wide, "S", "A")) / 3.0
        - (get_col(inv_wide, "B", "C") + get_col(inv_wide, "B", "N") + get_col(inv_wide, "B", "A")) / 3.0
    )
    smb_inv.name = "SMB_INV"

    return cma, smb_inv


def build_momentum_factor(monthly_df: pd.DataFrame) -> pd.Series:
    pieces = []

    for date, sub in monthly_df.groupby("Date", sort=True):
        sub = sub.copy()
        sub = sub[
            sub["Return"].notna()
            & sub["LagMarketCap"].notna()
            & (sub["LagMarketCap"] > 0)
            & sub["MOM_signal"].notna()
        ].copy()

        if len(sub) < 6:
            continue

        sub["Size"] = assign_size_bucket(sub["LagMarketCap"])
        sub["Char"] = assign_tercile_bucket(
            sub["MOM_signal"],
            low_label="L",
            mid_label="N",
            high_label="W",   # winners
        )

        port_rets = (
            sub.groupby(["Size", "Char"], as_index=False)
            .apply(lambda x: pd.Series({"VWRet": safe_weighted_average(x, "Return", "LagMarketCap")}))
            .reset_index(drop=True)
        )
        port_rets["Date"] = date
        pieces.append(port_rets)

    if not pieces:
        return pd.Series(dtype=float, name="MOM")

    wide = (
        pd.concat(pieces, ignore_index=True)
        .pivot(index="Date", columns=["Size", "Char"], values="VWRet")
        .sort_index()
    )

    mom = 0.5 * (get_col(wide, "S", "W") + get_col(wide, "B", "W")) - \
          0.5 * (get_col(wide, "S", "L") + get_col(wide, "B", "L"))
    mom.name = "MOM"

    return mom


# =============================================================================
# Main build function
# =============================================================================

def build_factor_csv(
    prepared_input_csv: str | Path = "results/extraction_static/prepared_step2_input.csv",
    stock_prices_csv: str | Path = "data/processed_data_lseg/all_stock_prices_nok.csv",
    dividends_csv: str | Path = "data/processed_data_lseg/dividends_monthly_nok.csv",
    market_cap_monthly_csv: str | Path = "data/processed_data_lseg/historical_market_cap_nok.csv",
    nibor_csv: str | Path = "data/nibor_monthly.csv",
    output_dir: str | Path = "results/extraction_static",
    rf_method: str = "simple",
) -> dict:
    project_root = find_project_root()

    prepared_input_csv = resolve_path(prepared_input_csv, project_root)
    stock_prices_csv = resolve_path(stock_prices_csv, project_root)
    dividends_csv = resolve_path(dividends_csv, project_root)
    market_cap_monthly_csv = resolve_path(market_cap_monthly_csv, project_root)
    nibor_csv = resolve_path(nibor_csv, project_root)
    output_dir = resolve_path(output_dir, project_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_df = load_prepared_panel(prepared_input_csv)
    mcap_df = load_market_cap_monthly(market_cap_monthly_csv)
    rf_df = load_nibor(nibor_csv, method=rf_method)
    returns_df = load_prices_and_build_returns(stock_prices_csv, dividends_csv=dividends_csv)

    analysis_tickers = prepared_df["Ticker"].dropna().astype(str).str.strip().unique()

    annual_inputs = build_annual_sort_inputs(prepared_df, mcap_df)
    memberships = build_annual_memberships(annual_inputs)
    monthly_df = build_monthly_base_panel(returns_df, mcap_df, analysis_tickers)

    # Market return
    market_ret = build_market_return(monthly_df)

    # Annual factors
    bm_wide = build_2x3_portfolio_returns(monthly_df, memberships["BM"])
    prof_wide = build_2x3_portfolio_returns(monthly_df, memberships["PROF"])
    inv_wide = build_2x3_portfolio_returns(monthly_df, memberships["INV"])

    hml, smb_bm = build_hml_and_smb_from_bm(bm_wide) if not bm_wide.empty else (pd.Series(dtype=float, name="HML"), pd.Series(dtype=float, name="SMB_BM"))
    rmw, smb_prof = build_rmw_and_smb_from_prof(prof_wide) if not prof_wide.empty else (pd.Series(dtype=float, name="RMW"), pd.Series(dtype=float, name="SMB_PROF"))
    cma, smb_inv = build_cma_and_smb_from_inv(inv_wide) if not inv_wide.empty else (pd.Series(dtype=float, name="CMA"), pd.Series(dtype=float, name="SMB_INV"))

    smb = pd.concat([smb_bm, smb_prof, smb_inv], axis=1).mean(axis=1)
    smb.name = "SMB"

    # Momentum
    mom = build_momentum_factor(monthly_df)

    # Combine
    factor_df = pd.concat(
        [
            market_ret,
            smb,
            hml,
            rmw,
            cma,
            mom,
            rf_df.set_index("Date")["RF"],
        ],
        axis=1,
    ).sort_index()

    factor_df["MKT"] = factor_df["RM"] - factor_df["RF"]

    factor_df = factor_df.reset_index().rename(columns={"index": "Date"})
    factor_df = factor_df[["Date", "MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF", "RM"]].copy()

    start_date = pd.Timestamp("2010-01-30")
    factor_df = factor_df[factor_df["Date"] >= start_date].copy()

    factor_csv_path = output_dir / "factor_data.csv"
    factor_df.to_csv(factor_csv_path, index=False)

    diagnostics = {
        "prepared_input_csv": str(prepared_input_csv),
        "stock_prices_csv": str(stock_prices_csv),
        "dividends_csv": str(dividends_csv),
        "market_cap_monthly_csv": str(market_cap_monthly_csv),
        "nibor_csv": str(nibor_csv),
        "output_csv": str(factor_csv_path),
        "n_factor_months": int(len(factor_df)),
        "date_min": str(factor_df["Date"].min()) if not factor_df.empty else None,
        "date_max": str(factor_df["Date"].max()) if not factor_df.empty else None,
        "n_analysis_tickers": int(len(pd.Index(analysis_tickers).unique())),
        "n_memberships_bm": int(len(memberships["BM"])),
        "n_memberships_prof": int(len(memberships["PROF"])),
        "n_memberships_inv": int(len(memberships["INV"])),
        "rf_method": rf_method,
        "notes": [
            "MKT is the value-weighted market return of firms in the analysis universe minus RF.",
            "Monthly returns include dividends from dividends_monthly_nok.csv.",
            "SMB/HML/RMW/CMA use annual July-June style holding periods with June size sorts and lagged annual accounting.",
            "MOM uses monthly 2x3 size-momentum sorts with momentum measured from t-12 to t-2.",
            "Assumes market caps are comparable across firms for sorting and weighting.",
        ],
    }

    diagnostics_path = output_dir / "factor_data_diagnostics.json"
    write_json(diagnostics_path, diagnostics)

    return {
        "factor_csv": str(factor_csv_path),
        "diagnostics_json": str(diagnostics_path),
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Nordic monthly factor CSV.")

    parser.add_argument(
        "--prepared_input_csv",
        type=str,
        default="results/extraction_static/prepared_step2_input.csv",
        help="Path to prepared_step2_input.csv",
    )
    parser.add_argument(
        "--stock_prices_csv",
        type=str,
        default="data/processed_data_lseg/all_stock_prices_nok.csv",
        help="Path to monthly stock prices CSV",
    )
    parser.add_argument(
        "--dividends_csv",
        type=str,
        default="data/processed_data_lseg/dividends_monthly_nok.csv",
        help="Path to monthly dividends CSV",
    )
    parser.add_argument(
        "--market_cap_monthly_csv",
        type=str,
        default="data/processed_data_lseg/historical_market_cap_nok.csv",
        help="Path to monthly market cap CSV",
    )
    parser.add_argument(
        "--nibor_csv",
        type=str,
        default="data/nibor_monthly.csv",
        help="Path to monthly NIBOR CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/extraction_static",
        help="Directory to store factor_data.csv",
    )
    parser.add_argument(
        "--rf_method",
        type=str,
        default="simple",
        choices=["simple", "compound"],
        help="How to convert annual NIBOR percent to monthly RF",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = build_factor_csv(
        prepared_input_csv=args.prepared_input_csv,
        stock_prices_csv=args.stock_prices_csv,
        dividends_csv=args.dividends_csv,
        market_cap_monthly_csv=args.market_cap_monthly_csv,
        nibor_csv=args.nibor_csv,
        output_dir=args.output_dir,
        rf_method=args.rf_method,
    )

    print("\nSaved factor outputs:")
    for k, v in result.items():
        print(f"  {k}: {v}")
