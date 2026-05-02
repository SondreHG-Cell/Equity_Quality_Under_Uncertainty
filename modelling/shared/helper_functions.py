from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# Generic helpers
# ============================================================

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


def parse_month_series(s: pd.Series) -> pd.Series:
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


# ============================================================
# Loaders
# ============================================================

def load_factor_data(factors_csv: str | Path) -> pd.DataFrame:
    project_root = find_project_root()
    factors_csv = resolve_path(factors_csv, project_root)

    df = pd.read_csv(factors_csv)

    if "Date" not in df.columns:
        raise ValueError("Factor CSV must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    required = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Factor CSV missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first")
    df = df.set_index("Date")

    return df


def load_prices_and_build_returns(path: str | Path) -> pd.DataFrame:
    project_root = find_project_root()
    path = resolve_path(path, project_root)

    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        raise ValueError("Stock price CSV must contain a 'Ticker' column.")

    month_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(c).strip())]
    if not month_cols:
        raise ValueError("Could not detect monthly columns like '2010-01' in stock price CSV.")

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

    long_df["Return"] = long_df.groupby("Ticker")["Price"].pct_change()

    returns_df = long_df.dropna(subset=["Return"]).copy()
    return returns_df[["Ticker", "Date", "Return"]].reset_index(drop=True)


def load_market_cap_monthly(path: str | Path) -> pd.DataFrame:
    project_root = find_project_root()
    path = resolve_path(path, project_root)

    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        raise ValueError("Market cap CSV must contain a 'Ticker' column.")

    month_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(c).strip())]
    if not month_cols:
        raise ValueError("Could not detect monthly columns like '2010-01' in market cap CSV.")

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


# ============================================================
# Portfolio return construction
# ============================================================

def assign_june_formation_year(dates: pd.Series) -> pd.Series:
    """
    Map monthly return dates to the portfolio formed at the prior June end.

    FormationYear Y is held from July of Y through June of Y+1.
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    return parsed.dt.year.where(parsed.dt.month >= 7, parsed.dt.year - 1)


def build_monthly_portfolio_returns(
    assignments: pd.DataFrame,
    stock_prices_csv: str | Path,
    market_cap_csv: str | Path,
    factors: pd.DataFrame,
    n_portfolios: int = 5,
) -> dict:
    """
    Uses:
    - yearly portfolio membership from assignments
    - monthly stock returns from stock_prices_csv
    - monthly lagged market cap weights from market_cap_csv

    Assumption:
    - portfolios are formed at June end
    - FormationYear Y applies to July Y through June Y+1
    """
    required_cols = ["Ticker", "FormationYear", "Method", "PortfolioNum", "Portfolio"]
    missing = [c for c in required_cols if c not in assignments.columns]
    if missing:
        raise ValueError(f"Assignments CSV missing required columns: {missing}")

    assignments = assignments.copy()
    assignments["Ticker"] = assignments["Ticker"].astype(str).str.strip()
    assignments["FormationYear"] = pd.to_numeric(assignments["FormationYear"], errors="coerce").astype("Int64")
    assignments["PortfolioNum"] = pd.to_numeric(assignments["PortfolioNum"], errors="coerce").astype("Int64")
    assignments["Portfolio"] = assignments["Portfolio"].astype(str)

    assignments = assignments.dropna(subset=["Ticker", "FormationYear", "Method", "PortfolioNum"]).copy()
    assignments["FormationYear"] = assignments["FormationYear"].astype(int)

    returns_df = load_prices_and_build_returns(stock_prices_csv)
    mcap_df = load_market_cap_monthly(market_cap_csv)

    stock_panel = returns_df.merge(
        mcap_df,
        on=["Ticker", "Date"],
        how="left",
        validate="1:1",
    ).copy()

    stock_panel["FormationYear"] = assign_june_formation_year(stock_panel["Date"])

    monthly_holdings = stock_panel.merge(
        assignments,
        on=["Ticker", "FormationYear"],
        how="inner",
        validate="many_to_many",
    ).copy()

    monthly_holdings = monthly_holdings[
        monthly_holdings["Return"].notna()
        & monthly_holdings["LagMarketCap"].notna()
        & (monthly_holdings["LagMarketCap"] > 0)
        & monthly_holdings["PortfolioNum"].notna()
    ].copy()

    # Monthly value weights within Date x Method x Portfolio
    denom = (
        monthly_holdings.groupby(["Date", "Method", "PortfolioNum"])["LagMarketCap"]
        .transform("sum")
    )
    monthly_holdings["Weight"] = monthly_holdings["LagMarketCap"] / denom
    monthly_holdings["WeightedReturn"] = monthly_holdings["Weight"] * monthly_holdings["Return"]

    portfolio_returns = (
        monthly_holdings.groupby(["Date", "Method", "PortfolioNum", "Portfolio"], as_index=False)
        .agg(
            PortfolioReturn=("WeightedReturn", "sum"),
            n_firms=("Ticker", "nunique"),
            total_lag_mcap=("LagMarketCap", "sum"),
        )
        .sort_values(["Date", "Method", "PortfolioNum"])
        .reset_index(drop=True)
    )

    returns_wide = (
        portfolio_returns.pivot(
            index="Date",
            columns=["Method", "Portfolio"],
            values="PortfolioReturn",
        )
        .sort_index()
    )

    # Align to factors index
    returns_wide = returns_wide.loc[returns_wide.index.intersection(factors.index)].sort_index()
    rf = factors.loc[returns_wide.index, "RF"].copy()

    methods = sorted(assignments["Method"].dropna().unique())

    q1_label = "Q1"
    qn_label = f"Q{n_portfolios}"

    ls_returns = {}
    q5_returns = {}

    for method in methods:
        subcols = returns_wide[method].columns.tolist() if method in returns_wide.columns.get_level_values(0) else []

        if q1_label in subcols and qn_label in subcols:
            ls_returns[method] = returns_wide[(method, qn_label)] - returns_wide[(method, q1_label)]
            q5_returns[method] = returns_wide[(method, qn_label)]

    return {
        "returns_wide": returns_wide,
        "ls_returns": ls_returns,
        "q5_returns": q5_returns,
        "rf": rf,
        "monthly_holdings": monthly_holdings,
    }


# ============================================================
# Probabilistic targets
# ============================================================

def build_probabilistic_targets(
    assignments: pd.DataFrame,
    stock_prices_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build:
    - y_prob = predicted p_q5 from Method 3 input
    - y_true = 1 if realised annual return is in cross-sectional Q5 within FormationYear

    Assumption:
    - realised holding-period return is July Y through June Y+1
      for FormationYear Y
    """
    if "p_q5" not in assignments.columns:
        raise ValueError("Assignments must contain 'p_q5' for probabilistic evaluation.")

    returns_df = load_prices_and_build_returns(stock_prices_csv)

    monthly = returns_df.copy()
    monthly["FormationYear"] = assign_june_formation_year(monthly["Date"])

    # July-June buy-and-hold return within FormationYear.
    annual_realised = (
        monthly.groupby(["Ticker", "FormationYear"], as_index=False)
        .agg(
            RealisedReturn=("Return", lambda x: (1 + x).prod() - 1),
            n_months=("Return", "count"),
        )
    )
    annual_realised = annual_realised.loc[annual_realised["n_months"] == 12].copy()

    base = assignments[
        ["Ticker", "FormationYear", "p_q5"]
    ].drop_duplicates(subset=["Ticker", "FormationYear"]).copy()

    df = base.merge(
        annual_realised,
        on=["Ticker", "FormationYear"],
        how="inner",
        validate="1:1",
    )

    # Realised Q5 within each formation year
    def assign_q5_flag(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        if len(sub) < 5:
            sub["y_true"] = np.nan
            return sub

        sub = sub.sort_values(["RealisedReturn", "Ticker"]).reset_index(drop=True)
        sub["_rank"] = np.arange(len(sub))
        sub["RealisedPortfolioNum"] = pd.qcut(
            sub["_rank"],
            q=5,
            labels=range(1, 6),
        ).astype(int)
        sub["y_true"] = (sub["RealisedPortfolioNum"] == 5).astype(int)
        return sub

    df = (
        df.groupby("FormationYear", group_keys=False)
        .apply(assign_q5_flag)
        .reset_index(drop=True)
    )

    df = df.dropna(subset=["y_true", "p_q5"]).copy()

    y_true = df["y_true"].astype(int).values
    y_prob = df["p_q5"].astype(float).values

    return y_true, y_prob
