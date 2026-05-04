"""
LSEG extraction script: pulls equity prices, market cap, CFO forecasts, and earnings announcement dates in each stock's
LOCAL TRADING CURRENCY, plus instrument-level currency metadata.

Outputs (in OUTPUT directory)
-----------------------------
    instrument_currencies.csv         Ticker -> Currency (with source field)
    all_stock_prices_local.csv        Wide: rows=tickers, cols=YYYY-MM, native ccy
    historical_market_cap_local.csv   Wide: same shape, native ccy
    cfo_forecasts_monthly_raw.csv     Long: Ticker, snapshot_date, cfo_forecast
    announcement_dates.csv            Earnings announcement dates per fiscal year
"""

import os
import time
import warnings
from pathlib import Path

import lseg.data as lseg
import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

load_dotenv()
APP_KEY = os.getenv("LSEG_APP_KEY")

BASE = Path(__file__).parent.parent
FOLDER = BASE / "data" / "prof_components_extracted"
OUTPUT = BASE / "data"

START = "2004-01-01"
END = "2026-03-31"

# Fallback only - we prefer LSEG's currency metadata when available.
EXCHANGE_CCY_FALLBACK = {
    ".OL": "NOK",
    ".ST": "SEK",
    ".CO": "DKK",
    ".HE": "EUR",
    ".IC": "ISK",
}


# =============================================================================
# Generic helpers
# =============================================================================

def get_tickers_from_folder(folder: Path) -> list[str]:
    tickers = [f.stem for f in folder.glob("*.csv")]
    print(f"Found {len(tickers)} tickers in '{folder}'")
    return sorted(tickers)


def get_currency_from_suffix(ticker: str) -> str:
    """Fallback currency mapping based on ticker suffix."""
    for suffix, ccy in EXCHANGE_CCY_FALLBACK.items():
        if ticker.endswith(suffix):
            return ccy
    return "UNKNOWN"


def save_transposed_monthly(df: pd.DataFrame, path: Path, label: str) -> None:
    """
    Save a date-indexed wide dataframe as:
        rows = tickers, columns = YYYY-MM, values = monthly observations.
    """
    if df.empty:
        print(f"WARNING: {label} is empty. Nothing saved.")
        return

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.resample("ME").last()

    out_t = out.T
    out_t.index.name = "Ticker"
    out_t.columns = [d.strftime("%Y-%m") for d in out_t.columns]
    out_t = out_t.dropna(how="all")

    out_t.to_csv(path)
    print(f"Saved {label} to {path} ({len(out_t)} firms, {len(out_t.columns)} months)")


# =============================================================================
# Currency metadata
# =============================================================================

def get_instrument_currencies(tickers: list[str]) -> pd.DataFrame:
    """
    Try multiple LSEG currency fields. Fall back to suffix mapping if none
    return data. Records the source field for each ticker so the audit trail
    can be reviewed.
    """
    candidate_fields = [
        "TR.Currency",
        "TR.PriceCloseCurrency",
        "TR.TradingCurrency",
        "TR.PrimaryQuoteCurrency",
        "CF_CURR",
    ]

    out = pd.DataFrame({"Ticker": tickers})
    successful_fields = []

    for field in candidate_fields:
        print(f"Trying LSEG currency field: {field}")
        try:
            df = lseg.get_data(universe=tickers, fields=[field])
        except Exception as e:
            print(f"  Failed: {type(e).__name__}: {e}")
            continue

        if df is None or df.empty:
            print("  Empty result.")
            continue

        ticker_col = "Instrument" if "Instrument" in df.columns else df.columns[0]
        value_cols = [c for c in df.columns if c != ticker_col]
        if not value_cols:
            continue

        sub = df[[ticker_col, value_cols[0]]].copy()
        sub.columns = ["Ticker", field]
        out = out.merge(sub, on="Ticker", how="left")
        successful_fields.append(field)

    def normalize_currency(value):
        if pd.isna(value):
            return np.nan
        value = str(value).strip().upper()
        replacements = {
            "EURO": "EUR",
            "DANISH KRONE": "DKK",
            "SWEDISH KRONA": "SEK",
            "NORWEGIAN KRONE": "NOK",
            "ICELAND KRONA": "ISK",
            "ICELANDIC KRONA": "ISK",
            "US DOLLAR": "USD",
            "U.S. DOLLAR": "USD",
        }
        return replacements.get(value, value)

    def pick_currency(row):
        # Collect all non-null normalized values from successful fields.
        values = []
        for field in successful_fields:
            val = normalize_currency(row.get(field))
            if pd.notna(val) and val != "":
                values.append((field, val))

        if not values:
            return pd.Series(
                [get_currency_from_suffix(row["Ticker"]), "suffix_fallback", ""]
            )

        first_field, first_value = values[0]
        # Disagreement check.
        disagreements = [v for (f, v) in values[1:] if v != first_value]
        warning = ""
        if disagreements:
            warning = (
                f"DISAGREE: {first_field}={first_value}; "
                + "; ".join(f"{f}={v}" for f, v in values[1:])
            )

        return pd.Series([first_value, first_field, warning])

    picked = out.apply(pick_currency, axis=1)
    out["Currency"] = picked[0]
    out["CurrencySource"] = picked[1]
    out["CurrencyWarning"] = picked[2]

    # Order columns nicely.
    first_cols = ["Ticker", "Currency", "CurrencySource", "CurrencyWarning"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    # Diagnostics
    print("\nCurrency summary:")
    print(out["Currency"].value_counts(dropna=False))
    print("\nCurrency source summary:")
    print(out["CurrencySource"].value_counts(dropna=False))

    n_unknown = (out["Currency"] == "UNKNOWN").sum()
    if n_unknown > 0:
        bad = out.loc[out["Currency"] == "UNKNOWN", "Ticker"].tolist()
        print(f"\nWARNING: {n_unknown} tickers with UNKNOWN currency:")
        print(f"  {bad}")

    n_disagree = (out["CurrencyWarning"] != "").sum()
    if n_disagree > 0:
        print(
            f"\nWARNING: {n_disagree} tickers had disagreeing LSEG currency fields. "
            "See 'CurrencyWarning' column in instrument_currencies.csv."
        )

    return out


# =============================================================================
# Earnings announcement dates
# =============================================================================

def get_announcement_dates(tickers: list[str], n_periods: int = 15) -> pd.DataFrame:
    """Fetch earnings announcement dates for FY0 through FY-(n_periods-1)."""
    records = []

    for offset in range(n_periods):
        period_label = f"FY-{offset}" if offset > 0 else "FY0"
        print(f"  Fetching {period_label} ({offset + 1}/{n_periods})...")

        try:
            df = lseg.get_data(
                universe=tickers,
                fields=[
                    "TR.F.PeriodEndDate",
                    "TR.EPSActReportDate",
                    "TR.RevenueActReportDate",
                ],
                parameters={"Period": period_label, "Frq": "FY"},
            )
        except Exception as e:
            print(f"    error on {period_label}: {type(e).__name__}: {e}")
            continue

        if df is None or df.empty:
            continue

        expected = ["Instrument", "PeriodEndDate", "EPSReportDate", "RevenueReportDate"]
        if len(df.columns) != len(expected):
            print(f"    unexpected column count: {df.columns.tolist()}")
            continue
        df.columns = expected
        df["RequestedPeriod"] = period_label
        records.append(df)

    if not records:
        print("WARNING: no data returned for any period.")
        return pd.DataFrame()

    raw = pd.concat(records, ignore_index=True)
    eps_n = raw["EPSReportDate"].notna().sum()
    rev_n = raw["RevenueReportDate"].notna().sum()
    print(f"\nNon-null EPS report dates:     {eps_n:,}")
    print(f"Non-null Revenue report dates: {rev_n:,}")

    raw["AnnouncementDate"] = raw["EPSReportDate"].fillna(raw["RevenueReportDate"])

    out = raw.rename(columns={"Instrument": "Ticker"})
    out["PeriodEndDate"] = pd.to_datetime(out["PeriodEndDate"], errors="coerce")
    out["AnnouncementDate"] = pd.to_datetime(out["AnnouncementDate"], errors="coerce")
    out = out.dropna(subset=["PeriodEndDate"])
    out["FiscalYear"] = out["PeriodEndDate"].dt.year

    out = (
        out.sort_values(["Ticker", "FiscalYear", "AnnouncementDate"])
           .drop_duplicates(subset=["Ticker", "FiscalYear"], keep="first")
    )

    gap_days = (out["AnnouncementDate"] - out["PeriodEndDate"]).dt.days
    plausible = gap_days.between(0, 270)
    n_implausible = (~plausible & out["AnnouncementDate"].notna()).sum()
    if n_implausible:
        print(f"Discarding {n_implausible:,} implausible announcement dates.")
        out.loc[~plausible, "AnnouncementDate"] = pd.NaT

    return out[["Ticker", "FiscalYear", "PeriodEndDate", "AnnouncementDate"]]


# =============================================================================
# Local-currency price extraction
# =============================================================================

def get_historical_prices_local(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Monthly adjusted close prices in NATIVE trading currency."""
    all_prices = {}
    missing = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        try:
            df = lseg.get_history(
                universe=[ticker],
                fields=["TR.TotalReturn"],
                interval="monthly",
                start=start,
                end=end,
            )
            if df is not None and not df.empty:
                col = df.columns[0]
                series = pd.to_numeric(df[col], errors="coerce")
                series.index = pd.to_datetime(series.index, errors="coerce")
                series = series[series.index.notna()]
                series = series.dropna()

                # Handle duplicate dates returned by LSEG
                if series.index.has_duplicates:
                    print(f"    WARN: duplicate dates for {ticker}; keeping last observation per date")
                    series = series.groupby(series.index).last()

                if series.notna().any():
                    all_prices[ticker] = series
                    continue
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
        missing.append(ticker)
        time.sleep(0.1)

    print(f"\nLocal prices retrieved for {len(all_prices)}/{len(tickers)} tickers (missing: {len(missing)}).")
    return pd.DataFrame(all_prices)



# =============================================================================
# Local-currency market cap
# =============================================================================

def get_historical_market_cap_local(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Monthly market cap in NATIVE trading currency."""
    all_mktcap = {}
    missing = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        try:
            df = lseg.get_history(
                universe=[ticker],
                fields=["TR.CompanyMarketCap"],
                interval="monthly",
                start=start,
                end=end,
            )
            if df is not None and not df.empty:
                col = df.columns[0]
                series = pd.to_numeric(df[col], errors="coerce")
                if series.notna().any():
                    all_mktcap[ticker] = series
                    continue
        except Exception:
            pass
        missing.append(ticker)
        time.sleep(0.1)

    print(f"\nLocal market cap retrieved for {len(all_mktcap)}/{len(tickers)} tickers.")
    return pd.DataFrame(all_mktcap)



# =============================================================================
# CFO forecasts (analyst consensus)
# =============================================================================

def get_monthly_cfo_forecasts(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Pulls analyst CFO forecasts in NOK directly.

    Note: We keep Curn=NOK here because the CFO forecast file is consumed by
    the HB model in NOK terms and the previous extraction has been validated
    against analyst snapshots. If you change the convention to local currency
    here, you must also adjust the HB model's expected CFO scaling.
    """
    rows = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        try:
            hist = lseg.get_history(
                universe=[ticker],
                fields=["TR.CashFlowFromOperationsMeanEstimate(Period=FY1, Scale=6, Curn=NOK)"],
                interval="monthly",
                start=start,
                end=end,
            )
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
            time.sleep(0.2)
            continue

        if hist is None or hist.empty:
            time.sleep(0.1)
            continue

        col = next((c for c in hist.columns if "mean estimate" in c.lower()), None)
        if col is None:
            print(f"    WARN: no mean-estimate column. Got: {hist.columns.tolist()}")
            time.sleep(0.1)
            continue

        series = hist[col].dropna()
        if series.empty:
            time.sleep(0.1)
            continue

        for dt, val in series.items():
            rows.append({
                "Ticker": ticker,
                "snapshot_date": pd.to_datetime(dt),
                "cfo_forecast": float(val),
            })
        time.sleep(0.1)

    out = pd.DataFrame(rows)
    if out.empty:
        print("\nWARNING: no CFO forecasts retrieved.")
        return out

    out = out.sort_values(["Ticker", "snapshot_date"]).reset_index(drop=True)
    print(f"\nCFO forecasts: {len(out)} snapshots across {out['Ticker'].nunique()} tickers.")
    print(f"  Date range: {out['snapshot_date'].min().date()} to {out['snapshot_date'].max().date()}")
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Connecting to LSEG Data Platform...")
    lseg.open_session("desktop.workspace", app_key=APP_KEY)
    print("Connected successfully.")

    try:
        tickers = get_tickers_from_folder(FOLDER)

        # ---------------------------------------------------------------
        # 1. Currency metadata (FIRST - everything else depends on it)
        # ---------------------------------------------------------------
        print("\n--- Fetching instrument currencies ---")
        currencies = get_instrument_currencies(tickers)
        currencies.to_csv(OUTPUT / "instrument_currencies.csv", index=False)
        print(f"Saved currencies to {OUTPUT / 'instrument_currencies.csv'}")

        # ---------------------------------------------------------------
        # 2. Announcement dates (currency-independent)
        # ---------------------------------------------------------------
        # print("\n--- Fetching announcement dates ---")
        # ann_dates = get_announcement_dates(tickers)
        # if not ann_dates.empty and ann_dates["AnnouncementDate"].notna().any():
        #     clean = ann_dates.dropna(subset=["AnnouncementDate"]).copy()
        #     clean["PeriodEndDate"] = clean["PeriodEndDate"].dt.strftime("%Y-%m-%d")
        #     clean["AnnouncementDate"] = clean["AnnouncementDate"].dt.strftime("%Y-%m-%d")
        #     clean.to_csv(OUTPUT / "announcement_dates.csv", index=False)
        #     print(f"Saved {len(clean):,} announcement rows to announcement_dates.csv")

        # ---------------------------------------------------------------
        # 3. Local prices
        # ---------------------------------------------------------------
        print(f"\n--- Fetching monthly LOCAL prices ({START} → {END}) ---")
        prices_local = get_historical_prices_local(tickers, START, END)
        # save_transposed_monthly(prices_local, OUTPUT / "all_stock_prices_local.csv", "local prices")
        save_transposed_monthly(prices_local, OUTPUT / "total_returns_local.csv", "local total returns")

        # ---------------------------------------------------------------
        # 4. Local market cap
        # ---------------------------------------------------------------
        # print(f"\n--- Fetching monthly LOCAL market cap ({START} → {END}) ---")
        # mktcap_local = get_historical_market_cap_local(tickers, START, END)
        # save_transposed_monthly(mktcap_local, OUTPUT / "historical_market_cap_local.csv", "local market cap")

        # ---------------------------------------------------------------
        # 5. CFO forecasts (NOK natively from LSEG via Curn=NOK)
        # ---------------------------------------------------------------
        # print(f"\n--- Fetching analyst CFO forecasts ---")
        # ocf_forecasts = get_monthly_cfo_forecasts(tickers, start="2010-01-01", end="2025-12-31")
        # if not ocf_forecasts.empty:
        #     ocf_forecasts.to_csv(OUTPUT / "cfo_forecasts_monthly_raw.csv", index=False)

        # print("\nDone with extraction.")
        # print("\nNext step: run the FX conversion script (fx_convert_to_nok.py).")
        # print("It will use Norges Bank rates to produce:")
        # print("  - all_stock_prices_nok.csv")
        # print("  - historical_market_cap_nok.csv")
        # print("  - dividends_raw_long_nok.csv")
        # print("  - dividends_monthly_nok.csv")
        # print("  - fx_conversion_audit.csv  (verify a few rows here)")

    finally:
        print("\nClosing LSEG session...")
        lseg.close_session()
        print("Session closed.")


if __name__ == "__main__":
    main()
