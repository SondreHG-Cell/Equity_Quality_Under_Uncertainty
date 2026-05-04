"""
LSEG extraction script: pulls equity prices, dividends, market cap, shares
outstanding, CFO forecasts, and earnings announcement dates, plus
instrument-level currency metadata.

Why local currency
------------------
Earlier tests showed that requesting Curn=NOK directly inside LSEG fields can
silently fail for some tickers (e.g. Helsinki stocks were returned in EUR
despite Curn=NOK being set). To make the pipeline auditable, we now:

    1. Pull prices and market cap in each stock's native trading currency.
    2. Pull dividends in each dividend event's native cash currency.
    3. Save instrument and dividend currency metadata for auditability.
    4. Run a separate FX conversion script (fx_convert_to_nok.py) that uses
       Norges Bank rates to convert local values to NOK.

Outputs (in OUTPUT directory: data/raw_data_lseg)
-------------------------------------------------
    instrument_currencies.csv         Ticker -> Currency (with source field)
    all_stock_prices_local.csv        Wide: rows=tickers, cols=YYYY-MM, native ccy
    historical_market_cap_local.csv   Wide: same shape, native ccy
    shares_outstanding.csv            Wide: same shape, count (no currency)
    dividends_raw_long_local.csv      Long dividend events. DPS_LOCAL is split-
                                      adjusted gross DPS in Currency.
    cfo_forecasts_monthly_raw.csv     Long: Ticker, snapshot_date, cfo_forecast
    announcement_dates.csv            Earnings announcement dates per fiscal year
    missing_*.csv                     Tickers where extraction failed
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

BASE = Path(__file__).resolve().parents[2]
FOLDER = BASE / "data" / "prof_components_extracted"
OUTPUT = BASE / "data" / "raw_data_lseg"

START = "2005-01-03"
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


def normalize_currency_code(value):
    """Normalize LSEG currency labels to ISO-like three-letter codes."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()
    if value in {"", "NAN", "NONE", "NULL"}:
        return np.nan

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

    def pick_currency(row):
        # Collect all non-null normalized values from successful fields.
        values = []
        for field in successful_fields:
            val = normalize_currency_code(row.get(field))
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
                fields=["TR.CLOSEPRICE(Adjusted=1)"],
                interval="monthly",
                start=start,
                end=end,
            )
            if df is not None and not df.empty:
                col = df.columns[0]
                series = pd.to_numeric(df[col], errors="coerce")
                if series.notna().any():
                    all_prices[ticker] = series
                    continue
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
        missing.append(ticker)
        time.sleep(0.1)

    print(f"\nLocal prices retrieved for {len(all_prices)}/{len(tickers)} tickers (missing: {len(missing)}).")
    if missing:
        pd.Series(missing, name="Ticker").to_csv(OUTPUT / "missing_prices_local.csv", index=False)
    return pd.DataFrame(all_prices)


# =============================================================================
# Local-currency dividends
# =============================================================================

def get_dividends_local(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Event-level gross cash dividends with ex-dates.

    LSEG returns TR.DivUnadjustedGross in the share-count terms that applied on
    the ex-date. Since prices are pulled with TR.CLOSEPRICE(Adjusted=1), we
    multiply each dividend by TR.DivAdjustmentFactor so DPS_LOCAL is on the
    same split-adjusted basis as prices.

    Returns long format:
        Ticker | ExDate | DPS_UNADJUSTED | AdjustmentFactor | DPS_LOCAL |
        Currency | AdjustmentBasis

    Currency is the dividend event's native cash currency from TR.DivCurr,
    which may differ from the instrument's trading currency.
    """
    rows = []
    missing = []
    missing_factor_tickers = set()

    fields = [
        "TR.DivExDate",
        "TR.DivUnadjustedGross",
        "TR.DivCurr",
        "TR.DivAdjustmentFactor",
    ]

    def norm_col(col: str) -> str:
        return "".join(ch for ch in str(col).lower() if ch.isalnum())

    def find_col(df: pd.DataFrame, field: str, contains_any: list[tuple[str, ...]]):
        normalized = {c: norm_col(c) for c in df.columns}
        target = norm_col(field)

        for col, clean in normalized.items():
            if clean == target:
                return col

        for col, clean in normalized.items():
            if col == "Instrument":
                continue
            for tokens in contains_any:
                if all(token in clean for token in tokens):
                    return col
        return None

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        try:
            df = lseg.get_data(
                universe=[ticker],
                fields=fields,
                parameters={
                    "SDate": start,
                    "EDate": end,
                },
            )
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
            missing.append(ticker)
            time.sleep(0.2)
            continue

        if df is None or df.empty:
            time.sleep(0.1)
            continue

        exdate_col = find_col(
            df,
            "TR.DivExDate",
            [("div", "ex", "date"), ("ex", "date")],
        )
        dividend_col = find_col(
            df,
            "TR.DivUnadjustedGross",
            [("div", "unadjusted", "gross"), ("unadjusted", "gross"), ("div", "gross")],
        )
        currency_col = find_col(
            df,
            "TR.DivCurr",
            [("div", "curr"), ("currency",)],
        )
        factor_col = find_col(
            df,
            "TR.DivAdjustmentFactor",
            [("div", "adjustment", "factor"), ("adjustment", "factor")],
        )

        if exdate_col is None or dividend_col is None:
            print(
                "    WARN: could not identify dividend ex-date/value columns. "
                f"Got: {df.columns.tolist()}"
            )
            missing.append(ticker)
            time.sleep(0.1)
            continue

        sub = pd.DataFrame({
            "ExDate": df[exdate_col],
            "DPS_UNADJUSTED": df[dividend_col],
        })

        if currency_col is not None:
            sub["Currency"] = df[currency_col].map(normalize_currency_code)
        else:
            sub["Currency"] = np.nan

        if factor_col is not None:
            raw_factor = pd.to_numeric(df[factor_col], errors="coerce")
            sub["AdjustmentFactor"] = raw_factor
            sub["AdjustmentBasis"] = np.where(
                raw_factor.notna(),
                "TR.DivUnadjustedGross_x_TR.DivAdjustmentFactor",
                "missing_adjustment_factor_default_1",
            )
        else:
            sub["AdjustmentFactor"] = np.nan
            sub["AdjustmentBasis"] = "factor_field_missing_default_1"

        sub["ExDate"] = pd.to_datetime(sub["ExDate"], errors="coerce")
        sub["DPS_UNADJUSTED"] = pd.to_numeric(sub["DPS_UNADJUSTED"], errors="coerce")

        valid_dividend = (
            sub["ExDate"].notna()
            & sub["DPS_UNADJUSTED"].notna()
            & (sub["DPS_UNADJUSTED"] > 0)
        )
        missing_factor = valid_dividend & sub["AdjustmentFactor"].isna()
        invalid_factor = (
            valid_dividend
            & sub["AdjustmentFactor"].notna()
            & (sub["AdjustmentFactor"] <= 0)
        )
        if missing_factor.any() or invalid_factor.any():
            missing_factor_tickers.add(ticker)
            sub.loc[missing_factor, "AdjustmentFactor"] = 1.0
            sub.loc[invalid_factor, "AdjustmentFactor"] = 1.0
            sub.loc[invalid_factor, "AdjustmentBasis"] = "invalid_adjustment_factor_default_1"

        sub["DPS_LOCAL"] = sub["DPS_UNADJUSTED"] * sub["AdjustmentFactor"]
        sub = sub.dropna(subset=["ExDate", "DPS_UNADJUSTED", "DPS_LOCAL"])
        sub = sub.loc[(sub["DPS_UNADJUSTED"] > 0) & (sub["DPS_LOCAL"] > 0)]
        sub["Ticker"] = ticker

        if not sub.empty:
            rows.append(
                sub[
                    [
                        "Ticker",
                        "ExDate",
                        "DPS_UNADJUSTED",
                        "AdjustmentFactor",
                        "DPS_LOCAL",
                        "Currency",
                        "AdjustmentBasis",
                    ]
                ]
            )
        time.sleep(0.1)

    if not rows:
        print("\nWARNING: no dividend rows returned for any ticker.")
        return pd.DataFrame(
            columns=[
                "Ticker",
                "ExDate",
                "DPS_UNADJUSTED",
                "AdjustmentFactor",
                "DPS_LOCAL",
                "Currency",
                "AdjustmentBasis",
            ]
        )

    out = pd.concat(rows, ignore_index=True).sort_values(["Ticker", "ExDate"]).reset_index(drop=True)
    print(f"\nDividends: {len(out):,} events across {out['Ticker'].nunique()} tickers.")
    print(f"  Date range: {out['ExDate'].min().date()} to {out['ExDate'].max().date()}")
    n_adjusted = (out["AdjustmentFactor"].round(12) != 1.0).sum()
    n_default_factor = out["AdjustmentBasis"].str.contains("default_1", na=False).sum()
    print(f"  Events with adjustment factor != 1: {n_adjusted:,}")
    print(f"  Events defaulting adjustment factor to 1: {n_default_factor:,}")
    if missing_factor_tickers:
        print(
            "  WARNING: Some dividend rows had missing/invalid adjustment factors. "
            "See AdjustmentBasis in dividends_raw_long_local.csv. Tickers: "
            f"{sorted(missing_factor_tickers)}"
        )
    if missing:
        pd.Series(missing, name="Ticker").to_csv(OUTPUT / "missing_dividends_local.csv", index=False)
    return out


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
    if missing:
        pd.Series(missing, name="Ticker").to_csv(OUTPUT / "missing_mktcap_local.csv", index=False)
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
        print("\n--- Fetching announcement dates ---")
        ann_dates = get_announcement_dates(tickers)
        if not ann_dates.empty and ann_dates["AnnouncementDate"].notna().any():
            clean = ann_dates.dropna(subset=["AnnouncementDate"]).copy()
            clean["PeriodEndDate"] = clean["PeriodEndDate"].dt.strftime("%Y-%m-%d")
            clean["AnnouncementDate"] = clean["AnnouncementDate"].dt.strftime("%Y-%m-%d")
            clean.to_csv(OUTPUT / "announcement_dates.csv", index=False)
            print(f"Saved {len(clean):,} announcement rows to announcement_dates.csv")

        # ---------------------------------------------------------------
        # 3. Local prices
        # ---------------------------------------------------------------
        print(f"\n--- Fetching monthly LOCAL prices ({START} → {END}) ---")
        prices_local = get_historical_prices_local(tickers, START, END)
        save_transposed_monthly(prices_local, OUTPUT / "all_stock_prices_local.csv", "local prices")

        # ---------------------------------------------------------------
        # 4. Local market cap
        # ---------------------------------------------------------------
        print(f"\n--- Fetching monthly LOCAL market cap ({START} → {END}) ---")
        mktcap_local = get_historical_market_cap_local(tickers, START, END)
        save_transposed_monthly(mktcap_local, OUTPUT / "historical_market_cap_local.csv", "local market cap")

        # ---------------------------------------------------------------
        # 6. Dividend events in their native cash currency
        # ---------------------------------------------------------------
        print(f"\n--- Fetching dividend events ({START} → {END}) ---")
        divs_local = get_dividends_local(tickers, START, END)

        # Attach instrument trading currency, but keep the dividend event
        # currency as the FX conversion currency. Dividends can be declared in
        # a different cash currency than the stock's trading currency.
        divs_local = divs_local.merge(
            currencies[["Ticker", "Currency"]].rename(columns={"Currency": "InstrumentCurrency"}),
            on="Ticker",
            how="left",
        )
        divs_local["Currency"] = divs_local["Currency"].fillna(divs_local["InstrumentCurrency"])

        n_missing_ccy = divs_local["Currency"].isna().sum()
        if n_missing_ccy > 0:
            bad = divs_local.loc[divs_local["Currency"].isna(), "Ticker"].unique().tolist()
            print(f"WARNING: {n_missing_ccy} dividend rows missing currency. Tickers: {bad}")

        n_ccy_diff = (
            divs_local["Currency"].notna()
            & divs_local["InstrumentCurrency"].notna()
            & (divs_local["Currency"] != divs_local["InstrumentCurrency"])
        ).sum()
        if n_ccy_diff > 0:
            print(
                f"NOTE: {n_ccy_diff:,} dividend rows use a different cash currency "
                "than the instrument trading currency."
            )

        divs_local.to_csv(OUTPUT / "dividends_raw_long_local.csv", index=False)
        print(f"Saved long dividends to {OUTPUT / 'dividends_raw_long_local.csv'}")

        # # ---------------------------------------------------------------
        # # 7. CFO forecasts (NOK natively from LSEG via Curn=NOK)
        # # ---------------------------------------------------------------
        # print(f"\n--- Fetching analyst CFO forecasts ---")
        # ocf_forecasts = get_monthly_cfo_forecasts(tickers, start="2010-01-01", end="2025-12-31")
        # if not ocf_forecasts.empty:
        #     ocf_forecasts.to_csv(OUTPUT / "cfo_forecasts_monthly_raw.csv", index=False)

        print("\nDone with extraction.")
        print("\nNext step: run the FX conversion script (fx_convert_to_nok.py).")
        print("It will use Norges Bank rates to produce:")
        print("  - all_stock_prices_nok.csv")
        print("  - historical_market_cap_nok.csv")
        print("  - dividends_raw_long_nok.csv")
        print("  - dividends_monthly_nok.csv")
        print("  - fx_conversion_audit.csv  (verify a few rows here)")

    finally:
        print("\nClosing LSEG session...")
        lseg.close_session()
        print("Session closed.")


if __name__ == "__main__":
    main()
