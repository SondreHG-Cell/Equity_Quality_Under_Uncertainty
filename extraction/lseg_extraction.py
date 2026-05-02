import lseg.data as lseg
import pandas as pd
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
app_key = os.getenv("LSEG_APP_KEY")

BASE = Path(__file__).parent.parent
FOLDER = BASE / "data" / "prof_components_extracted"
START = "2004-01-01"
END = "2026-03-31"
OUTPUT = BASE / "data"

EXCHANGE_CCY = {
    ".OL": "NOK",
    ".ST": "SEK",
    ".CO": "DKK",
    ".HE": "EUR",
    ".IC": "ISK",
}
 
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
    
    # Coalesce: EPS date first, then Revenue date as fallback
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
        print(f"Discarding {n_implausible:,} announcement dates outside the "
              f"0-270 day window (likely database misalignment).")
        out.loc[~plausible, "AnnouncementDate"] = pd.NaT
 
    return out[["Ticker", "FiscalYear", "PeriodEndDate", "AnnouncementDate"]]

def get_currency(ticker: str) -> str:
    for suffix, ccy in EXCHANGE_CCY.items():
        if ticker.endswith(suffix):
            return ccy
    return "NOK"

def get_tickers_from_folder(folder: Path) -> list[str]:
    """Extract ticker symbols from CSV filenames in the folder."""
    tickers = [f.stem for f in folder.glob("*.csv")]
    print(f"Found {len(tickers)} tickers in '{folder}'")
    return sorted(tickers)

def get_historical_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_prices = {}
    missing = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        try:
            df = lseg.get_history(
                universe=[ticker],
                fields=["TR.CLOSEPRICE(Adjusted=1, Curn=NOK)"],
                interval="monthly",
                start=start,
                end=end,
            )
            if df is not None and not df.empty:
                col = df.columns[0]
                if df[col].notna().any():
                    all_prices[ticker] = df[col]
                    continue
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
        missing.append(ticker)
        time.sleep(0.1)

    print(f"\nPrices retrieved for {len(all_prices)}/{len(tickers)} tickers (missing: {len(missing)}).")
    return pd.DataFrame(all_prices)

def get_historical_market_cap(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_mktcap = {}

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        try:
            df = lseg.get_history(
                universe=[ticker],
                fields=["TR.CompanyMarketCap(Curn=NOK)"],  # replace with whatever works
                interval="monthly",
                start=start,
                end=end,
            )
            if df is not None and not df.empty:
                col = df.columns[0]
                if df[col].notna().any():
                    all_mktcap[ticker] = df[col]
        except Exception:
            pass

        time.sleep(0.1)

    print(f"\nMarket cap retrieved for {len(all_mktcap)}/{len(tickers)} tickers.")
    return pd.DataFrame(all_mktcap)

def get_monthly_cfo_forecasts(tickers: list[str], start: str = "2010-01-01", end: str = "2025-12-31",) -> pd.DataFrame:

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
    print(f"\nCFO forecasts retrieved: {len(out)} monthly snapshots across {out['Ticker'].nunique()} tickers.")
    print(f"  Date range: {out['snapshot_date'].min().date()} to {out['snapshot_date'].max().date()}")
    print(f"  Median observations per ticker: {out.groupby('Ticker').size().median():.0f}")
    return out

def get_fx_rates(start: str, end: str) -> pd.DataFrame:
    pairs = ["SEKNOK=X", "DKKNOK=X", "EURNOK=X", "ISKNOK=X"]
    fx = lseg.get_history(
        universe=pairs,
        fields=["MID_PRICE"],
        interval="monthly",
        start=start,
        end=end,
    )
    fx.columns = ["SEK", "DKK", "EUR", "ISK"]
    fx["NOK"] = 1.0
    return fx

def get_shares_outstanding(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_shares = {}

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        try:
            df = lseg.get_history(
                universe=[ticker],
                fields=["TR.SharesOutstanding"],  # replace with whatever works
                interval="monthly",
                start=start,
                end=end,
            )
            if df is not None and not df.empty:
                col = df.columns[0]
                if df[col].notna().any():
                    all_shares[ticker] = df[col]
        except Exception:
            pass

        time.sleep(0.1)

    print(f"\nShares outstanding retrieved for {len(all_shares)}/{len(tickers)} tickers.")
    return pd.DataFrame(all_shares)

def save_transposed(df: pd.DataFrame, path: Path, label: str):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample("ME").last()
    df_T = df.T
    df_T.index.name = "Ticker"
    df_T.columns = [d.strftime("%Y-%m") for d in df_T.columns]
    df_T = df_T.dropna(how="all")
    df_T.to_csv(path)
    print(f"Saved {label} to {path} ({len(df_T)} firms, {len(df_T.columns)} months)")

def main():
 
    print("Connecting to LSEG Data Platform...")
    lseg.open_session("desktop.workspace", app_key=app_key)
    print("Connected successfully.")
 
    tickers = get_tickers_from_folder(FOLDER)
 
    print("\n--- Fetching announcement dates ---")
    ann_dates = get_announcement_dates(tickers)
 
    if not ann_dates.empty and ann_dates["AnnouncementDate"].notna().any():
        # Clean output: drop rows where the announcement date couldn't be
        # determined (no I/B/E/S coverage, or filtered out by sanity check).
        clean = ann_dates.dropna(subset=["AnnouncementDate"]).copy()
        clean["PeriodEndDate"] = clean["PeriodEndDate"].dt.strftime("%Y-%m-%d")
        clean["AnnouncementDate"] = clean["AnnouncementDate"].dt.strftime("%Y-%m-%d")
 
        out_path = OUTPUT / "announcement_dates.csv"
        clean.to_csv(out_path, index=False)
 
        print(f"\nSaved {len(clean):,} rows to {out_path}")
        print(f"  Unique tickers: {clean['Ticker'].nunique()}")
        print(f"  Fiscal years:   {clean['FiscalYear'].min()}-{clean['FiscalYear'].max()}")
        print(f"  Dropped (no announcement date): {len(ann_dates) - len(clean):,}")
    else:
        print("\nNo announcement dates retrieved. Run diagnose_announcement_fields() to investigate.")
 
    # print(f"Fetching monthly prices ({START} → {END})...")
    # prices = get_historical_prices(tickers, START, END)
 
    # print(f"\n--- Fetching monthly market cap ({START} → {END}) ---")
    # mktcap = get_historical_market_cap(tickers, START, END)
 
    # print(f"\n--- Fetching monthly shares outstanding ({START} → {END}) ---")
    # shares_outstanding = get_shares_outstanding(tickers, START, END)
 
    # print(f"\n--- Fetching analyst OCF forecasts via get_history ---")
    # ocf_forecasts = get_monthly_cfo_forecasts(tickers, start="2010-01-01", end="2025-12-31")
 
    # print("Fetching FX rates from LSEG...")
    # fx = get_fx_rates(START, END)
 
    # fx.to_csv(OUTPUT / "fx_rates.csv")
 
    # Transpose: tickers as rows, dates as columns
 
    # save_transposed(prices, OUTPUT / "all_stock_prices.csv", "prices")
    # save_transposed(prices_nok, OUTPUT / "all_stock_prices_nok.csv", "prices (NOK)")
    # save_transposed(mktcap, OUTPUT / "historical_market_cap.csv", "market cap")
    # save_transposed(shares_outstanding, OUTPUT / "shares_outstanding.csv", "shares outstanding")
    # ocf_forecasts.to_csv(OUTPUT / "cfo_forecasts_monthly_raw.csv", index=False)
    lseg.close_session()
 
if __name__ == "__main__":
    main()