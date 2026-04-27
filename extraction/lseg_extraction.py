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
                fields=["TR.CashFlowFromOperationsMeanEstimate(Period=FY1,Scale=6)"],
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

def convert_to_nok(prices: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    converted = prices.copy()
    for ticker in converted.columns:
        ccy = get_currency(ticker)
        if ccy == "NOK":
            continue
        fx_aligned = fx[ccy].reindex(converted.index, method="ffill")
        converted[ticker] = converted[ticker] * fx_aligned
    return converted

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

    # print(f"Fetching monthly prices ({START} → {END})...")
    # prices = get_historical_prices(tickers, START, END)

    print(f"\n--- Fetching monthly market cap ({START} → {END}) ---")
    mktcap = get_historical_market_cap(tickers, START, END)

    # print(f"\n--- Fetching monthly shares outstanding ({START} → {END}) ---")
    # shares_outstanding = get_shares_outstanding(tickers, START, END)

    # print(f"\n--- Fetching analyst OCF forecasts via get_history ---")
    # ocf_forecasts = get_monthly_cfo_forecasts(tickers, start="2010-01-01", end="2025-12-31")

    # print("Fetching FX rates from LSEG...")
    # fx = get_fx_rates(START, END)

    # fx.to_csv(OUTPUT / "fx_rates.csv")

    # print("Converting all prices to NOK...")
    # prices_nok = convert_to_nok(prices, fx)

    # mktcap_nok = convert_to_nok(mktcap, fx)

    # Transpose: tickers as rows, dates as columns

    # save_transposed(prices, OUTPUT / "all_stock_prices.csv", "prices")
    # save_transposed(prices_nok, OUTPUT / "all_stock_prices_nok.csv", "prices (NOK)")
    save_transposed(mktcap, OUTPUT / "historical_market_cap.csv", "market cap")
    # save_transposed(shares_outstanding, OUTPUT / "shares_outstanding.csv", "shares outstanding")
    # ocf_forecasts.to_csv(OUTPUT / "cfo_forecasts_monthly_raw.csv", index=False)
    lseg.close_session()

if __name__ == "__main__":
    main()