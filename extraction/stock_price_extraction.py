import lseg.data as lseg
import pandas as pd
import os
import time
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
    field_used = {}

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")
        for field in ["OFF_CLOSE", "TRDPRC_1"]:
            try:
                df = lseg.get_history(
                    universe=[ticker],
                    fields=[field],
                    interval="monthly",
                    start=start,
                    end=end,
                )
                if df is not None and not df.empty and df[field].notna().any():
                    all_prices[ticker] = df[field]
                    field_used[ticker] = field
                    break
            except Exception:
                continue
        
        time.sleep(0.2)

    primary = sum(1 for v in field_used.values() if v == "OFF_CLOSE")
    fallback = sum(1 for v in field_used.values() if v == "TRDPRC_1")
    missing = len(tickers) - len(field_used)
    print(f"\nField usage: OFF_CLOSE={primary}, TRDPRC_1 (fallback)={fallback}, no data={missing}")
    print(f"Data retrieved for {len(all_prices)}/{len(tickers)} tickers.")

    return pd.DataFrame(all_prices)

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

def convert_prices_to_nok(prices: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    converted = prices.copy()
    for ticker in converted.columns:
        ccy = get_currency(ticker)
        if ccy == "NOK":
            continue
        fx_aligned = fx[ccy].reindex(converted.index, method="ffill")
        converted[ticker] = converted[ticker] * fx_aligned
    return converted

def main():

    print("Connecting to LSEG Data Platform...")
    lseg.open_session("desktop.workspace", app_key=app_key)
    print("Connected successfully.")

    tickers = get_tickers_from_folder(FOLDER)

    print(f"Fetching monthly prices ({START} → {END})...")
    prices = get_historical_prices(tickers, START, END)

    print("Fetching FX rates from LSEG...")
    fx = get_fx_rates(START, END)

    fx.to_csv(OUTPUT / "fx_rates.csv")

    print("Converting all prices to NOK...")
    prices_nok = convert_prices_to_nok(prices, fx)

    # Transpose: tickers as rows, dates as columns
    prices_T = prices_nok.T
    prices_T.index.name = "Ticker"
    prices_T.columns = [d.strftime("%Y-%m") for d in prices_T.columns]
    prices_T = prices_T.dropna(how="all")

    prices_T.to_csv(OUTPUT / "all_stock_prices.csv")
    print(f"Saved ({len(prices_T)} firms, {len(prices_T.columns)} months)")

    lseg.close_session()

if __name__ == "__main__":
    main()