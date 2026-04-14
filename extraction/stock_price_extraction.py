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
FOLDER = BASE / "PROF/prof_components_extracted"
START = "2004-01-01"
END = "2024-12-31"
OUTPUT = BASE / "stock_prices/all_stock_prices.csv"

def get_tickers_from_folder(folder: Path) -> list[str]:
    """Extract ticker symbols from CSV filenames in the folder."""
    tickers = [f.stem for f in folder.glob("*.csv")]
    print(f"Found {len(tickers)} tickers in '{folder}'")
    return sorted(tickers)

def get_historical_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_prices = {}

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
                    break
            except Exception:
                continue
        
        time.sleep(0.2)

    print(f"\nData retrieved for {len(all_prices)}/{len(tickers)} tickers.")
    return pd.DataFrame(all_prices)

def main():

    print("Connecting to LSEG Data Platform...")
    lseg.open_session("desktop.workspace", app_key=app_key)
    print("Connected successfully.")

    tickers = get_tickers_from_folder(FOLDER)

    print(f"Fetching monthly prices ({START} → {END})...")
    prices = get_historical_prices(tickers, START, END)

    # Transpose: tickers as rows, dates as columns
    prices_T = prices.T
    prices_T.index.name = "Ticker"
    prices_T.columns = [d.strftime("%Y-%m") for d in prices_T.columns]

    # Drop firms with no data at all
    prices_T = prices_T.dropna(how="all")

    prices_T.to_csv(OUTPUT)
    print(f"Saved to {OUTPUT} ({len(prices_T)} firms, {len(prices_T.columns)} months)")

    lseg.close_session()

if __name__ == "__main__":
    main()