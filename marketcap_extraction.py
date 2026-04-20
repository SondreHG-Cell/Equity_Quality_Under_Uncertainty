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
OUTPUT = BASE / "market_cap/all_market_cap.csv"

def get_tickers_from_folder(folder: Path) -> list[str]:
    """Extract ticker symbols from CSV filenames in the folder."""
    tickers = [f.stem for f in folder.glob("*.csv")]
    print(f"Found {len(tickers)} tickers in '{folder}'")
    return sorted(tickers)

def get_historical_market_cap(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_mktcap = {}

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}")

        # MV = market value (price * shares outstanding) — månedlig sluttkurs
        for field in ["MV", "TR.CompanyMarketCap"]:
            try:
                df = lseg.get_history(
                    universe=[ticker],
                    fields=[field],
                    interval="monthly",
                    start=start,
                    end=end,
                )
                if df is not None and not df.empty and df[field].notna().any():
                    all_mktcap[ticker] = df[field]
                    break
            except Exception:
                continue

        time.sleep(0.2)

    print(f"\nData retrieved for {len(all_mktcap)}/{len(tickers)} tickers.")
    return pd.DataFrame(all_mktcap)

def main():
    print("Connecting to LSEG Data Platform...")
    lseg.open_session("desktop.workspace", app_key=app_key)
    print("Connected successfully.")

    # Lag output-mappen hvis den ikke finnes
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    tickers = get_tickers_from_folder(FOLDER)
    print(f"Fetching monthly market cap ({START} → {END})...")

    mktcap = get_historical_market_cap(tickers, START, END)

    # Transpose: tickers som rader, datoer som kolonner
    mktcap_T = mktcap.T
    mktcap_T.index.name = "Ticker"
    mktcap_T.columns = [d.strftime("%Y-%m") for d in mktcap_T.columns]

    # Fjern selskaper uten noe data
    mktcap_T = mktcap_T.dropna(how="all")
    mktcap_T.to_csv(OUTPUT)

    print(f"Saved to {OUTPUT} ({len(mktcap_T)} firms, {len(mktcap_T.columns)} months)")
    lseg.close_session()

if __name__ == "__main__":
    main()