import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

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

def convert_to_nok(df: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """
    df: Ticker as index, 'YYYY-MM' as columns (prices or mktcap)
    fx: 'YYYY-MM' as index, currency codes as columns (SEK, DKK, EUR, ISK)
    """
    converted = df.copy()

    for ticker in converted.index:
        ccy = get_currency(ticker)
        if ccy == "NOK":
            continue

        for col in converted.columns:
            if col in fx.index and pd.notna(converted.loc[ticker, col]):
                rate = fx.loc[col, ccy]
                if pd.notna(rate):
                    converted.loc[ticker, col] = converted.loc[ticker, col] * rate

    return converted

def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate month columns (pandas renames them as YYYY-MM.1)."""
    # Strip .1, .2 suffixes back to original YYYY-MM
    clean_cols = [c.split(".")[0] if c[0].isdigit() else c for c in df.columns]

    # If no dupes after stripping, nothing to do
    if len(clean_cols) == len(set(clean_cols)):
        df.columns = clean_cols
        return df

    dupes = set(c for c in clean_cols if clean_cols.count(c) > 1)
    print(f"Found {len(dupes)} duplicate months: {sorted(dupes)[:5]}...")

    df.columns = clean_cols
    combined = df.T.groupby(df.columns).first().T
    print(f"Columns: {len(clean_cols)} → {combined.shape[1]} after dedup")
    return combined

def main():
    # Load data
    prices = pd.read_csv(DATA / "all_stock_prices.csv", index_col="Ticker")
    mktcap = pd.read_csv(DATA / "historical_market_cap.csv", index_col="Ticker")
    fx = pd.read_csv(DATA / "fx_rates.csv", index_col=0)

    # Clean duplicates
    prices = dedup_columns(prices)
    mktcap = dedup_columns(mktcap)

    print(f"Prices: {prices.shape[0]} firms, {prices.shape[1]} months")
    print(f"Market cap: {mktcap.shape[0]} firms, {mktcap.shape[1]} months")
    print(f"FX rates: {fx.shape[0]} months, currencies: {fx.columns.tolist()}")

    # Count how many firms need conversion
    for ccy in ["SEK", "DKK", "EUR", "ISK"]:
        n = sum(1 for t in prices.index if get_currency(t) == ccy)
        if n > 0:
            print(f"  {ccy}: {n} firms to convert")

    # Convert
    prices_nok = convert_to_nok(prices, fx)
    mktcap_nok = convert_to_nok(mktcap, fx)

    # Save
    prices_nok.to_csv(DATA / "all_stock_prices_nok.csv")
    mktcap_nok.to_csv(DATA / "historical_market_cap_nok.csv")

    print(f"\nSaved prices_nok: {DATA / 'all_stock_prices_nok.csv'}")
    print(f"Saved mktcap_nok: {DATA / 'historical_market_cap_nok.csv'}")

if __name__ == "__main__":
    main()