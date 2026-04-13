"""
data_loader.py
--------------
Loads and merges prof and accrual CSV files for all tickers.
Paths are configured centrally in config.py.
"""

import logging
import pandas as pd
from pathlib import Path

from config import PROF_DIR, ACC_DIR

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------

PROF_COLS_REQUIRED = ["Year", "Ticker", "PROF"]
PROF_COLS_OPTIONAL = ["CompanyName", "Industry", "Sector",
                       "REVT", "COGS", "XSGA", "XRD", "XINT", "BE", "MIB",
                       "COGS_DA", "XSGA_DA"]

ACC_COLS_REQUIRED  = ["Year", "Ticker", "ACT", "CHE", "LCT", "AT", "OANCF", "PPEGT"]
ACC_COLS_ZERO_FILL = ["STD", "TXP"]


# ---------------------------------------------------------------------------
# Single-ticker loader
# ---------------------------------------------------------------------------

def _load_single(path: Path, zero_fill_cols: list[str] | None = None) -> pd.DataFrame:
    """Read one CSV, enforce Year as int, optionally zero-fill selected columns."""
    df = pd.read_csv(path)
    df["Year"] = df["Year"].astype(int)

    if zero_fill_cols:
        for col in zero_fill_cols:
            if col in df.columns:
                n_missing = df[col].isna().sum()
                if n_missing > 0:
                    log.debug("  %s: zero-filling %d missing values in %s",
                              path.stem, n_missing, col)
                df[col] = df[col].fillna(0.0)

    return df


def _drop_missing_required(df: pd.DataFrame, required: list[str],
                            source_label: str) -> pd.DataFrame:
    """Drop rows where any required column is null; log how many were dropped."""
    mask_bad = df[required].isna().any(axis=1)
    n_bad = mask_bad.sum()
    if n_bad > 0:
        log.warning("  %s: dropping %d rows with missing required columns %s",
                    source_label, n_bad, required)
    return df[~mask_bad].copy()


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all(
    prof_dir: str | Path = PROF_DIR,
    acc_dir:  str | Path = ACC_DIR,
) -> pd.DataFrame:
    """
    Load all tickers from both directories and return a merged DataFrame.

    Parameters
    ----------
    prof_dir : path to prof CSV folder (defaults to config.PROF_DIR)
    acc_dir  : path to accrual CSV folder (defaults to config.ACC_DIR)

    Returns
    -------
    pd.DataFrame with one row per (Ticker, Year).
    """
    prof_dir = Path(prof_dir)
    acc_dir  = Path(acc_dir)

    prof_files = {p.stem: p for p in sorted(prof_dir.glob("*.csv"))}
    acc_files  = {p.stem: p for p in sorted(acc_dir.glob("*.csv"))}

    tickers_prof = set(prof_files.keys())
    tickers_acc  = set(acc_files.keys())

    only_in_prof = tickers_prof - tickers_acc
    only_in_acc  = tickers_acc  - tickers_prof
    common       = tickers_prof & tickers_acc

    if only_in_prof:
        log.warning("Tickers in prof but NOT in acc (%d): %s",
                    len(only_in_prof), sorted(only_in_prof))
    if only_in_acc:
        log.warning("Tickers in acc but NOT in prof (%d): %s",
                    len(only_in_acc), sorted(only_in_acc))

    log.info("Loading %d tickers present in both directories.", len(common))

    frames = []

    for ticker in sorted(common):
        df_prof = _load_single(prof_files[ticker])
        df_prof = _drop_missing_required(df_prof, PROF_COLS_REQUIRED,
                                         f"{ticker}/prof")

        df_acc = _load_single(acc_files[ticker], zero_fill_cols=ACC_COLS_ZERO_FILL)
        df_acc = _drop_missing_required(df_acc, ACC_COLS_REQUIRED,
                                        f"{ticker}/acc")

        acc_drop = [c for c in df_acc.columns
                    if c in df_prof.columns and c not in ("Year", "Ticker")]
        df_acc_clean = df_acc.drop(columns=acc_drop)

        merged = pd.merge(df_prof, df_acc_clean, on=["Year", "Ticker"], how="inner")

        if merged.empty:
            log.warning("  %s: no overlapping years after merge — skipped.", ticker)
            continue

        frames.append(merged)

    if not frames:
        raise ValueError("No data loaded. Check directory paths and file names.")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["Ticker", "Year"]).reset_index(drop=True)

    log.info("Loaded %d firm-years across %d tickers.",
             len(data), data["Ticker"].nunique())

    return data


# ---------------------------------------------------------------------------
# Quick summary helper
# ---------------------------------------------------------------------------

def describe_dataset(data: pd.DataFrame) -> None:
    """Print a brief summary of the loaded dataset."""
    print(f"\n{'='*55}")
    print(f"  Dataset summary")
    print(f"{'='*55}")
    print(f"  Firm-years      : {len(data):,}")
    print(f"  Unique tickers  : {data['Ticker'].nunique():,}")
    print(f"  Year range      : {data['Year'].min()} – {data['Year'].max()}")
    print(f"  Columns         : {len(data.columns)}")
    print(f"\n  Missing values per column (where > 0):")
    missing = data.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("    None")
    else:
        for col, n in missing.items():
            print(f"    {col:<20} {n:>6} ({100*n/len(data):.1f}%)")
    print(f"{'='*55}\n")