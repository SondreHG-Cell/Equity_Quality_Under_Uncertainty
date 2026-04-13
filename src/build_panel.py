from pathlib import Path
import pandas as pd

from config import PROF_DIR, ACC_DIR


def load_panel(
    data_dir: str | Path,
    min_year: int = 2005,
    max_year: int = 2024,
) -> pd.DataFrame:
    """
    Loads all firm CSVs from data_dir into a single panel DataFrame.

    Args:
        data_dir: Path to the folder containing firm CSVs.
        min_year: Earliest year to include.
        max_year: Latest year to include.

    Returns:
        A concatenated panel DataFrame of all valid firms.
    """
    data_dir = Path(data_dir)
    rows = []
    bad_files = []

    for fp in sorted(data_dir.glob("*.csv")):
        firm = fp.stem
        df = pd.read_csv(fp)

        if "Year" not in df.columns:
            bad_files.append(fp.name)
            continue

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"]).copy()
        df["Year"] = df["Year"].astype(int)
        df = df[(df["Year"] >= min_year) & (df["Year"] <= max_year)].sort_values("Year")

        if "firm" not in df.columns:
            df["firm"] = firm
        if "Ticker" not in df.columns:
            df["Ticker"] = firm

        rows.append(df)

    if bad_files:
        print(f"Warning: {len(bad_files)} files skipped (no 'Year' column): {bad_files}")

    if not rows:
        raise ValueError("No valid CSV files were loaded into panel.")

    return pd.concat(rows, ignore_index=True)


def load_panel_prof(min_year: int = 2005, max_year: int = 2024) -> pd.DataFrame:
    return load_panel(PROF_DIR, min_year, max_year)


def load_panel_acc(min_year: int = 2005, max_year: int = 2024) -> pd.DataFrame:
    return load_panel(ACC_DIR, min_year, max_year)