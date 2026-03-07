from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "annual_xslx"          # your workbooks
MAPPINGS_DIR = BASE_DIR / "mappings"          # JSON outputs from LLM
OUT_DIR = BASE_DIR / "prof_components"        # firm-level CSVs

SKIPROWS = 18  # start at Excel row 19, consistent with your mapping step


# ---------- helpers ----------
def _clean_label(s) -> str:
    return "" if s is None else str(s).strip()

def first_nonnull_series(series_list: List[pd.Series]) -> pd.Series:
    """
    Given multiple series (same concept, different labels), return a series that for each period
    takes the first non-null value in the given priority order.
    """
    if not series_list:
        return pd.Series(dtype=float)

    # outer align on all indices
    df = pd.concat(series_list, axis=1)
    # bfill across columns moves the first non-null to the leftmost position
    # then take first column
    out = df.bfill(axis=1).iloc[:, 0]
    return out

def _to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    # handle european formats and thousand separators
    s = s.replace("\u00a0", "").replace(" ", "")
    # if both comma and dot exist, assume comma is thousands separator
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        # if only comma, treat comma as decimal
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_statement_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Read a full sheet (all columns) starting from statement section.
    Column A contains labels; remaining columns are year/period columns.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, skiprows=SKIPROWS)
    if df.shape[1] == 0:
        return df

    # ensure first column is treated as label column
    label_col = df.columns[0]
    df[label_col] = df[label_col].apply(_clean_label)

    # drop completely empty label rows
    df = df[df[label_col] != ""].copy()

    return df


def find_row(df: pd.DataFrame, row_label: str) -> Optional[pd.Series]:
    """
    Find the first row where column A equals row_label (exact match).
    """
    if df.empty:
        return None
    label_col = df.columns[0]
    hit = df[df[label_col] == row_label]
    if hit.empty:
        return None
    return hit.iloc[0]


def extract_series_from_row(row: pd.Series) -> pd.Series:
    """
    Convert numeric columns of a row into a pandas Series indexed by column names (years/periods).
    """
    # drop label column (first)
    vals = row.iloc[1:]
    vals = vals.apply(_to_float)
    vals.index = [str(c).strip() for c in vals.index]
    return vals


def sum_series(series_list: List[pd.Series]) -> pd.Series:
    """
    Align and sum multiple series (outer join on columns/years).
    """
    if not series_list:
        return pd.Series(dtype=float)
    df = pd.concat(series_list, axis=1)
    return df.sum(axis=1, min_count=1)


# ---------- main extraction ----------
def load_mapping(mapping_path: Path) -> Dict:
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def extract_components_for_firm(xlsx_path: Path, mapping_path: Path) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by year/period with PROF components as columns.
    Columns: REVT, COGS, XSGA_COMPONENTS, XRD, XINT, BE, MIB
    """
    mapping = load_mapping(mapping_path)["variables"]

    # cache sheets so we only read each once
    sheet_cache: Dict[str, pd.DataFrame] = {}

    # store extracted series per variable
    out: Dict[str, pd.Series] = {}

    for item in mapping:
        var = item["variable"]
        choices = item["final_choice"]  # list[{"sheet_name","row_label"}]

        if not choices:
            out[var] = pd.Series(dtype=float)  # will be filled with 0 after we establish index
            continue

        series_list = []
        for ch in choices:
            sheet = ch["sheet_name"]
            label = ch["row_label"]

            if sheet not in sheet_cache:
                sheet_cache[sheet] = read_statement_sheet(xlsx_path, sheet)

            row = find_row(sheet_cache[sheet], label)
            if row is None:
                continue
            series_list.append(extract_series_from_row(row))

        if var in {"XINT", "REVT"}:
            out[var] = first_nonnull_series(series_list)
        else:
            out[var] = sum_series(series_list)

    # create a union index of all years present across variables
    all_years = set()
    for s in out.values():
        all_years.update(list(s.index))
    all_years = sorted(all_years)

    df_out = pd.DataFrame(index=all_years)
    for var, s in out.items():
        df_out[var] = s.reindex(all_years)

    # any missing variable series becomes 0 (your rule for empty final_choice)
    for var in ["REVT", "COGS", "XSGA_COMPONENTS", "XRD", "XINT", "BE", "MIB"]:
        if var not in df_out.columns:
            df_out[var] = 0.0
        else:
            df_out[var] = df_out[var].fillna(0.0)

    # Optional: drop non-year columns if your headers include non-year text.
    # If your columns are like "31-12-2025", keep them as-is.
    df_out.index.name = "Period"

    return df_out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping_files = sorted(MAPPINGS_DIR.glob("*.json"))
    if not mapping_files:
        raise FileNotFoundError(f"No mappings found in {MAPPINGS_DIR}")

    for mp in mapping_files:
        firm_id = mp.stem
        xlsx_path = INPUT_DIR / f"{firm_id}.xlsx"
        if not xlsx_path.exists():
            print(f"[skip] missing workbook for {firm_id}")
            continue

        df = extract_components_for_firm(xlsx_path, mp)
        out_path = OUT_DIR / f"{firm_id}.csv"
        df.to_csv(out_path, index=True)
        print(f"[ok] {firm_id} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()