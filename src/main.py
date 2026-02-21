from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = None

from src.fundamentals_config import SHEETS, METRICS

def make_unique_columns(cols: list[Any]) -> list[str]:
    cleaned = [("" if pd.isna(c) else str(c).strip()) for c in cols]
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cleaned:
        if not c:
            c = "unnamed"
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def find_header_row(df_raw: pd.DataFrame) -> int:
    col0 = df_raw.iloc[:, 0].astype(str).str.strip()
    hits = col0[col0.eq("Field Name")].index.tolist()
    if not hits:
        raise ValueError("Couldn't find 'Field Name' in column A.")
    return int(hits[0])

def extract_metadata(df_raw: pd.DataFrame, header_row: int) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    top = df_raw.iloc[:header_row, :2]
    for _, row in top.iterrows():
        k = row.iloc[0]
        if pd.isna(k) or str(k).strip() == "":
            continue
        v = row.iloc[1] if top.shape[1] > 1 else None
        meta[str(k).strip()] = None if pd.isna(v) else v
    return meta

def parse_sheet(path: Path, sheet_name: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.Timestamp]]:
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    header_row = find_header_row(df_raw)
    meta = extract_metadata(df_raw, header_row)

    cols = make_unique_columns(df_raw.iloc[header_row].tolist())
    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = cols

    df = df.rename(columns={cols[0]: "field"})
    df = df[df["field"].notna()].copy()
    df["field"] = df["field"].astype(str).str.strip()

    col_dt: dict[str, pd.Timestamp] = {}
    rename: dict[str, str] = {}
    for c in df.columns:
        if c == "field":
            continue
        dt = pd.to_datetime(str(c), errors="coerce", dayfirst=True)
        if pd.notna(dt):
            iso = dt.date().isoformat()
            rename[c] = iso
            col_dt[iso] = dt

    df = df.rename(columns=rename)

    value_cols = [c for c in df.columns if c != "field"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

    return meta, df, col_dt

def choose_period_columns(meta: dict[str, Any], col_dt: dict[str, pd.Timestamp], include_interim: bool) -> list[str]:
    if not col_dt:
        return []

    if include_interim:
        return sorted(col_dt.keys(), key=lambda c: col_dt[c])

    ped = meta.get("Period End Date")
    if hasattr(ped, "month") and hasattr(ped, "day"):
        target_md = (ped.month, ped.day)
    else:
        mds = [(dt.month, dt.day) for dt in col_dt.values()]
        target_md = max(set(mds), key=mds.count)

    annual = [c for c, dt in col_dt.items() if (dt.month, dt.day) == target_md]
    return sorted(annual, key=lambda c: col_dt[c])

def match_row_exact(wide: pd.DataFrame, field: str) -> Optional[pd.Series]:
    f_low = wide["field"].astype(str).str.lower().str.strip()
    target = field.lower().strip()
    rows = wide[f_low.eq(target)]
    if rows.empty:
        return None
    return rows.iloc[0] 

def extract_fundamentals_wide(excel_path: Path, include_interim: bool = False) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    available = set(xls.sheet_names)
    missing = [tab for tab in SHEETS.values() if tab not in available]
    if missing:
        raise ValueError(f"Missing required sheet(s): {missing}. Available: {sorted(available)}")

    stmt: dict[str, dict[str, Any]] = {}
    for key, tab in SHEETS.items():
        meta, wide, col_dt = parse_sheet(excel_path, tab)
        periods = choose_period_columns(meta, col_dt, include_interim=include_interim)
        stmt[key] = {"meta": meta, "wide": wide, "periods": periods}

    company_name = stmt["income_statement"]["meta"].get("Company Name")
    company_name = str(company_name).strip()

    sector = stmt["income_statement"]["meta"].get("TRBC Industry Group")
    sector = str(sector).strip()

    records: list[dict[str, Any]] = []
    for metric, cfg in METRICS.items():
        sheet_key = cfg["statement"]
        field = cfg["field"]

        wide = stmt[sheet_key]["wide"]
        periods = stmt[sheet_key]["periods"]

        row = match_row_exact(wide, field)
        if row is None or not periods:
            continue

        for p in periods:
            val = row.get(p, np.nan)
            if pd.isna(val):
                continue
            records.append(
                {
                    "company_name": company_name, 
                    "sector": sector, "period_end": p, 
                    "metric": metric, 
                    "value": float(val)
                }
            )

    if not records:
        return pd.DataFrame(columns=["company_name", "period_end"] + list(METRICS.keys()))

    df_long = pd.DataFrame(records).drop_duplicates(subset=["company_name", "sector", "period_end", "metric"], keep="first")

    wide_df = (
        df_long.pivot_table(index=["company_name", "sector", "period_end"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    wide_df["period_end"] = pd.to_datetime(wide_df["period_end"], errors="coerce")
    wide_df = wide_df.sort_values(["company_name", "period_end"]).reset_index(drop=True)
    wide_df["period_end"] = wide_df["period_end"].dt.date.astype(str)

    ordered = ["company_name", "sector", "period_end"] + [m for m in METRICS.keys() if m in wide_df.columns]
    return wide_df[ordered]

def move_to_processed(src: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / src.name

    if dest.exists():
        stem, suffix = src.stem, src.suffix
        i = 1
        while True:
            candidate = processed_dir / f"{stem}__{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1

    shutil.move(str(src), str(dest))
    return dest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Single Excel file (optional)")
    parser.add_argument("--folder", type=Path, default=None, help="Folder of Excel files (optional)")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "FUNDAMENTALS_SUMMARY.csv")
    parser.add_argument("--include-interim", action="store_true")

    # NEW: move processed files
    parser.add_argument(
        "--move-processed",
        action="store_true",
        help="Move successfully processed Excel files into a processed folder",
    )
    parser.add_argument(
        "--processed-folder",
        type=Path,
        default=ROOT / "processed",
        help="Where to move processed Excel files",
    )

    args = parser.parse_args()

    if args.input is None and args.folder is None:
        raise ValueError("Provide --input <file.xlsx> or --folder <folder_path>")

    out_csv = args.out if args.out.is_absolute() else (ROOT / args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if args.folder is not None:
        folder = args.folder if args.folder.is_absolute() else (ROOT / args.folder)
        files = sorted([p for p in folder.glob("*.xlsx") if not p.name.startswith("~$")])
        if not files:
            raise ValueError(f"No .xlsx files found in: {folder}")
    else:
        excel_path = args.input if args.input.is_absolute() else (ROOT / args.input)
        files = [excel_path]

    extracted: list[pd.DataFrame] = []

    for f in files:
        try:
            df = extract_fundamentals_wide(f, include_interim=args.include_interim)

            if df.empty:
                print(f"Skipped (no rows extracted): {f.name}")
                continue

            extracted.append(df)
            print(f"Extracted {len(df)} rows from: {f.name}")

            # Move only if requested AND extraction succeeded
            if args.move_processed:
                dest = move_to_processed(f, args.processed_folder)
                print(f"Moved -> {dest}")

        except Exception as e:
            print(f"Failed: {f.name} -> {e}")

    if not extracted:
        print("No data extracted from any file.")
        return

    df_all = pd.concat(extracted, ignore_index=True)

    # Sort nicely
    df_all["period_end"] = pd.to_datetime(df_all["period_end"], errors="coerce")
    df_all = df_all.sort_values(["company_name", "period_end"]).reset_index(drop=True)
    df_all["period_end"] = df_all["period_end"].dt.date.astype(str)

    df_all.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Rows written: {len(df_all)}")

if __name__ == "__main__":
    main()
