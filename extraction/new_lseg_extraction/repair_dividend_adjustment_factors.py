from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dividend_adjustment_fallbacks import (
    apply_split_fallback,
    build_yahoo_split_cache,
    load_split_events,
)


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RAW_DATA = BASE / "data" / "raw_data_lseg"

DEFAULT_INPUT = RAW_DATA / "dividends_raw_long_local.csv"
DEFAULT_OUTPUT = RAW_DATA / "dividends_raw_long_local.csv"
SPLIT_EVENTS_CSV = RAW_DATA / "split_events.csv"
YAHOO_SPLIT_EVENTS_CSV = RAW_DATA / "yahoo_split_events.csv"
YAHOO_SPLIT_STATUS_CSV = RAW_DATA / "yahoo_split_fetch_status.csv"

UNRESOLVED_BASIS_PATTERNS = (
    "missing_adjustment_factor_default_1",
    "factor_field_missing_default_1",
    "TR.DivAdjustmentFactor_blank_assumed_1",
    "TR.DivAdjustmentFactor_field_missing_assumed_1",
    "unverified_missing_adjustment_factor_default_1",
    "missing_adjustment_factor_needs_fallback",
    "factor_field_missing_needs_fallback",
    "invalid_adjustment_factor_needs_fallback",
)


def unresolved_adjustment_mask(df: pd.DataFrame) -> pd.Series:
    factor = pd.to_numeric(df.get("AdjustmentFactor"), errors="coerce")
    unresolved = factor.isna() | (factor <= 0)

    if "AdjustmentBasis" in df.columns:
        basis = df["AdjustmentBasis"].fillna("").astype(str)
        for pattern in UNRESOLVED_BASIS_PATTERNS:
            unresolved = unresolved | basis.eq(pattern)

    return unresolved


def apply_adjustment_hierarchy(
    df: pd.DataFrame,
    *,
    start: str,
    end: str,
    use_yahoo: bool,
    refresh_yahoo: bool,
    yahoo_sleep_seconds: float,
) -> pd.DataFrame:
    out = df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out["ExDate"] = pd.to_datetime(out["ExDate"], errors="coerce")
    out["DPS_UNADJUSTED"] = pd.to_numeric(out["DPS_UNADJUSTED"], errors="coerce")
    out["AdjustmentFactor"] = pd.to_numeric(out.get("AdjustmentFactor"), errors="coerce")

    if "DPS_ADJUSTED_GROSS" not in out.columns:
        out["DPS_ADJUSTED_GROSS"] = np.nan
    out["DPS_ADJUSTED_GROSS"] = pd.to_numeric(out["DPS_ADJUSTED_GROSS"], errors="coerce")

    if "AdjustmentBasis" not in out.columns:
        out["AdjustmentBasis"] = "missing_adjustment_factor_needs_fallback"

    unresolved = unresolved_adjustment_mask(out)

    # Hierarchy step 2: infer the factor from adjusted/unadjusted dividend
    # amounts when both are present.
    implied_factor = out["DPS_ADJUSTED_GROSS"] / out["DPS_UNADJUSTED"]
    valid_implied = unresolved & implied_factor.notna() & np.isfinite(implied_factor) & (implied_factor > 0)
    out.loc[valid_implied, "AdjustmentFactor"] = implied_factor.loc[valid_implied]
    out.loc[valid_implied, "AdjustmentBasis"] = "TR.DivAdjustedGross_over_TR.DivUnadjustedGross"
    unresolved = unresolved_adjustment_mask(out)

    # Hierarchy step 3: audited split/corporate-action events supplied locally.
    split_events = load_split_events(SPLIT_EVENTS_CSV)
    if unresolved.any() and not split_events.empty:
        out = apply_split_fallback(
            out,
            unresolved,
            split_events,
            split_status=None,
            source_label="split_events_csv",
        )
        unresolved = unresolved_adjustment_mask(out)

    # Hierarchy step 4: Yahoo split events as fallback/audit.
    if unresolved.any() and use_yahoo:
        tickers_need_yahoo = sorted(out.loc[unresolved, "Ticker"].dropna().unique())
        yahoo_events, yahoo_status = build_yahoo_split_cache(
            tickers_need_yahoo,
            start,
            end,
            YAHOO_SPLIT_EVENTS_CSV,
            YAHOO_SPLIT_STATUS_CSV,
            refresh=refresh_yahoo,
            sleep_seconds=yahoo_sleep_seconds,
        )
        out = apply_split_fallback(
            out,
            unresolved,
            yahoo_events,
            split_status=yahoo_status,
            source_label="yahoo",
        )
        unresolved = unresolved_adjustment_mask(out)

    # Hierarchy step 5: only unresolved rows get the explicit unverified default.
    if unresolved.any():
        out.loc[unresolved, "AdjustmentFactor"] = 1.0
        out.loc[unresolved, "AdjustmentBasis"] = "unverified_missing_adjustment_factor_default_1"

    out["DPS_LOCAL"] = out["DPS_UNADJUSTED"] * out["AdjustmentFactor"]

    ordered = [
        "Ticker",
        "ExDate",
        "DPS_UNADJUSTED",
        "DPS_ADJUSTED_GROSS",
        "AdjustmentFactor",
        "DPS_LOCAL",
        "Currency",
        "AdjustmentBasis",
        "SplitFallbackSource",
        "SplitFallbackEventCount",
        "SplitFallbackEvents",
        "InstrumentCurrency",
    ]
    return out[[c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair dividend adjustment factors in an existing raw LSEG dividend file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2005-01-03")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--no-yahoo", action="store_true", help="Do not fetch/use Yahoo split fallback.")
    parser.add_argument("--refresh-yahoo", action="store_true", help="Refresh Yahoo split cache even for cached tickers.")
    parser.add_argument("--yahoo-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak copy when output overwrites input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    before_unresolved = int(unresolved_adjustment_mask(df).sum())

    repaired = apply_adjustment_hierarchy(
        df,
        start=args.start,
        end=args.end,
        use_yahoo=not args.no_yahoo,
        refresh_yahoo=args.refresh_yahoo,
        yahoo_sleep_seconds=args.yahoo_sleep_seconds,
    )
    after_unverified = int(repaired["AdjustmentBasis"].eq("unverified_missing_adjustment_factor_default_1").sum())
    n_adjusted = int((pd.to_numeric(repaired["AdjustmentFactor"], errors="coerce").round(12) != 1.0).sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == args.input.resolve() and not args.no_backup:
        backup = args.input.with_suffix(args.input.suffix + ".bak")
        shutil.copy2(args.input, backup)
        print(f"Backup written to {backup}")

    repaired.to_csv(args.output, index=False)
    print(f"Repaired dividends written to {args.output}")
    print(f"Rows unresolved before repair: {before_unresolved:,}")
    print(f"Rows with adjustment factor != 1 after repair: {n_adjusted:,}")
    print(f"Rows still using unverified default factor: {after_unverified:,}")
    print("\nAdjustmentBasis counts:")
    print(repaired["AdjustmentBasis"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
