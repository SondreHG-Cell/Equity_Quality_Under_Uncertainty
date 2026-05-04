from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


YAHOO_TICKER_OVERRIDES = {
    "20202.OL": "2020.OL",
    "ACADE.ST": "ACAD.ST",
    "AFGA.OL": "AFG.OL",
    "AKSOA.OL": "AKSO.OL",
    "ALIVsdb.ST": "ALIV-SDB.ST",
    "ARCHA.OL": "ARCH.OL",
    "ARRA.OL": "ARR.OL",
    "ATTE.ST": "ATT.ST",
    "BERAHF.IC": "OLGERD.IC",
    "BOLJ.ST": "BOL.ST",
    "BRGB.OL": "BRG.OL",
    "BRIMH.IC": "BRIM.IC",
    "CLOEb.ST": "CLA-B.ST",
    "CPHCAPST.CO": "CPHCAP-ST.CO",
    "CTTS.ST": "CTT.ST",
    "DEDIC.ST": "DEDI.ST",
    "EIMS.IC": "EIM.IC",
    "ENSG.CO": "ESG.CO",
    "EPEND.ST": "EPEN.ST",
    "EPIRa.ST": "EPI-A.ST",
    "EVOG.ST": "EVO.ST",
    "GSFG.OL": "GSF.OL",
    "HAPD.IC": "HAMP.IC",
    "HHDC.CO": "HH.CO",
    "HUMAN.ST": "HUM.ST",
    "JINJ.OL": "JIN.OL",
    "K2APREF.ST": "K2A-PREF.ST",
    "KALDA.IC": "KALD.IC",
    "KARNO.ST": "KAR.ST",
    "KCCK.OL": "KCC.OL",
    "KCRA.HE": "KCR.HE",
    "LIMET.ST": "LIME.ST",
    "LUNG.ST": "LUG.ST",
    "MEREN.ST": "MER.ST",
    "NETCG.CO": "NET.CO",
    "NOVOb.CO": "NOVO-B.CO",
    "NOHOP.HE": "NOHO.HE",
    "NYFO.ST": "NYF.ST",
    "ODLO.OL": None,
    "PANDXb.ST": "PNDX-B.ST",
    "PENR.OL": "PEN.OL",
    "PRSO.OL": None,
    "RAILG.ST": "RAIL.ST",
    "SHOTE.ST": "SHOT.ST",
    "SIGR.CO": "SIG.CO",
    "SPGP.CO": "SPG.CO",
    "STOGR.CO": "STG.CO",
    "VOLVb.ST": "VOLV-B.ST",
    "ATCOa.ST": "ATCO-A.ST",
    "ASSAb.ST": "ASSA-B.ST",
    "ERICb.ST": "ERIC-B.ST",
    "HMb.ST": "HM-B.ST",
}

SPLIT_EVENT_COLUMNS = ["Ticker", "YahooTicker", "SplitDate", "SplitRatio", "Source"]
SPLIT_STATUS_COLUMNS = [
    "Ticker",
    "YahooTicker",
    "FetchStatus",
    "FetchMessage",
    "PriceRows",
    "SplitRows",
    "FetchedAt",
]


def yahoo_ticker_for(lseg_ticker: str) -> str | None:
    """Map local LSEG share-class ticker spelling to Yahoo's ticker spelling."""
    ticker = str(lseg_ticker).strip()

    if ticker in YAHOO_TICKER_OVERRIDES:
        return YAHOO_TICKER_OVERRIDES[ticker]

    # Only convert lowercase a/b class suffixes.
    # Examples:
    #   VOLVb.ST -> VOLV-B.ST
    #   ATCOa.ST -> ATCO-A.ST
    #
    # Do NOT convert:
    #   ABB.ST  -> AB-B.ST
    #   AAK.ST  -> AA-K.ST
    #   8TRA.ST -> 8TR-A.ST
    match = re.match(r"^(.+?)([ab])\.(ST|CO)$", ticker)
    if match:
        return f"{match.group(1)}-{match.group(2).upper()}.{match.group(3)}"

    return ticker


def load_split_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SPLIT_EVENT_COLUMNS)

    df = pd.read_csv(path)
    for col in SPLIT_EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["YahooTicker"] = df["YahooTicker"].fillna(df["Ticker"]).astype(str).str.strip()
    df["SplitDate"] = pd.to_datetime(df["SplitDate"], errors="coerce")
    df["SplitRatio"] = pd.to_numeric(df["SplitRatio"], errors="coerce")
    df["Source"] = df["Source"].fillna("split_events_csv").astype(str)

    df = df.dropna(subset=["Ticker", "SplitDate", "SplitRatio"])
    df = df.loc[df["SplitRatio"] > 0].copy()
    return df[SPLIT_EVENT_COLUMNS].sort_values(["Ticker", "SplitDate"]).reset_index(drop=True)


def load_split_fetch_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SPLIT_STATUS_COLUMNS)

    df = pd.read_csv(path)
    for col in SPLIT_STATUS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["YahooTicker"] = df["YahooTicker"].fillna(df["Ticker"]).astype(str).str.strip()
    df["FetchStatus"] = df["FetchStatus"].fillna("unknown").astype(str)
    return df[SPLIT_STATUS_COLUMNS].drop_duplicates(subset=["Ticker"], keep="last")


def _empty_split_result(ticker: str, yahoo_ticker: str, status: str, message: str) -> tuple[pd.DataFrame, dict]:
    status_row = {
        "Ticker": ticker,
        "YahooTicker": yahoo_ticker,
        "FetchStatus": status,
        "FetchMessage": message,
        "PriceRows": 0,
        "SplitRows": 0,
        "FetchedAt": pd.Timestamp.utcnow().isoformat(),
    }
    return pd.DataFrame(columns=SPLIT_EVENT_COLUMNS), status_row


def fetch_yahoo_split_events(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """
    Fetch Yahoo split events for one ticker.

    A successful Yahoo price history with zero split rows is informative: it
    verifies that Yahoo has no split event in the requested window. An empty
    Yahoo download is not informative, so it should not justify defaulting to 1.
    """
    yahoo_ticker = yahoo_ticker_for(ticker)
    if yahoo_ticker is None:
        return _empty_split_result(
            ticker,
            "",
            "no_yahoo_mapping",
            "Ticker map marks this LSEG ticker as having no Yahoo Finance data.",
        )

    try:
        import yfinance as yf
    except ImportError:
        return _empty_split_result(ticker, yahoo_ticker, "missing_yfinance", "Install yfinance to use Yahoo split fallback.")

    try:
        hist = yf.download(
            yahoo_ticker,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return _empty_split_result(ticker, yahoo_ticker, type(exc).__name__, str(exc))

    if hist is None or hist.empty:
        return _empty_split_result(
            ticker,
            yahoo_ticker,
            "empty_download",
            "Yahoo returned no rows. This is often a ticker mapping issue or rate limit.",
        )

    if isinstance(hist.columns, pd.MultiIndex):
        if yahoo_ticker in hist.columns.get_level_values(-1):
            hist = hist.xs(yahoo_ticker, axis=1, level=-1)
        elif yahoo_ticker in hist.columns.get_level_values(0):
            hist = hist.xs(yahoo_ticker, axis=1, level=0)
        else:
            hist = hist.copy()
            hist.columns = hist.columns.get_level_values(0)

    price_rows = int(len(hist))
    if "Stock Splits" not in hist.columns:
        events = pd.DataFrame(columns=SPLIT_EVENT_COLUMNS)
    else:
        splits = pd.to_numeric(hist["Stock Splits"], errors="coerce")
        splits = splits.loc[splits.notna() & (splits > 0) & (splits != 1)]
        events = pd.DataFrame({
            "Ticker": ticker,
            "YahooTicker": yahoo_ticker,
            "SplitDate": pd.to_datetime(splits.index).tz_localize(None).normalize(),
            "SplitRatio": splits.astype(float).values,
            "Source": "yahoo_stock_splits",
        })

    status_row = {
        "Ticker": ticker,
        "YahooTicker": yahoo_ticker,
        "FetchStatus": "ok",
        "FetchMessage": "",
        "PriceRows": price_rows,
        "SplitRows": int(len(events)),
        "FetchedAt": pd.Timestamp.utcnow().isoformat(),
    }
    return events[SPLIT_EVENT_COLUMNS], status_row


def build_yahoo_split_cache(
    tickers: list[str],
    start: str,
    end: str,
    event_cache_path: Path,
    status_cache_path: Path,
    *,
    refresh: bool = False,
    sleep_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load/fill a Yahoo split-event cache for the requested tickers.

    The status cache matters because "no split rows" is only a verified no-split
    result when Yahoo returned price history successfully.
    """
    event_cache_path.parent.mkdir(parents=True, exist_ok=True)
    status_cache_path.parent.mkdir(parents=True, exist_ok=True)

    events = load_split_events(event_cache_path)
    status = load_split_fetch_status(status_cache_path)
    terminal_statuses = {"ok", "no_yahoo_mapping"}
    cached_rows = status.loc[status["FetchStatus"].isin(terminal_statuses)].copy()
    if not cached_rows.empty:
        cached_rows["CurrentYahooTicker"] = cached_rows["Ticker"].map(lambda ticker: yahoo_ticker_for(ticker) or "")
        cached_rows = cached_rows.loc[cached_rows["YahooTicker"].fillna("").eq(cached_rows["CurrentYahooTicker"])]
    ok_cached = set(cached_rows["Ticker"])
    requested = [str(t).strip() for t in tickers if str(t).strip()]

    to_fetch = requested if refresh else [ticker for ticker in requested if ticker not in ok_cached]
    fetched_event_frames = []
    fetched_status_rows = []

    for i, ticker in enumerate(to_fetch):
        print(f"    Yahoo split fallback [{i + 1}/{len(to_fetch)}] {ticker} as {yahoo_ticker_for(ticker)}")
        ticker_events, ticker_status = fetch_yahoo_split_events(ticker, start, end)
        fetched_event_frames.append(ticker_events)
        fetched_status_rows.append(ticker_status)
        if sleep_seconds:
            time.sleep(sleep_seconds)

    if fetched_status_rows:
        fetched_tickers = {row["Ticker"] for row in fetched_status_rows}
        events = events.loc[~events["Ticker"].isin(fetched_tickers)].copy()
        status = status.loc[~status["Ticker"].isin(fetched_tickers)].copy()

        fetched_events = pd.concat(fetched_event_frames, ignore_index=True) if fetched_event_frames else pd.DataFrame(columns=SPLIT_EVENT_COLUMNS)
        fetched_status = pd.DataFrame(fetched_status_rows)
        events = pd.concat([events, fetched_events], ignore_index=True)
        status = pd.concat([status, fetched_status], ignore_index=True)

        events = events[SPLIT_EVENT_COLUMNS].sort_values(["Ticker", "SplitDate"]).reset_index(drop=True)
        status = status[SPLIT_STATUS_COLUMNS].drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
        events.to_csv(event_cache_path, index=False)
        status.to_csv(status_cache_path, index=False)

    return events, status


def split_adjustment_for_ex_date(
    ticker: str,
    ex_date: pd.Timestamp,
    split_events: pd.DataFrame,
    split_status: pd.DataFrame | None = None,
    *,
    source_label: str = "yahoo",
) -> tuple[float, str, int, str]:
    """
    Return (factor, basis, event_count, event_detail) for one dividend ex-date.

    Split ratios are interpreted as new shares per old share. A later 5-for-1
    split therefore scales historical dividends by 1/5.
    """
    if pd.isna(ex_date):
        return np.nan, f"{source_label}_split_unavailable", 0, ""

    ticker = str(ticker).strip()
    events = split_events.loc[split_events["Ticker"].astype(str).str.strip().eq(ticker)].copy()
    events["SplitDate"] = pd.to_datetime(events["SplitDate"], errors="coerce")
    events["SplitRatio"] = pd.to_numeric(events["SplitRatio"], errors="coerce")
    later = events.loc[(events["SplitDate"] > pd.Timestamp(ex_date)) & (events["SplitRatio"] > 0)].sort_values("SplitDate")

    if not later.empty:
        factor = float((1.0 / later["SplitRatio"]).prod())
        details = ";".join(
            f"{row.SplitDate.date()}:{float(row.SplitRatio):g}"
            for row in later.itertuples(index=False)
        )
        return factor, f"{source_label}_split_events", int(len(later)), details

    if split_status is not None and not split_status.empty:
        status_rows = split_status.loc[split_status["Ticker"].astype(str).str.strip().eq(ticker)]
        if not status_rows.empty and str(status_rows.iloc[-1]["FetchStatus"]) == "ok":
            return 1.0, f"{source_label}_no_later_split_verified_1", 0, ""

    return np.nan, f"{source_label}_split_unavailable", 0, ""


def apply_split_fallback(
    dividends: pd.DataFrame,
    unresolved_mask: pd.Series,
    split_events: pd.DataFrame,
    split_status: pd.DataFrame | None = None,
    *,
    source_label: str = "yahoo",
) -> pd.DataFrame:
    out = dividends.copy()
    for col, default in [
        ("SplitFallbackSource", ""),
        ("SplitFallbackEventCount", 0),
        ("SplitFallbackEvents", ""),
    ]:
        if col not in out.columns:
            out[col] = default

    for idx in out.index[unresolved_mask]:
        factor, basis, event_count, event_detail = split_adjustment_for_ex_date(
            out.at[idx, "Ticker"],
            out.at[idx, "ExDate"],
            split_events,
            split_status,
            source_label=source_label,
        )
        if pd.notna(factor) and factor > 0:
            out.at[idx, "AdjustmentFactor"] = factor
            out.at[idx, "AdjustmentBasis"] = basis
            out.at[idx, "SplitFallbackSource"] = source_label
            out.at[idx, "SplitFallbackEventCount"] = event_count
            out.at[idx, "SplitFallbackEvents"] = event_detail

    return out
