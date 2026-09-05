from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import io
import os

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "macro" / "macro_market_archive.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
KST = timezone(timedelta(hours=9))

FRED = {
    "macro_us2y": ("DGS2", "미국 2년 국채금리", "%"),
    "macro_us10y": ("DGS10", "미국 10년 국채금리", "%"),
    "macro_us30y": ("DGS30", "미국 30년 국채금리", "%"),
    "macro_real10y": ("DFII10", "미국 10년 실질금리", "%"),
}
YF = {
    "macro_usdkrw": ("KRW=X", "USD/KRW", "KRW/USD"),
    "macro_dxy": ("DX-Y.NYB", "DXY", "index"),
    "macro_gold": ("GC=F", "Gold futures", "USD/oz"),
}
# BOK ECOS market-rate daily table 817Y002.
# Codes verified against public ECOS catalogue references.
ECOS = {
    "macro_kr3y": ("010200000", "한국 국고3년", "%"),
    "macro_kr10y": ("010210000", "한국 국고10년", "%"),
    "macro_cd91": ("010502000", "CD91일", "%"),
}
COLS = ["obs_date","series_id","category","indicator","value","unit","frequency","source","source_type","loaded_at","notes"]
MIN_DAILY_OBS = 100
KOREA_MIN_OBS = 65
BACKFILL_START = "2025-09-01"
KOREA_BACKFILL_START = "2026-04-01"


def read_old() -> pd.DataFrame:
    if OUT.exists() and OUT.stat().st_size:
        return pd.read_csv(OUT)
    return pd.DataFrame(columns=COLS)


def start_for_series(old: pd.DataFrame, series_id: str, minimum: int = MIN_DAILY_OBS, backfill_start: str = BACKFILL_START) -> str:
    z = old[old["series_id"].astype(str) == series_id] if not old.empty else pd.DataFrame()
    if z.empty or len(z) < minimum:
        return backfill_start
    dates = pd.to_datetime(z["obs_date"], errors="coerce").dropna()
    if dates.empty:
        return backfill_start
    return (dates.max().date() - timedelta(days=14)).isoformat()


def fred_series(series_id: str, fred_id: str, label: str, unit: str, start: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}&cosd={start}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text))
    x.columns = ["obs_date", "value"]
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x.dropna(subset=["value"])
    x["series_id"] = series_id
    x["category"] = "방어자산"
    x["indicator"] = label
    x["unit"] = unit
    x["frequency"] = "daily"
    x["source"] = f"FRED {fred_id}"
    x["source_type"] = "official"
    x["loaded_at"] = datetime.now(KST).isoformat()
    x["notes"] = "automated GitHub Actions incremental collector"
    return x[COLS]


def yf_series(series_id: str, ticker: str, label: str, unit: str, start: str) -> pd.DataFrame:
    x = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if x is None or x.empty:
        return pd.DataFrame(columns=COLS)
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [c[0] for c in x.columns]
    x = x.reset_index()
    date_col = "Date" if "Date" in x.columns else x.columns[0]
    out = pd.DataFrame({"obs_date": pd.to_datetime(x[date_col]).dt.strftime("%Y-%m-%d"), "value": pd.to_numeric(x["Close"], errors="coerce")})
    out = out.dropna(subset=["value"])
    out["series_id"] = series_id
    out["category"] = "방어자산"
    out["indicator"] = label
    out["unit"] = unit
    out["frequency"] = "daily"
    out["source"] = f"Yahoo Finance {ticker}"
    out["source_type"] = "public market"
    out["loaded_at"] = datetime.now(KST).isoformat()
    out["notes"] = "automated GitHub Actions incremental collector"
    return out[COLS]


def _ecos_request(key: str, start_row: int, end_row: int, start_compact: str, end_compact: str, item_code: str) -> dict:
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/{start_row}/{end_row}/"
        f"817Y002/D/{start_compact}/{end_compact}/{item_code}"
    )
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    payload = r.json()
    if "RESULT" in payload:
        raise RuntimeError(f"ECOS error {payload['RESULT']}")
    return payload


def _ecos_rows(key: str, start_compact: str, end_compact: str, item_code: str) -> list[dict]:
    # The documented ECOS sample key permits at most 10 rows per request.  Query the
    # first page, read list_total_count, then paginate without needing a secret key.
    # A configured real key can use larger pages to reduce request count.
    page_size = 1000 if key != "sample" else 10
    first_end = page_size
    first = _ecos_request(key, 1, first_end, start_compact, end_compact, item_code)
    block = first.get("StatisticSearch", {})
    rows = list(block.get("row", []) or [])
    total = int(block.get("list_total_count", len(rows)) or len(rows))
    if not rows and total <= 0:
        raise RuntimeError("ECOS returned no rows")
    start_row = first_end + 1
    while start_row <= total:
        end_row = min(start_row + page_size - 1, total)
        payload = _ecos_request(key, start_row, end_row, start_compact, end_compact, item_code)
        page_rows = payload.get("StatisticSearch", {}).get("row", []) or []
        if not page_rows:
            raise RuntimeError(f"ECOS pagination returned no rows for {start_row}-{end_row} of {total}")
        rows.extend(page_rows)
        start_row = end_row + 1
    return rows


def ecos_series(series_id: str, item_code: str, label: str, unit: str, start: str) -> pd.DataFrame:
    # Prefer a real API key when configured; ECOS also exposes a documented sample key
    # suitable for limited public queries. Pagination keeps sample-key requests within
    # the 10-row cap while still allowing a full 3-month backfill.
    key = os.environ.get("BOK_ECOS_API_KEY", "sample")
    start_compact = pd.Timestamp(start).strftime("%Y%m%d")
    end_compact = datetime.now(KST).strftime("%Y%m%d")
    rows = _ecos_rows(key, start_compact, end_compact, item_code)
    out = pd.DataFrame({
        "obs_date": [str(x.get("TIME", "")) for x in rows],
        "value": [pd.to_numeric(x.get("DATA_VALUE"), errors="coerce") for x in rows],
    })
    out["obs_date"] = pd.to_datetime(out["obs_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["obs_date", "value"])
    if out.empty:
        raise RuntimeError("ECOS rows contained no valid dated numeric observations")
    out["series_id"] = series_id
    out["category"] = "방어자산"
    out["indicator"] = label
    out["unit"] = unit
    out["frequency"] = "daily"
    out["source"] = f"BOK ECOS 817Y002/{item_code}"
    out["source_type"] = "official"
    out["loaded_at"] = datetime.now(KST).isoformat()
    out["notes"] = "official ECOS daily market-rate backfill/incremental collector"
    return out[COLS]


def main() -> None:
    old = read_old()
    frames = [old]
    for sid, (fid, label, unit) in FRED.items():
        try:
            frames.append(fred_series(sid, fid, label, unit, start_for_series(old, sid)))
        except Exception as exc:
            print(f"FRED failed {sid}: {exc}")
    for sid, (ticker, label, unit) in YF.items():
        try:
            frames.append(yf_series(sid, ticker, label, unit, start_for_series(old, sid)))
        except Exception as exc:
            print(f"Yahoo failed {sid}: {exc}")
    for sid, (item_code, label, unit) in ECOS.items():
        try:
            start = start_for_series(old, sid, minimum=KOREA_MIN_OBS, backfill_start=KOREA_BACKFILL_START)
            frames.append(ecos_series(sid, item_code, label, unit, start))
        except Exception as exc:
            print(f"ECOS failed {sid}: {exc}")

    z = pd.concat(frames, ignore_index=True)
    z["obs_date"] = pd.to_datetime(z["obs_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    z["value"] = pd.to_numeric(z["value"], errors="coerce")
    z = z.dropna(subset=["obs_date","series_id","value"]).drop_duplicates(["obs_date","series_id"], keep="last")
    z = z.sort_values(["series_id","obs_date"])

    # Never regress an already healthy series because a source call partially failed.
    old_counts = old.groupby("series_id").size().to_dict() if not old.empty else {}
    new_counts = z.groupby("series_id").size().to_dict() if not z.empty else {}
    for sid, n in old_counts.items():
        if new_counts.get(sid, 0) < n:
            raise RuntimeError(f"macro archive regression for {sid}: old={n} new={new_counts.get(sid, 0)}")

    z.to_csv(OUT, index=False)
    print(z.groupby("series_id")["obs_date"].agg(["count","min","max"]).to_string())

if __name__ == "__main__":
    main()
