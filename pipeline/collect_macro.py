from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import io

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
COLS = ["obs_date","series_id","category","indicator","value","unit","frequency","source","source_type","loaded_at","notes"]


def read_old() -> pd.DataFrame:
    if OUT.exists() and OUT.stat().st_size:
        return pd.read_csv(OUT)
    return pd.DataFrame(columns=COLS)


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
    close_col = "Close"
    out = pd.DataFrame({"obs_date": pd.to_datetime(x[date_col]).dt.strftime("%Y-%m-%d"), "value": pd.to_numeric(x[close_col], errors="coerce")})
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


def main() -> None:
    old = read_old()
    if old.empty:
        start = "2024-01-01"
    else:
        dates = pd.to_datetime(old["obs_date"], errors="coerce").dropna()
        start = (dates.max().date() - timedelta(days=14)).isoformat() if not dates.empty else "2024-01-01"
    frames = [old]
    for sid, (fid, label, unit) in FRED.items():
        try:
            frames.append(fred_series(sid, fid, label, unit, start))
        except Exception as exc:
            print(f"FRED failed {sid}: {exc}")
    for sid, (ticker, label, unit) in YF.items():
        try:
            frames.append(yf_series(sid, ticker, label, unit, start))
        except Exception as exc:
            print(f"Yahoo failed {sid}: {exc}")
    z = pd.concat(frames, ignore_index=True)
    z["obs_date"] = pd.to_datetime(z["obs_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    z = z.dropna(subset=["obs_date","series_id"]).drop_duplicates(["obs_date","series_id"], keep="last")
    z = z.sort_values(["series_id","obs_date"])
    z.to_csv(OUT, index=False)
    print(z.groupby("series_id")["obs_date"].agg(["count","min","max"]).to_string())

if __name__ == "__main__":
    main()
