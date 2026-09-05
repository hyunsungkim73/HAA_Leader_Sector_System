from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
META = ROOT / "data" / "meta"
for p in (DERIVED, META):
    p.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))

# Public 2026-06-10 KOSPI200/KOSDAQ150 caches. These are used as a login-free
# fallback when KRX/pykrx index endpoints are unavailable in GitHub Actions.
K200_URL = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kospi200_cache.json"
K150_URL = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kosdaq150_cache.json"


def fetch_universe() -> tuple[list[str], dict]:
    k200 = requests.get(K200_URL, timeout=30).json()
    k150 = requests.get(K150_URL, timeout=30).json()
    a = [str(x["code"]).zfill(6) for x in k200.get("stocks", []) if x.get("code")]
    b = [str(x["code"]).zfill(6) for x in k150.get("stocks", []) if x.get("code")]
    symbols = [f"{x}.KS" for x in a] + [f"{x}.KQ" for x in b]
    symbols = sorted(set(symbols))
    meta = {
        "source": "public_github_constituent_cache",
        "kospi200_cache_updated_at": k200.get("updated_at"),
        "kosdaq150_cache_updated_at": k150.get("updated_at"),
        "kospi200_count": len(a),
        "kosdaq150_count": len(b),
        "combined_count": len(symbols),
        "urls": [K200_URL, K150_URL],
    }
    (META / "breadth_universe.json").write_text(json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2), encoding="utf-8")
    return symbols, meta


def collect_prices(symbols: list[str], start: str) -> pd.DataFrame:
    # Batch Yahoo request is much faster than one request per ticker.
    px = yf.download(symbols, start=start, auto_adjust=False, progress=False, threads=True, group_by="column")
    if px.empty:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    if isinstance(px.columns, pd.MultiIndex):
        if "Close" not in px.columns.get_level_values(0):
            return pd.DataFrame(columns=["date", "ticker", "close"])
        close = px["Close"].copy()
    else:
        close = px[["Close"]].copy()
        close.columns = symbols[:1]
    close.index = pd.to_datetime(close.index)
    long = close.stack(future_stack=True).reset_index()
    long.columns = ["date", "ticker", "close"]
    long["close"] = pd.to_numeric(long["close"], errors="coerce")
    return long.dropna(subset=["close"])


def compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    parts = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ma20"] = g["close"].rolling(20, min_periods=20).mean()
        g["sd20"] = g["close"].rolling(20, min_periods=20).std(ddof=0)
        g["upper2"] = g["ma20"] + 2 * g["sd20"]
        g["eligible"] = g["upper2"].notna()
        g["breakout"] = g["eligible"] & (g["close"] > g["upper2"])
        parts.append(g[["date", "ticker", "eligible", "breakout"]])
    z = pd.concat(parts, ignore_index=True)
    eligible = z[z["eligible"]].copy()
    agg = eligible.groupby("date").agg(
        universe_count=("ticker", "nunique"),
        breakout_count=("breakout", "sum"),
    ).reset_index()
    agg["breadth_pct"] = agg["breakout_count"] / agg["universe_count"]
    agg["breadth_5dma"] = agg["breadth_pct"].rolling(5, min_periods=1).mean()
    agg["slope"] = agg["breadth_5dma"].diff()
    agg["acceleration"] = agg["slope"].diff()
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")
    return agg


def upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    out = out.drop_duplicates(keys, keep="last").sort_values(keys)
    out.to_csv(path, index=False)


def main() -> None:
    symbols, meta = fetch_universe()
    start = (datetime.now(KST).date() - timedelta(days=430)).isoformat()
    prices = collect_prices(symbols, start)
    breadth = compute_breadth(prices)
    if breadth.empty:
        raise RuntimeError("Breadth calculation produced no rows")
    upsert(DERIVED / "breadth_bb20_2sigma_daily.csv", breadth, ["date"])
    latest = breadth.iloc[-1].to_dict()
    status = {
        "generated_at_kst": datetime.now(KST).isoformat(),
        "start_requested": start,
        "symbol_count": len(symbols),
        "price_rows": int(len(prices)),
        "breadth_rows": int(len(breadth)),
        "latest": latest,
        **meta,
    }
    (META / "breadth_last_run.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
