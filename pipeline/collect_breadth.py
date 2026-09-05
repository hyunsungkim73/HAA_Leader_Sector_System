from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
from pykrx import stock

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
META = ROOT / "data" / "meta"
for p in (RAW, DERIVED, META):
    p.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))

# KRX index tickers commonly used by pykrx. If upstream changes, the collector logs failure.
KOSPI200_INDEX = "1028"
KOSDAQ150_INDEX = "2203"


def get_constituents(index_code: str, date: str) -> list[str]:
    return list(stock.get_index_portfolio_deposit_file(index_code, date))


def latest_business_day(max_lookback: int = 10) -> str:
    d = datetime.now(KST).date()
    for _ in range(max_lookback):
        s = d.strftime("%Y%m%d")
        try:
            if stock.get_market_ticker_list(s, market="ALL"):
                return s
        except Exception:
            pass
        d -= timedelta(days=1)
    raise RuntimeError("Unable to resolve latest KRX business day")


def load_or_build_universe(asof: str) -> tuple[list[str], dict]:
    cache = META / "breadth_universe.json"
    details = {"asof": asof, "source": "pykrx", "fallback": False}
    try:
        k200 = get_constituents(KOSPI200_INDEX, asof)
        k150 = get_constituents(KOSDAQ150_INDEX, asof)
        tickers = sorted(set(k200 + k150))
        if len(tickers) < 300:
            raise RuntimeError(f"unexpectedly small universe: {len(tickers)}")
        payload = {"asof": asof, "kospi200": k200, "kosdaq150": k150, "combined_count": len(tickers)}
        cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return tickers, details
    except Exception as exc:
        details["fallback"] = True
        details["error"] = str(exc)
        if not cache.exists():
            raise
        payload = json.loads(cache.read_text(encoding="utf-8"))
        tickers = sorted(set(payload.get("kospi200", []) + payload.get("kosdaq150", [])))
        details["source"] = "cached_universe"
        details["cache_asof"] = payload.get("asof")
        return tickers, details


def collect_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for i, ticker in enumerate(tickers, 1):
        try:
            x = stock.get_market_ohlcv_by_date(start, end, ticker)
            if x is None or x.empty:
                continue
            x = x.reset_index().rename(columns={"날짜": "date", "종가": "close"})
            if "date" not in x.columns:
                x = x.rename(columns={x.columns[0]: "date"})
            if "close" not in x.columns and "종가" in x.columns:
                x = x.rename(columns={"종가": "close"})
            x = x[["date", "close"]].copy()
            x["ticker"] = ticker
            frames.append(x)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["date", "close", "ticker"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["close"])


def compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    rows = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ma20"] = g["close"].rolling(20, min_periods=20).mean()
        g["sd20"] = g["close"].rolling(20, min_periods=20).std(ddof=0)
        g["upper2"] = g["ma20"] + 2*g["sd20"]
        g["breakout"] = g["close"] > g["upper2"]
        rows.append(g[["date", "ticker", "breakout"]])
    z = pd.concat(rows, ignore_index=True)
    agg = z.groupby("date").agg(
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
    end = latest_business_day()
    end_date = datetime.strptime(end, "%Y%m%d").date()
    start_date = end_date - timedelta(days=430)
    start = start_date.strftime("%Y%m%d")
    universe, details = load_or_build_universe(end)
    prices = collect_prices(universe, start, end)
    breadth = compute_breadth(prices)
    if not breadth.empty:
        upsert(DERIVED / "breadth_bb20_2sigma_daily.csv", breadth, ["date"])
    status = {
        "generated_at_kst": datetime.now(KST).isoformat(),
        "start": start,
        "end": end,
        "universe_count": len(universe),
        "price_rows": int(len(prices)),
        "breadth_rows": int(len(breadth)),
        **details,
    }
    (META / "breadth_last_run.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
