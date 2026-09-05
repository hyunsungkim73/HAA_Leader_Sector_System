from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import re

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

# Official PLUS ETF product pages. Each page embeds the full current PDF basket
# in JavaScript as `const etfPdfList = [...]`.  The basket contains one extra
# cash/non-stock row, so filtering to exactly six-digit stock codes yields the
# KOSPI200 200 stocks and KOSDAQ150 150 stocks.
PLUS_K200 = "https://www.plusetf.co.kr/product/detail?n=006184"  # PLUS 200 / 152100
PLUS_K150 = "https://www.plusetf.co.kr/product/detail?n=006318"  # PLUS 코스닥150 / 301400

# Older public caches remain emergency fallback only.
K200_FALLBACK = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kospi200_cache.json"
K150_FALLBACK = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kosdaq150_cache.json"

UA = {"User-Agent": "Mozilla/5.0 (compatible; HAA-Leader-Sector-System/1.0)"}


def parse_plus_pdf_basket(url: str, suffix: str, expected: int) -> tuple[list[str], dict]:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    text = r.text
    m = re.search(r"(?:const|let|var)\s+etfPdfList\s*=\s*(\[.*?\]);", text, re.S)
    if not m:
        raise RuntimeError(f"etfPdfList not found: {url}")
    basket = json.loads(m.group(1))
    codes = []
    for row in basket:
        code = str(row.get("jmCd", "")).strip()
        if re.fullmatch(r"\d{6}", code):
            codes.append(code)
    codes = list(dict.fromkeys(codes))
    if len(codes) != expected:
        raise RuntimeError(f"PLUS basket count mismatch {url}: expected={expected}, six-digit={len(codes)}, raw={len(basket)}")
    date_match = re.search(r'id="pdfDate"[^>]+data-max-date="([0-9-]+)"', text)
    asof = date_match.group(1) if date_match else (basket[0].get("wkdate") if basket else None)
    symbols = [f"{c}.{suffix}" for c in codes]
    return symbols, {"url": url, "raw_rows": len(basket), "stock_rows": len(codes), "asof": asof}


def fallback_universe() -> tuple[list[str], dict]:
    k200 = requests.get(K200_FALLBACK, timeout=30).json()
    k150 = requests.get(K150_FALLBACK, timeout=30).json()
    a = [str(x["code"]).zfill(6) for x in k200.get("stocks", []) if x.get("code")]
    b = [str(x["code"]).zfill(6) for x in k150.get("stocks", []) if x.get("code")]
    symbols = sorted(set([f"{x}.KS" for x in a] + [f"{x}.KQ" for x in b]))
    return symbols, {
        "source": "public_github_constituent_cache_fallback",
        "kospi200_count": len(a), "kosdaq150_count": len(b), "combined_count": len(symbols),
        "urls": [K200_FALLBACK, K150_FALLBACK],
    }


def fetch_universe() -> tuple[list[str], dict]:
    try:
        k200, m200 = parse_plus_pdf_basket(PLUS_K200, "KS", 200)
        k150, m150 = parse_plus_pdf_basket(PLUS_K150, "KQ", 150)
        symbols = k200 + k150
        if len(symbols) != 350 or len(set(symbols)) != 350:
            raise RuntimeError(f"combined PLUS stock universe must equal 350, got {len(symbols)} / unique {len(set(symbols))}")
        meta = {
            "source": "PLUS_official_etf_pdf_baskets",
            "kospi200_count": 200, "kosdaq150_count": 150, "combined_count": 350,
            "kospi200": m200, "kosdaq150": m150,
        }
    except Exception as exc:
        symbols, meta = fallback_universe()
        meta["official_plus_error"] = str(exc)
    (META / "breadth_universe.json").write_text(json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2), encoding="utf-8")
    return symbols, meta


def collect_prices(symbols: list[str], start: str) -> pd.DataFrame:
    px = yf.download(symbols, start=start, auto_adjust=False, progress=False, threads=True, group_by="column")
    if px.empty:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    if isinstance(px.columns, pd.MultiIndex):
        if "Close" not in px.columns.get_level_values(0):
            return pd.DataFrame(columns=["date", "ticker", "close"])
        close = px["Close"].copy()
    else:
        close = px[["Close"]].copy(); close.columns = symbols[:1]
    close.index = pd.to_datetime(close.index)
    long = close.stack(future_stack=True).reset_index(); long.columns = ["date", "ticker", "close"]
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
    agg = eligible.groupby("date").agg(universe_count=("ticker", "nunique"), breakout_count=("breakout", "sum")).reset_index()
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
        "generated_at_kst": datetime.now(KST).isoformat(), "start_requested": start,
        "symbol_count": len(symbols), "price_rows": int(len(prices)), "breadth_rows": int(len(breadth)),
        "latest": latest, **meta,
    }
    (META / "breadth_last_run.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
