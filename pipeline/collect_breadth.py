from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import re
from urllib.parse import quote

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
PLUS_BASE = "https://www.plusetf.co.kr"
PLUS_PRODUCTS = {
    "k200": {"n": "006184", "title": "PLUS 200", "suffix": "KS", "expected": 200},
    "k150": {"n": "006318", "title": "PLUS 코스닥150", "suffix": "KQ", "expected": 150},
}
K200_FALLBACK = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kospi200_cache.json"
K150_FALLBACK = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kosdaq150_cache.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; HAA-Leader-Sector-System/1.0)"}


def extract_asof(page_text: str) -> str:
    patterns = [
        r'id="pdfDate"[^>]*data-max-date="([0-9-]+)"',
        r'id="pdfDate"[^>]*value="([0-9-]+)"',
        r'pdfDate[^0-9]{0,100}([0-9]{4}\.[0-9]{2}\.[0-9]{2})',
        r'([0-9]{4}\.[0-9]{2}\.[0-9]{2})',
    ]
    for pat in patterns:
        m = re.search(pat, page_text, re.S)
        if m:
            return m.group(1).replace('.', '-')
    return datetime.now(KST).date().isoformat()


def six_digit_codes_from_frame(df: pd.DataFrame) -> list[str]:
    codes = []
    for col in df.columns:
        for v in df[col].astype(str):
            s = v.strip().replace(".0", "")
            if re.fullmatch(r"\d{6}", s):
                codes.append(s)
    return list(dict.fromkeys(codes))


def download_plus_basket(n: str, title: str, suffix: str, expected: int) -> tuple[list[str], dict]:
    product_url = f"{PLUS_BASE}/product/detail?n={n}"
    session = requests.Session()
    session.headers.update(UA)
    page = session.get(product_url, timeout=30)
    page.raise_for_status()
    asof = extract_asof(page.text)
    d = asof.replace('-', '')
    excel_url = f"{PLUS_BASE}/excel/product/pdf?n={n}&d={d}&title={quote(title)}"
    r = session.get(excel_url, timeout=60)
    r.raise_for_status()
    if len(r.content) < 500:
        raise RuntimeError(f"PLUS excel payload too small: {len(r.content)} bytes; url={excel_url}")

    errors = []
    frames = []
    for engine in ["openpyxl", "xlrd", None]:
        try:
            book = pd.read_excel(BytesIO(r.content), sheet_name=None, header=None, engine=engine)
            frames = list(book.values())
            if frames:
                break
        except Exception as exc:
            errors.append(f"{engine}:{exc}")
    if not frames:
        raise RuntimeError("PLUS Excel parse failed: " + " | ".join(errors))
    codes = []
    for df in frames:
        codes.extend(six_digit_codes_from_frame(df))
    codes = list(dict.fromkeys(codes))
    etf_ticker = {"006184": "152100", "006318": "301400"}.get(n)
    if etf_ticker in codes:
        codes.remove(etf_ticker)
    if len(codes) != expected:
        m = re.search(r"(?:const|let|var)\s+etfPdfList\s*=\s*(\[.*?\]);", page.text, re.S)
        embedded = []
        if m:
            try:
                embedded = [str(x.get("jmCd", "")).strip() for x in json.loads(m.group(1))]
                embedded = list(dict.fromkeys([x for x in embedded if re.fullmatch(r"\d{6}", x)]))
            except Exception:
                pass
        raise RuntimeError(
            f"PLUS Excel stock count mismatch {title}: expected={expected}, got={len(codes)}, "
            f"embedded={len(embedded)}, asof={asof}, codes_head={codes[:12]}"
        )
    symbols = [f"{c}.{suffix}" for c in codes]
    return symbols, {
        "product_url": product_url,
        "excel_url": excel_url,
        "asof": asof,
        "stock_rows": len(codes),
        "payload_bytes": len(r.content),
    }


def fallback_universe() -> tuple[list[str], dict]:
    k200 = requests.get(K200_FALLBACK, timeout=30).json()
    k150 = requests.get(K150_FALLBACK, timeout=30).json()
    a = [str(x["code"]).zfill(6) for x in k200.get("stocks", []) if x.get("code")]
    b = [str(x["code"]).zfill(6) for x in k150.get("stocks", []) if x.get("code")]
    symbols = sorted(set([f"{x}.KS" for x in a] + [f"{x}.KQ" for x in b]))
    return symbols, {
        "source": "public_github_constituent_cache_fallback",
        "kospi200_count": len(a),
        "kosdaq150_count": len(b),
        "combined_count": len(symbols),
        "urls": [K200_FALLBACK, K150_FALLBACK],
    }


def fetch_universe() -> tuple[list[str], dict]:
    try:
        k200, m200 = download_plus_basket(**PLUS_PRODUCTS["k200"])
        k150, m150 = download_plus_basket(**PLUS_PRODUCTS["k150"])
        symbols = k200 + k150
        if len(symbols) != 350 or len(set(symbols)) != 350:
            raise RuntimeError(
                f"combined PLUS stock universe must equal 350, got {len(symbols)} / unique {len(set(symbols))}"
            )
        meta = {
            "source": "PLUS_official_excel_pdf_baskets",
            "kospi200_count": 200,
            "kosdaq150_count": 150,
            "combined_count": 350,
            "kospi200": m200,
            "kosdaq150": m150,
        }
    except Exception as exc:
        symbols, meta = fallback_universe()
        meta["official_plus_error"] = str(exc)
    (META / "breadth_universe.json").write_text(
        json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return symbols, meta


def _download_close(symbols: list[str], start: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    px = yf.download(
        symbols,
        start=start,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
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


def alternate_exchange_symbol(symbol: str) -> str | None:
    m = re.fullmatch(r"(\d{6})\.(KS|KQ)", symbol)
    if not m:
        return None
    code, suffix = m.groups()
    return f"{code}.{'KQ' if suffix == 'KS' else 'KS'}"


def collect_prices(symbols: list[str], start: str) -> tuple[pd.DataFrame, dict]:
    prices = _download_close(symbols, start)
    present = set(prices["ticker"].astype(str).unique()) if not prices.empty else set()
    missing = [s for s in symbols if s not in present]

    recovered = {}
    retry_frames = []
    for original in missing:
        alternate = alternate_exchange_symbol(original)
        if alternate is None:
            continue
        trial = _download_close([alternate], start)
        if trial.empty:
            continue
        trial = trial.copy()
        trial["ticker"] = original
        retry_frames.append(trial)
        recovered[original] = alternate

    if retry_frames:
        prices = pd.concat([prices, *retry_frames], ignore_index=True)
        prices = prices.drop_duplicates(["date", "ticker"], keep="last")

    final_present = set(prices["ticker"].astype(str).unique()) if not prices.empty else set()
    unresolved = [s for s in symbols if s not in final_present]
    retry_meta = {
        "bulk_missing_count": len(missing),
        "bulk_missing_symbols": missing,
        "alternate_suffix_recovered_count": len(recovered),
        "alternate_suffix_map": recovered,
        "unresolved_count": len(unresolved),
        "unresolved_symbols": unresolved,
    }
    return prices, retry_meta


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
        universe_count=("ticker", "nunique"), breakout_count=("breakout", "sum")
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
    prices, retry_meta = collect_prices(symbols, start)
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
        "price_retry": retry_meta,
        **meta,
    }
    (META / "breadth_last_run.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
