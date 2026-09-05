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
import FinanceDataReader as fdr
from pykrx import stock

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
INDEX_CODES = {"k200": ("1028", "KS", 200), "k150": ("2203", "KQ", 150)}
K200_FALLBACK = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kospi200_cache.json"
K150_FALLBACK = "https://raw.githubusercontent.com/thebigone9414/stock/dev/data/kosdaq150_cache.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; HAA-Leader-Sector-System/1.0)"}


def extract_asof(page_text: str) -> str:
    for pat in [
        r'id="pdfDate"[^>]*data-max-date="([0-9-]+)"',
        r'id="pdfDate"[^>]*value="([0-9-]+)"',
        r'pdfDate[^0-9]{0,100}([0-9]{4}\.[0-9]{2}\.[0-9]{2})',
        r'([0-9]{4}\.[0-9]{2}\.[0-9]{2})',
    ]:
        m = re.search(pat, page_text, re.S)
        if m:
            return m.group(1).replace('.', '-')
    return datetime.now(KST).date().isoformat()


def normalize_six_digit_code(value: object) -> str | None:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if re.fullmatch(r"\d{6}", s) else None


def code_column_candidates(frames: list[pd.DataFrame], etf_ticker: str | None) -> list[dict]:
    candidates = []
    for sheet_index, df in enumerate(frames):
        for col in df.columns:
            raw = [normalize_six_digit_code(v) for v in df[col].tolist()]
            codes = list(dict.fromkeys([x for x in raw if x]))
            if etf_ticker in codes:
                codes.remove(etf_ticker)
            if not codes:
                continue
            non_null = int(df[col].notna().sum())
            candidates.append({
                "sheet": sheet_index,
                "column": str(col),
                "count": len(codes),
                "ratio": len(codes) / max(non_null, 1),
                "codes": codes,
            })
    return candidates


def choose_stock_codes(frames: list[pd.DataFrame], expected: int, etf_ticker: str | None) -> tuple[list[str], list[dict]]:
    candidates = code_column_candidates(frames, etf_ticker)
    exact = [c for c in candidates if c["count"] == expected]
    if exact:
        exact.sort(key=lambda c: c["ratio"], reverse=True)
        return exact[0]["codes"], candidates
    return [], candidates


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
        raise RuntimeError(f"PLUS excel payload too small: {len(r.content)} bytes")
    errors, frames = [], []
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
    etf_ticker = {"006184": "152100", "006318": "301400"}.get(n)
    codes, candidates = choose_stock_codes(frames, expected, etf_ticker)
    if len(codes) != expected:
        diagnostic = sorted(
            [{k: v for k, v in c.items() if k != "codes"} for c in candidates],
            key=lambda x: abs(x["count"] - expected),
        )[:8]
        raise RuntimeError(
            f"PLUS Excel stock code-column mismatch {title}: expected={expected}, selected={len(codes)}, "
            f"asof={asof}, candidate_columns={diagnostic}"
        )
    return [f"{c}.{suffix}" for c in codes], {
        "source": "PLUS_official_excel_pdf_baskets",
        "product_url": product_url,
        "excel_url": excel_url,
        "asof": asof,
        "stock_rows": len(codes),
    }


def pykrx_universe() -> tuple[list[str], dict]:
    pieces, meta = [], {"source": "pykrx_KRX_index_constituents"}
    for key, (idx, suffix, expected) in INDEX_CODES.items():
        codes = stock.get_index_portfolio_deposit_file(idx, alternative=True)
        codes = list(dict.fromkeys([str(x).zfill(6) for x in codes if re.fullmatch(r"\d{6}", str(x).zfill(6))]))
        if len(codes) != expected:
            raise RuntimeError(f"pykrx {key} expected={expected}, got={len(codes)}")
        pieces.extend([f"{c}.{suffix}" for c in codes])
        meta[f"{key}_count"] = len(codes)
    symbols = list(dict.fromkeys(pieces))
    if len(symbols) != 350:
        raise RuntimeError(f"pykrx combined universe expected=350, got={len(symbols)}")
    meta["combined_count"] = len(symbols)
    return symbols, meta


def cache_universe() -> tuple[list[str], dict]:
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
    errors = []
    try:
        k200, m200 = download_plus_basket(**PLUS_PRODUCTS["k200"])
        k150, m150 = download_plus_basket(**PLUS_PRODUCTS["k150"])
        symbols = k200 + k150
        if len(symbols) != 350 or len(set(symbols)) != 350:
            raise RuntimeError(f"PLUS combined universe expected=350, got={len(symbols)} unique={len(set(symbols))}")
        meta = {"source": "PLUS_official_excel_pdf_baskets", "combined_count": 350, "kospi200": m200, "kosdaq150": m150}
    except Exception as exc:
        errors.append(f"PLUS:{exc}")
        try:
            symbols, meta = pykrx_universe()
        except Exception as exc2:
            errors.append(f"pykrx:{exc2}")
            symbols, meta = cache_universe()
    meta["fallback_errors"] = errors
    (META / "breadth_universe.json").write_text(json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2), encoding="utf-8")
    return symbols, meta


def _download_close_yf(symbols: list[str], start: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["date", "ticker", "close"])
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


def _download_close_fdr(symbol: str, start: str) -> pd.DataFrame:
    code = symbol.split(".")[0]
    try:
        df = fdr.DataReader(code, start)
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    out = df[["Close"]].reset_index()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["ticker"] = symbol
    return out[["date", "ticker", "close"]].dropna(subset=["close"])


def alternate_exchange_symbol(symbol: str) -> str | None:
    m = re.fullmatch(r"(\d{6})\.(KS|KQ)", symbol)
    if not m:
        return None
    code, suffix = m.groups()
    return f"{code}.{'KQ' if suffix == 'KS' else 'KS'}"


def collect_prices(symbols: list[str], start: str) -> tuple[pd.DataFrame, dict]:
    prices = _download_close_yf(symbols, start)
    present = set(prices["ticker"].astype(str).unique()) if not prices.empty else set()
    missing_after_yf = [s for s in symbols if s not in present]

    fdr_frames, fdr_recovered = [], []
    for s in missing_after_yf:
        trial = _download_close_fdr(s, start)
        if not trial.empty:
            fdr_frames.append(trial)
            fdr_recovered.append(s)
    if fdr_frames:
        prices = pd.concat([prices, *fdr_frames], ignore_index=True).drop_duplicates(["date", "ticker"], keep="last")

    present = set(prices["ticker"].astype(str).unique()) if not prices.empty else set()
    missing_after_fdr = [s for s in symbols if s not in present]
    alt_frames, alt_map = [], {}
    for original in missing_after_fdr:
        alternate = alternate_exchange_symbol(original)
        if alternate is None:
            continue
        trial = _download_close_yf([alternate], start)
        if trial.empty:
            continue
        trial = trial.copy(); trial["ticker"] = original
        alt_frames.append(trial); alt_map[original] = alternate
    if alt_frames:
        prices = pd.concat([prices, *alt_frames], ignore_index=True).drop_duplicates(["date", "ticker"], keep="last")

    final_present = set(prices["ticker"].astype(str).unique()) if not prices.empty else set()
    unresolved = [s for s in symbols if s not in final_present]
    return prices, {
        "bulk_missing_count": len(missing_after_yf),
        "bulk_missing_symbols": missing_after_yf,
        "fdr_recovered_count": len(fdr_recovered),
        "fdr_recovered_symbols": fdr_recovered,
        "alternate_suffix_recovered_count": len(alt_map),
        "alternate_suffix_map": alt_map,
        "unresolved_count": len(unresolved),
        "unresolved_symbols": unresolved,
    }


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
    (META / "breadth_last_run.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
