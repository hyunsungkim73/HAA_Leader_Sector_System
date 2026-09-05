from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from io import StringIO

import pandas as pd
import requests
import FinanceDataReader as fdr
from pykrx import stock

import collect_breadth as cb

KST = timezone(timedelta(hours=9))
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36"}
INDEXES = {
    "k200": ("1028", "KS", 200),
    "k150": ("2203", "KQ", 150),
}
SOURCES = {
    "k200": ("https://kr.investing.com/indices/kospi-200-components", "KS", 200),
    "k150": ("https://kr.investing.com/indices/kosdaq-150-components", "KQ", 150),
}


def _norm_name(value: object) -> str:
    s = str(value).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9A-Z가-힣]", "", s)
    return s


def _clean_codes(values: list[object]) -> list[str]:
    codes: list[str] = []
    for value in values:
        s = str(value).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s.isdigit() and 1 <= len(s) <= 6:
            codes.append(s.zfill(6))
    return list(dict.fromkeys(codes))


def _pykrx_exact_universe() -> tuple[list[str], dict]:
    errors: list[str] = []
    today = datetime.now(KST).date()
    for days_back in range(1, 15):
        d = (today - timedelta(days=days_back)).strftime("%Y%m%d")
        pieces: list[str] = []
        detail: dict[str, dict] = {}
        try:
            for key, (index_code, suffix, expected) in INDEXES.items():
                raw = stock.get_index_portfolio_deposit_file(index_code, date=d, alternative=True)
                codes = _clean_codes(list(raw))
                if len(codes) != expected:
                    raise RuntimeError(f"{key} expected={expected}, got={len(codes)}")
                pieces.extend([f"{code}.{suffix}" for code in codes])
                detail[key] = {"index_code": index_code, "expected": expected, "count": len(codes)}
            symbols = list(dict.fromkeys(pieces))
            if len(symbols) != 350:
                raise RuntimeError(f"combined expected=350, got={len(symbols)}")
            return symbols, {
                "source": "pykrx_KRX_index_constituents",
                "asof_requested": d,
                "combined_count": 350,
                "sources": detail,
                "prior_attempt_errors": errors,
            }
        except Exception as exc:
            errors.append(f"{d}:{type(exc).__name__}:{exc}")
    raise RuntimeError("pykrx exact universe failed across recent dates: " + " | ".join(errors[-5:]))


def _component_names(url: str, expected: int) -> list[str]:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    candidates: list[list[str]] = []
    for t in tables:
        if t.empty:
            continue
        first = t.columns[0]
        vals = [str(x).strip() for x in t[first].dropna().tolist()]
        vals = [x for x in vals if x and x.lower() not in {"종목명", "name"}]
        uniq = list(dict.fromkeys(vals))
        if len(uniq) >= expected:
            candidates.append(uniq[:expected])
    if not candidates:
        raise RuntimeError(f"No constituent table with >= {expected} rows from {url}")
    names = min(candidates, key=lambda x: abs(len(x) - expected))[:expected]
    if len(names) != expected:
        raise RuntimeError(f"Expected {expected} names from {url}, got {len(names)}")
    return names


def _listing_map() -> dict[str, str]:
    listing = fdr.StockListing("KRX")
    if listing is None or listing.empty:
        raise RuntimeError("FinanceDataReader KRX listing is empty")
    code_col = next((c for c in ["Code", "Symbol"] if c in listing.columns), None)
    name_col = next((c for c in ["Name", "MarketName"] if c in listing.columns), None)
    if not code_col or not name_col:
        raise RuntimeError(f"Unexpected FDR listing columns: {list(listing.columns)}")
    out: dict[str, str] = {}
    for _, row in listing.iterrows():
        code = str(row[code_col]).zfill(6)
        name = _norm_name(row[name_col])
        if re.fullmatch(r"\d{6}", code) and name:
            out.setdefault(name, code)
    return out


def _investing_exact_universe() -> tuple[list[str], dict]:
    mapping = _listing_map()
    symbols: list[str] = []
    meta = {"source": "Investing_index_components_crosswalked_with_FDR_KRX_listing", "sources": {}}
    unmatched_all: list[str] = []
    for key, (url, suffix, expected) in SOURCES.items():
        names = _component_names(url, expected)
        codes: list[str] = []
        unmatched: list[str] = []
        for name in names:
            code = mapping.get(_norm_name(name))
            if code:
                codes.append(code)
            else:
                unmatched.append(name)
        codes = list(dict.fromkeys(codes))
        if unmatched or len(codes) != expected:
            raise RuntimeError(f"{key}: matched={len(codes)}/{expected}; unmatched={unmatched[:20]}")
        symbols.extend([f"{c}.{suffix}" for c in codes])
        meta["sources"][key] = {"url": url, "expected": expected, "matched": len(codes)}
        unmatched_all.extend(unmatched)
    symbols = list(dict.fromkeys(symbols))
    if len(symbols) != 350:
        raise RuntimeError(f"Exact combined universe must be 350, got {len(symbols)}")
    meta["combined_count"] = 350
    meta["unmatched"] = unmatched_all
    return symbols, meta


def exact_universe() -> tuple[list[str], dict]:
    errors: list[str] = []
    try:
        return _pykrx_exact_universe()
    except Exception as exc:
        errors.append(f"pykrx:{type(exc).__name__}:{exc}")
    try:
        symbols, meta = _investing_exact_universe()
        meta["fallback_errors"] = errors
        return symbols, meta
    except Exception as exc:
        errors.append(f"investing_fdr:{type(exc).__name__}:{exc}")
    raise RuntimeError("Exact BB350 universe unavailable; refusing partial output. " + " | ".join(errors))


_base_compute_breadth = cb.compute_breadth


def strict_compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    breadth = _base_compute_breadth(prices)
    if breadth.empty:
        return breadth
    latest_count = int(breadth.iloc[-1]["universe_count"])
    if latest_count != 350:
        raise RuntimeError(f"Latest BB350 eligible universe must be exactly 350, got {latest_count}; refusing partial output")
    return breadth


cb.fetch_universe = exact_universe
cb.compute_breadth = strict_compute_breadth

if __name__ == "__main__":
    cb.main()
