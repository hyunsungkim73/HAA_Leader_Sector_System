from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import re

import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock

import collect_breadth as cb

KST = timezone(timedelta(hours=9))
META = Path(__file__).resolve().parents[1] / "data" / "meta"

INDEXES = {
    "kospi200": {"code": "1028", "expected": 200, "suffix": "KS", "market": "KOSPI"},
    "kosdaq150": {"code": "2203", "expected": 150, "suffix": "KQ", "market": "KOSDAQ"},
}


def _report_krx_env_presence() -> None:
    print(f"KRX_ID_PRESENT={'yes' if bool(os.getenv('KRX_ID')) else 'no'}")
    print(f"KRX_PW_PRESENT={'yes' if bool(os.getenv('KRX_PW')) else 'no'}")


def _norm_name(value: object) -> str:
    s = str(value).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9A-Z가-힣]", "", s)
    return s


def _normalize_code(value: object) -> str | None:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit() and 1 <= len(s) <= 6:
        return s.zfill(6)
    return None


def _clean_codes(values: list[object]) -> list[str]:
    codes = [code for value in values if (code := _normalize_code(value))]
    return list(dict.fromkeys(codes))


def _describe_raw_codes(values: list[object]) -> str:
    normalized = [_normalize_code(v) for v in values]
    valid = [v for v in normalized if v]
    duplicates = sorted(code for code, count in Counter(valid).items() if count > 1)
    invalid = [str(v).strip() for v, code in zip(values, normalized) if not code]
    return (
        f"raw={len(values)} valid={len(valid)} unique={len(set(valid))} "
        f"duplicates={duplicates[:10]} invalid={invalid[:10]}"
    )


def _pykrx_exact_universe() -> tuple[list[str], dict]:
    errors: list[str] = []
    today = datetime.now(KST).date()
    # Query explicit recent dates so pykrx does not depend on an implicit latest-date
    # resolution that can fail around weekends/holidays. Only an exact 200+150 set passes.
    for days_back in range(1, 15):
        d = (today - timedelta(days=days_back)).strftime("%Y%m%d")
        pieces: list[str] = []
        detail: dict[str, dict] = {}
        try:
            for key, cfg in INDEXES.items():
                raw = list(stock.get_index_portfolio_deposit_file(cfg["code"], date=d, alternative=True))
                codes = _clean_codes(raw)
                expected = int(cfg["expected"])
                if len(codes) != expected:
                    raise RuntimeError(
                        f"{key} expected={expected}, got={len(codes)}; {_describe_raw_codes(raw)}"
                    )
                pieces.extend([f"{code}.{cfg['suffix']}" for code in codes])
                detail[key] = {
                    "index_code": cfg["code"],
                    "expected": expected,
                    "count": len(codes),
                }
            symbols = list(dict.fromkeys(pieces))
            if len(symbols) != 350:
                raise RuntimeError(f"combined expected=350, got={len(symbols)}")
            return symbols, {
                "source": "pykrx_KRX_index_constituents",
                "source_type": "exact_index_constituents",
                "asof_requested": d,
                "combined_count": 350,
                "sources": detail,
                "prior_attempt_errors": errors,
            }
        except Exception as exc:
            errors.append(f"{d}:{type(exc).__name__}:{exc}")
    raise RuntimeError("pykrx exact universe failed across recent dates: " + " | ".join(errors[-5:]))


def _fdr_krx_listing_map() -> dict[str, dict[str, str]]:
    df = fdr.StockListing("KRX")
    if df is None or df.empty:
        raise RuntimeError("FinanceDataReader StockListing('KRX') returned no rows")
    code_col = "Code" if "Code" in df.columns else "Symbol" if "Symbol" in df.columns else None
    if not code_col or "Name" not in df.columns:
        raise RuntimeError(f"Unexpected FDR KRX listing columns: {list(df.columns)}")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        code = str(row[code_col]).strip()
        name = str(row["Name"]).strip()
        if not re.fullmatch(r"\d{6}", code):
            continue
        market = str(row.get("Market", "")).strip().upper()
        out[code] = {"name": name, "name_norm": _norm_name(name), "market": market}
    if len(out) < 2000:
        raise RuntimeError(f"FDR KRX listing validation map unexpectedly small: {len(out)}")
    return out


def _fdr_index_constituents(index_key: str, listing_map: dict[str, dict[str, str]]) -> tuple[list[str], dict, list[dict]]:
    cfg = INDEXES[index_key]
    index_code = cfg["code"]
    expected = int(cfg["expected"])
    suffix = cfg["suffix"]
    expected_market = cfg["market"]

    df = fdr.SnapDataReader(f"KRX/INDEX/STOCK/{index_code}")
    if df is None or df.empty:
        raise RuntimeError(f"FDR SnapDataReader KRX/INDEX/STOCK/{index_code} returned no rows")
    if "Code" not in df.columns or "Name" not in df.columns:
        raise RuntimeError(f"Unexpected FDR index constituent columns for {index_code}: {list(df.columns)}")

    rows: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = str(row["Code"]).strip()
        snap_name = str(row["Name"]).strip()
        if not re.fullmatch(r"\d{6}", code):
            errors.append(f"invalid_code:{code}:{snap_name}")
            continue
        if code in seen:
            errors.append(f"duplicate_code:{code}:{snap_name}")
            continue
        seen.add(code)
        listed = listing_map.get(code)
        if not listed:
            errors.append(f"not_in_FDR_KRX_listing:{code}:{snap_name}")
            continue
        if _norm_name(snap_name) != listed["name_norm"]:
            errors.append(f"name_mismatch:{code}:{snap_name}!={listed['name']}")
            continue
        listed_market = listed["market"]
        if expected_market not in listed_market:
            errors.append(f"market_mismatch:{code}:{snap_name}:{listed_market}")
            continue
        rows.append({
            "code": code,
            "name": listed["name"],
            "market": expected_market,
            "index_code": index_code,
            "symbol": f"{code}.{suffix}",
        })

    if errors:
        raise RuntimeError(f"FDR internal validation failed for {index_key}: {errors[:20]}")
    if len(rows) != expected:
        raise RuntimeError(f"FDR {index_key} constituent count mismatch: got={len(rows)} expected={expected}")

    symbols = [r["symbol"] for r in rows]
    meta = {
        "index_code": index_code,
        "expected": expected,
        "count": len(rows),
        "source": f"FinanceDataReader SnapDataReader KRX/INDEX/STOCK/{index_code}",
        "validation": "code and name matched against FinanceDataReader StockListing('KRX')",
        "asof_kst": datetime.now(KST).date().isoformat(),
    }
    return symbols, meta, rows


def _fdr_exact_universe() -> tuple[list[str], dict]:
    listing_map = _fdr_krx_listing_map()
    k200, m200, rows200 = _fdr_index_constituents("kospi200", listing_map)
    k150, m150, rows150 = _fdr_index_constituents("kosdaq150", listing_map)
    symbols = k200 + k150
    if len(symbols) != 350 or len(set(symbols)) != 350:
        raise RuntimeError(f"FDR exact index universe must be 350 unique symbols; got {len(symbols)} / {len(set(symbols))}")
    constituents = rows200 + rows150
    return symbols, {
        "source": "FinanceDataReader_KRX_index_constituents",
        "source_type": "exact_index_constituents",
        "validation_policy": "FDR index constituents internally validated against FDR KRX listing",
        "definition": "KOSPI200 (KRX index 1028) + KOSDAQ150 (KRX index 2203) exact constituents",
        "kospi200": m200,
        "kosdaq150": m150,
        "combined_count": 350,
        "constituents": constituents,
        "symbols": symbols,
    }


def exact_universe() -> tuple[list[str], dict]:
    errors: list[str] = []
    try:
        symbols, meta = _pykrx_exact_universe()
    except Exception as exc:
        errors.append(f"pykrx:{type(exc).__name__}:{exc}")
        try:
            symbols, meta = _fdr_exact_universe()
        except Exception as exc2:
            errors.append(f"fdr_exact:{type(exc2).__name__}:{exc2}")
            raise RuntimeError("Exact BB350 universe unavailable; refusing proxy/partial output. " + " | ".join(errors))
    meta["fallback_errors"] = errors
    (META / "breadth_universe.json").write_text(json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2), encoding="utf-8")
    return symbols, meta


def _fdr_one(symbol: str, start: str) -> pd.DataFrame:
    code = symbol.split(".")[0]
    try:
        df = fdr.DataReader(code, start)
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    out = df[["Close"]].reset_index()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["ticker"] = symbol
    return out[["date", "ticker", "close"]].dropna(subset=["date", "close"])


def exact_prices(symbols: list[str], start: str) -> tuple[pd.DataFrame, dict]:
    # Keep the shared Yahoo bulk collector first, then use FinanceDataReader as the
    # required per-symbol price fallback. Exchange-suffix retry remains a last resort.
    return cb.collect_prices(symbols, start)


_base_compute_breadth = cb.compute_breadth


def strict_compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    breadth = _base_compute_breadth(prices)
    if breadth.empty:
        return breadth
    latest_count = int(breadth.iloc[-1]["universe_count"])
    if latest_count != 350:
        raise RuntimeError(
            f"Latest BB350 eligible universe must be exactly 350, got {latest_count}; refusing partial overwrite"
        )
    return breadth


cb.fetch_universe = exact_universe
cb.collect_prices = exact_prices
cb.compute_breadth = strict_compute_breadth

if __name__ == "__main__":
    _report_krx_env_presence()
    cb.main()
