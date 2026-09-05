from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re

import pandas as pd
import FinanceDataReader as fdr

import collect_breadth as cb

KST = timezone(timedelta(hours=9))
META = Path(__file__).resolve().parents[1] / "data" / "meta"

INDEXES = {
    "kospi200": {"code": "1028", "expected": 200, "suffix": "KS", "market": "KOSPI"},
    "kosdaq150": {"code": "2203", "expected": 150, "suffix": "KQ", "market": "KOSDAQ"},
}


def _norm_name(value: object) -> str:
    s = str(value).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9A-Z가-힣]", "", s)
    return s


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


def _index_constituents(index_key: str, listing_map: dict[str, dict[str, str]]) -> tuple[list[str], dict, list[dict]]:
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


def fdr_only_universe() -> tuple[list[str], dict]:
    listing_map = _fdr_krx_listing_map()
    k200, m200, rows200 = _index_constituents("kospi200", listing_map)
    k150, m150, rows150 = _index_constituents("kosdaq150", listing_map)
    symbols = k200 + k150
    if len(symbols) != 350 or len(set(symbols)) != 350:
        raise RuntimeError(f"FDR index universe must be 350 unique symbols; got {len(symbols)} / {len(set(symbols))}")

    constituents = rows200 + rows150
    meta = {
        "source": "FinanceDataReader_only",
        "source_type": "user_approved_single_source",
        "validation_policy": "FDR-only; no external double-check; internal FDR code-name-market validation required",
        "definition": "KOSPI200 (KRX index 1028) + KOSDAQ150 (KRX index 2203) constituents from FinanceDataReader SnapDataReader",
        "kospi200": m200,
        "kosdaq150": m150,
        "combined_count": 350,
        "constituents": constituents,
        "symbols": symbols,
    }
    (META / "breadth_universe.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
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


def fdr_only_prices(symbols: list[str], start: str) -> tuple[pd.DataFrame, dict]:
    frames = []
    recovered = []
    unresolved = []
    for s in symbols:
        x = _fdr_one(s, start)
        if x.empty:
            unresolved.append(s)
        else:
            frames.append(x)
            recovered.append(s)
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "close"])
    return prices, {
        "provider": "FinanceDataReader_only",
        "requested_count": len(symbols),
        "recovered_count": len(recovered),
        "unresolved_count": len(unresolved),
        "unresolved_symbols": unresolved,
        "double_check": False,
        "universe_validation": "FDR SnapDataReader constituents internally validated against FDR KRX listing",
    }


cb.fetch_universe = fdr_only_universe
cb.collect_prices = fdr_only_prices

if __name__ == "__main__":
    cb.main()
