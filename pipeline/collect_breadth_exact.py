from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re

import pandas as pd
import requests as http_requests
import FinanceDataReader as fdr
import FinanceDataReader.krx.snap as fdr_snap
from pykrx import stock

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


def _normalize_code(value: object) -> str | None:
    s = str(value).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    if re.fullmatch(r"[0-9A-Z]{6}", s):
        return s
    if s.isdigit() and 1 <= len(s) < 6:
        return s.zfill(6)
    return None


def _clean_codes(values: list[object]) -> list[str]:
    codes = [code for value in values if (code := _normalize_code(value))]
    return list(dict.fromkeys(codes))


def _pykrx_exact_universe() -> tuple[list[str], dict]:
    errors: list[str] = []
    today = datetime.now(KST).date()
    for days_back in range(1, 15):
        requested = (today - timedelta(days=days_back)).strftime("%Y%m%d")
        pieces: list[str] = []
        detail: dict[str, dict] = {}
        try:
            for key, cfg in INDEXES.items():
                raw = list(stock.get_index_portfolio_deposit_file(cfg["code"], date=requested, alternative=True))
                codes = _clean_codes(raw)
                expected = int(cfg["expected"])
                if len(codes) != expected:
                    invalid = [str(v).strip() for v in raw if _normalize_code(v) is None]
                    raise RuntimeError(
                        f"{key} expected={expected}, got={len(codes)}, raw={len(raw)}, invalid={invalid[:10]}"
                    )
                pieces.extend([f"{code}.{cfg['suffix']}" for code in codes])
                detail[key] = {
                    "index_code": cfg["code"],
                    "expected": expected,
                    "count": len(codes),
                    "codes": codes,
                }
            symbols = list(dict.fromkeys(pieces))
            if len(symbols) != 350:
                raise RuntimeError(f"combined expected=350, got={len(symbols)}")
            return symbols, {
                "source": "pykrx_KRX_index_constituents",
                "source_type": "exact_index_constituents",
                "validation_policy": "KRX index constituent endpoint via pykrx; exact 200+150 only",
                "asof_requested": requested,
                "combined_count": 350,
                "sources": detail,
                "prior_attempt_errors": errors,
                "symbols": symbols,
            }
        except Exception as exc:
            errors.append(f"{requested}:{type(exc).__name__}:{exc}")
    raise RuntimeError("pykrx exact universe failed across recent dates: " + " | ".join(errors[-8:]))


class _HttpsRequestsProxy:
    """Transport-only hardening for FinanceDataReader's KRX snap reader."""

    @staticmethod
    def _url(url: str) -> str:
        return url.replace("http://data.krx.co.kr", "https://data.krx.co.kr")

    @classmethod
    def get(cls, url: str, *args, **kwargs):
        kwargs.setdefault("timeout", 30)
        return http_requests.get(cls._url(url), *args, **kwargs)

    @classmethod
    def post(cls, url: str, *args, **kwargs):
        kwargs.setdefault("timeout", 30)
        return http_requests.post(cls._url(url), *args, **kwargs)


def _install_fdr_krx_https_transport() -> None:
    fdr_snap.requests = _HttpsRequestsProxy


def _fdr_krx_listing_map() -> dict[str, dict[str, str]]:
    df = fdr.StockListing("KRX")
    if df is None or df.empty:
        raise RuntimeError("FinanceDataReader StockListing('KRX') returned no rows")
    code_col = "Code" if "Code" in df.columns else "Symbol" if "Symbol" in df.columns else None
    if not code_col or "Name" not in df.columns:
        raise RuntimeError(f"Unexpected FDR KRX listing columns: {list(df.columns)}")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        code = _normalize_code(row[code_col])
        name = str(row["Name"]).strip()
        if not code:
            continue
        market = str(row.get("Market", "")).strip().upper()
        out[code] = {"name": name, "name_norm": _norm_name(name), "market": market}
    if len(out) < 2000:
        raise RuntimeError(f"FDR KRX listing validation map unexpectedly small: {len(out)}")
    return out


def _fdr_index_constituents(
    index_key: str, listing_map: dict[str, dict[str, str]]
) -> tuple[list[str], dict, list[dict]]:
    cfg = INDEXES[index_key]
    index_code = cfg["code"]
    expected = int(cfg["expected"])
    suffix = cfg["suffix"]
    expected_market = cfg["market"]

    last_exc: Exception | None = None
    df = pd.DataFrame()
    for _ in range(3):
        try:
            df = fdr.SnapDataReader(f"KRX/INDEX/STOCK/{index_code}")
            if df is not None and not df.empty:
                break
        except Exception as exc:
            last_exc = exc
    if df is None or df.empty:
        raise RuntimeError(
            f"FDR SnapDataReader KRX/INDEX/STOCK/{index_code} returned no rows"
            + (f"; last_error={type(last_exc).__name__}:{last_exc}" if last_exc else "")
        )
    if "Code" not in df.columns or "Name" not in df.columns:
        raise RuntimeError(f"Unexpected FDR index constituent columns for {index_code}: {list(df.columns)}")

    rows: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = _normalize_code(row["Code"])
        snap_name = str(row["Name"]).strip()
        if not code:
            errors.append(f"invalid_code:{row['Code']}:{snap_name}")
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
        "transport_note": "FinanceDataReader KRX snap transport forced from http to https on data.krx.co.kr",
        "asof_kst": datetime.now(KST).date().isoformat(),
    }
    return symbols, meta, rows


def _fdr_exact_universe() -> tuple[list[str], dict]:
    _install_fdr_krx_https_transport()
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
    (META / "breadth_universe.json").write_text(
        json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
    frames: list[pd.DataFrame] = []
    recovered: list[str] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fdr_one, symbol, start): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
            except Exception:
                frame = pd.DataFrame()
            if frame is None or frame.empty:
                failures.append(symbol)
            else:
                frames.append(frame)
                recovered.append(symbol)
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "close"])
    if not prices.empty:
        prices = prices.drop_duplicates(["date", "ticker"], keep="last")
    present = set(prices["ticker"].astype(str).unique()) if not prices.empty else set()
    unresolved = [symbol for symbol in symbols if symbol not in present]
    return prices, {
        "source_policy": "FinanceDataReader prices for exact KRX constituent universe",
        "requested_count": len(symbols),
        "fdr_recovered_count": len(recovered),
        "fdr_recovered_symbols": sorted(recovered),
        "unresolved_count": len(unresolved),
        "unresolved_symbols": unresolved,
        "request_failures": sorted(set(failures)),
    }


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
    cb.main()
