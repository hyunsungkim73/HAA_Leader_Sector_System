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

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
META = ROOT / "data" / "meta"
for p in (DERIVED, META):
    p.mkdir(parents=True, exist_ok=True)
KST = timezone(timedelta(hours=9))
INDEXES = {
    "kospi200": {"code": "1028", "expected": 200, "suffix": "KS", "market": "KOSPI"},
    "kosdaq150": {"code": "2203", "expected": 150, "suffix": "KQ", "market": "KOSDAQ"},
}


def _norm_name(value: object) -> str:
    s = re.sub(r"\s+", "", str(value).strip().upper())
    return re.sub(r"[^0-9A-Z가-힣]", "", s)


def _code(value: object) -> str | None:
    s = str(value).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    if re.fullmatch(r"[0-9A-Z]{6}", s):
        return s
    return s.zfill(6) if s.isdigit() and len(s) < 6 else None


class _HttpsOnly:
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


def _listing() -> dict[str, dict[str, str]]:
    df = fdr.StockListing("KRX")
    code_col = "Code" if "Code" in df.columns else "Symbol" if "Symbol" in df.columns else None
    if df is None or df.empty or code_col is None or "Name" not in df.columns:
        raise RuntimeError("FinanceDataReader KRX listing unavailable or malformed")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        code = _code(row[code_col])
        name = str(row["Name"]).strip()
        if code and name:
            out[code] = {"name": name, "norm": _norm_name(name), "market": str(row.get("Market", "")).upper()}
    if len(out) < 2000:
        raise RuntimeError(f"FinanceDataReader KRX listing unexpectedly small: {len(out)}")
    return out


def _members(key: str, listing: dict[str, dict[str, str]]) -> tuple[list[str], dict, list[dict]]:
    cfg = INDEXES[key]
    df = fdr.SnapDataReader(f"KRX/INDEX/STOCK/{cfg['code']}")
    if df is None or df.empty or not {"Code", "Name"}.issubset(df.columns):
        raise RuntimeError(f"FinanceDataReader index {cfg['code']} unavailable or malformed")
    rows, errors, seen = [], [], set()
    for _, row in df.iterrows():
        code, snap_name = _code(row["Code"]), str(row["Name"]).strip()
        if not code or code in seen:
            errors.append(f"invalid_or_duplicate:{row['Code']}:{snap_name}")
            continue
        seen.add(code)
        listed = listing.get(code)
        if not listed:
            errors.append(f"not_listed:{code}:{snap_name}")
            continue
        if _norm_name(snap_name) != listed["norm"]:
            errors.append(f"name_mismatch:{code}:{snap_name}!={listed['name']}")
            continue
        if cfg["market"] not in listed["market"]:
            errors.append(f"market_mismatch:{code}:{listed['market']}")
            continue
        rows.append({"code": code, "name": listed["name"], "market": cfg["market"], "index_code": cfg["code"], "symbol": f"{code}.{cfg['suffix']}"})
    if errors:
        raise RuntimeError(f"FinanceDataReader validation failed for {key}: {errors[:20]}")
    if len(rows) != cfg["expected"]:
        raise RuntimeError(f"FinanceDataReader {key} count={len(rows)} expected={cfg['expected']}")
    symbols = [r["symbol"] for r in rows]
    return symbols, {"index_code": cfg["code"], "expected": cfg["expected"], "count": len(rows)}, rows


def exact_universe() -> tuple[list[str], dict]:
    # This process intentionally never imports pykrx. pykrx auto-login creates a
    # separate KRX session and is outside the authorized Stage-2 source policy.
    fdr_snap.requests = _HttpsOnly
    listing = _listing()
    a, ma, ra = _members("kospi200", listing)
    b, mb, rb = _members("kosdaq150", listing)
    symbols = a + b
    if len(symbols) != 350 or len(set(symbols)) != 350:
        raise RuntimeError(f"Exact BB350 must contain 350 unique constituents; got={len(symbols)}")
    meta = {
        "source": "FinanceDataReader_KRX_index_constituents",
        "source_type": "exact_index_constituents",
        "source_policy": "FinanceDataReader-only",
        "definition": "KOSPI200 (1028) + KOSDAQ150 (2203)",
        "validation_policy": "FDR index code/name/market cross-check against FDR StockListing('KRX')",
        "asof_kst": datetime.now(KST).date().isoformat(),
        "kospi200": ma, "kosdaq150": mb, "combined_count": 350,
        "constituents": ra + rb, "symbols": symbols,
    }
    (META / "breadth_universe.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return symbols, meta


def _price(symbol: str, start: str) -> pd.DataFrame:
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
    return out[["date", "ticker", "close"]].dropna()


def collect_prices(symbols: list[str], start: str) -> tuple[pd.DataFrame, dict]:
    frames, ok = [], []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_price, s, start): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            frame = future.result()
            if not frame.empty:
                frames.append(frame); ok.append(symbol)
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "close"])
    present = set(prices["ticker"].unique()) if not prices.empty else set()
    unresolved = [s for s in symbols if s not in present]
    return prices, {"source_policy": "FinanceDataReader-only", "requested_count": len(symbols), "recovered_count": len(ok), "unresolved_count": len(unresolved), "unresolved_symbols": unresolved}


def compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date").copy()
        ma = g["close"].rolling(20, min_periods=20).mean()
        sd = g["close"].rolling(20, min_periods=20).std(ddof=0)
        g["eligible"] = ma.notna()
        g["breakout"] = g["eligible"] & (g["close"] > ma + 2 * sd)
        parts.append(g[["date", "ticker", "eligible", "breakout"]])
    if not parts:
        return pd.DataFrame()
    eligible = pd.concat(parts, ignore_index=True)
    eligible = eligible[eligible["eligible"]]
    agg = eligible.groupby("date").agg(universe_count=("ticker", "nunique"), breakout_count=("breakout", "sum")).reset_index()
    agg["breadth_pct"] = agg["breakout_count"] / agg["universe_count"]
    agg["breadth_5dma"] = agg["breadth_pct"].rolling(5, min_periods=1).mean()
    agg["slope"] = agg["breadth_5dma"].diff(); agg["acceleration"] = agg["slope"].diff()
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")
    return agg


def _upsert(path: Path, new: pd.DataFrame) -> None:
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    out.drop_duplicates(["date"], keep="last").sort_values("date").to_csv(path, index=False)


def main() -> None:
    symbols, meta = exact_universe()
    start = (datetime.now(KST).date() - timedelta(days=430)).isoformat()
    prices, price_meta = collect_prices(symbols, start)
    breadth = compute_breadth(prices)
    if breadth.empty:
        raise RuntimeError("Breadth calculation produced no rows")
    latest_count = int(breadth.iloc[-1]["universe_count"])
    if latest_count != 350:
        raise RuntimeError(f"Latest exact BB350 count={latest_count}; refusing partial overwrite")
    _upsert(DERIVED / "breadth_bb20_2sigma_daily.csv", breadth)
    status = {"generated_at_kst": datetime.now(KST).isoformat(), "start_requested": start, "symbol_count": len(symbols), "price_rows": len(prices), "breadth_rows": len(breadth), "latest": breadth.iloc[-1].to_dict(), "price_retry": price_meta, **meta}
    (META / "breadth_last_run.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
