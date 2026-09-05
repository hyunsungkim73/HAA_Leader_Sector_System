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


def _code_col(df: pd.DataFrame) -> str:
    for c in ["Code", "Symbol"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"FDR listing code column not found: {list(df.columns)}")


def _marcap_col(df: pd.DataFrame) -> str:
    for c in ["Marcap", "MarketCap", "Market Cap"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"FDR listing market-cap column not found: {list(df.columns)}")


def _top_market_codes(market: str, n: int, suffix: str) -> tuple[list[str], dict]:
    df = fdr.StockListing(market)
    if df is None or df.empty:
        raise RuntimeError(f"FinanceDataReader StockListing({market!r}) returned no rows")
    cc = _code_col(df)
    mc = _marcap_col(df)
    z = df[[cc, mc]].copy()
    z[cc] = z[cc].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    z[mc] = pd.to_numeric(z[mc], errors="coerce")
    z = z[z[cc].str.fullmatch(r"\d{6}", na=False) & z[mc].notna()]
    z = z.sort_values([mc, cc], ascending=[False, True]).drop_duplicates(cc, keep="first")
    if len(z) < n:
        raise RuntimeError(f"FDR {market} listing has only {len(z)} usable rows; need {n}")
    codes = z.head(n)[cc].tolist()
    return [f"{c}.{suffix}" for c in codes], {
        "market": market,
        "selection": f"top_{n}_by_FDR_market_cap",
        "count": len(codes),
        "asof_kst": datetime.now(KST).date().isoformat(),
    }


def fdr_only_universe() -> tuple[list[str], dict]:
    k200, m200 = _top_market_codes("KOSPI", 200, "KS")
    k150, m150 = _top_market_codes("KOSDAQ", 150, "KQ")
    symbols = k200 + k150
    if len(symbols) != 350 or len(set(symbols)) != 350:
        raise RuntimeError(f"FDR-only universe must be 350 unique symbols; got {len(symbols)} / {len(set(symbols))}")
    meta = {
        "source": "FinanceDataReader_only",
        "source_type": "user_approved_single_source_proxy",
        "validation_policy": "user_approved_no_double_check",
        "definition": "KOSPI top 200 + KOSDAQ top 150 by FDR market capitalization; operational BB350 proxy, not official index-membership exact universe",
        "kospi200_proxy": m200,
        "kosdaq150_proxy": m150,
        "combined_count": 350,
    }
    (META / "breadth_universe.json").write_text(
        json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2), encoding="utf-8"
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
    }


cb.fetch_universe = fdr_only_universe
cb.collect_prices = fdr_only_prices

if __name__ == "__main__":
    cb.main()
