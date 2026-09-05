from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re

import pandas as pd
from pykrx import stock

import collect_breadth as cb

KST = timezone(timedelta(hours=9))
META = Path(__file__).resolve().parents[1] / "data" / "meta"
INDEXES = {
    "k200": ("1028", "KS", 200),
    "k150": ("2203", "KQ", 150),
}


def _clean_codes(values: list[object]) -> list[str]:
    codes: list[str] = []
    for value in values:
        s = str(value).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s.isdigit() and 1 <= len(s) <= 6:
            codes.append(s.zfill(6))
    return list(dict.fromkeys(codes))


def pykrx_exact_universe() -> tuple[list[str], dict]:
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
                detail[key] = {
                    "index_code": index_code,
                    "expected": expected,
                    "count": len(codes),
                }
            symbols = list(dict.fromkeys(pieces))
            if len(symbols) != 350:
                raise RuntimeError(f"combined expected=350, got={len(symbols)}")
            meta = {
                "source": "pykrx_KRX_index_constituents",
                "source_type": "official_index_membership",
                "asof_requested": d,
                "combined_count": 350,
                "sources": detail,
                "prior_attempt_errors": errors,
            }
            (META / "breadth_universe.json").write_text(
                json.dumps(meta | {"symbols": symbols}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return symbols, meta
        except Exception as exc:
            errors.append(f"{d}:{type(exc).__name__}:{exc}")
    raise RuntimeError(
        "Exact pykrx BB350 universe unavailable; refusing proxy/partial output. "
        + " | ".join(errors[-5:])
    )


_base_compute_breadth = cb.compute_breadth


def strict_compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    breadth = _base_compute_breadth(prices)
    if breadth.empty:
        return breadth
    latest_count = int(breadth.iloc[-1]["universe_count"])
    if latest_count != 350:
        raise RuntimeError(
            f"Latest BB350 eligible universe must be exactly 350, got {latest_count}; refusing partial output"
        )
    return breadth


cb.fetch_universe = pykrx_exact_universe
cb.compute_breadth = strict_compute_breadth

if __name__ == "__main__":
    cb.main()
