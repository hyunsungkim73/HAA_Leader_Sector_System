from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"
META = DATA / "meta"
for p in (RAW, DERIVED, META):
    p.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))
TODAY = datetime.now(KST).date()

# Core sector ETF / benchmark proxies. Keep codes stable; change only with explicit review.
KR_TICKERS = {
    "KOSPI200": "069500.KS",            # KODEX 200 proxy for benchmark continuity
    "semiconductor": "091160.KS",       # KODEX 반도체
    "auto": "091180.KS",                # KODEX 자동차
    "battery": "305720.KS",             # KODEX 2차전지산업
    "bio_health": "266420.KS",          # KODEX 헬스케어
    "financial": "091170.KS",           # KODEX 은행
    "shipbuilding": "466920.KS",        # SOL 조선TOP3플러스 (fallback handled below)
    "steel_materials": "117680.KS",     # KODEX 철강
    "consumer": "266410.KS",            # KODEX 필수소비재
    "media_platform": "266360.KS",      # KODEX 미디어&엔터테인먼트
    "construction": "117700.KS",        # KODEX 건설
    "biosimilar_cdmo": "0001P0.KS",     # 마이티 바이오시밀러&CDMO액티브
    "k_beauty": "479850.KS",            # HANARO K-뷰티
    "k_food": "438900.KS",              # HANARO Fn K-푸드
}

FALLBACK_TICKERS = {
    "shipbuilding": ["139230.KS", "466920.KS"],
    "media_platform": ["266360.KS", "228810.KS"],
    "bio_health": ["266420.KS", "244580.KS"],
}

US_TICKERS = {"TIP": "TIP"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def upsert_csv(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    if new is None or new.empty:
        return
    old = _read_csv(path)
    all_df = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    all_df = all_df.drop_duplicates(subset=keys, keep="last")
    if "date" in all_df.columns:
        all_df["date"] = pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")
        all_df = all_df.sort_values(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(path, index=False)


def yf_history(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date"})
    cols = [c for c in ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def collect_price(name: str, ticker: str, min_start: str = "2024-01-01") -> dict:
    path = RAW / "prices" / f"{name}.csv"
    old = _read_csv(path)
    if old.empty:
        start = min_start
    else:
        last = pd.to_datetime(old["date"]).max().date()
        start = (last - timedelta(days=10)).isoformat()
    candidates = [ticker] + FALLBACK_TICKERS.get(name, [])
    last_err = None
    used = None
    df = pd.DataFrame()
    for cand in dict.fromkeys(candidates):
        try:
            df = yf_history(cand, start)
            if not df.empty:
                used = cand
                break
        except Exception as exc:
            last_err = str(exc)
    if not df.empty:
        df.insert(1, "ticker", used)
        upsert_csv(path, df, ["date"])
    return {"series": name, "ticker_requested": ticker, "ticker_used": used, "rows_new": int(len(df)), "error": last_err}


def calc_momentum(price_df: pd.DataFrame, name: str) -> pd.DataFrame:
    if price_df.empty or "close" not in price_df.columns:
        return pd.DataFrame()
    x = price_df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    for n in (21, 63, 126):
        x[f"m{n}"] = x["close"] / x["close"].shift(n) - 1
    # Weekly snapshots: last observation of each ISO week.
    x["week"] = x["date"].dt.to_period("W-FRI")
    w = x.groupby("week", as_index=False).tail(1).copy()
    w["series"] = name
    for n in (21, 63, 126):
        w[f"slope_m{n}"] = w[f"m{n}"].diff()
        w[f"accel_m{n}"] = w[f"slope_m{n}"].diff()
    cols = ["date", "series", "close", "m21", "m63", "m126", "slope_m21", "slope_m63", "slope_m126", "accel_m21", "accel_m63", "accel_m126"]
    return w[cols].tail(80)


def build_relative_strength() -> None:
    bench_path = RAW / "prices" / "KOSPI200.csv"
    bench = _read_csv(bench_path)
    if bench.empty:
        return
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.sort_values("date")
    for n in (21, 63, 126):
        bench[f"b{n}"] = bench["close"] / bench["close"].shift(n) - 1
    bench = bench[["date", "b21", "b63", "b126"]]
    rows = []
    for name in KR_TICKERS:
        if name == "KOSPI200":
            continue
        p = _read_csv(RAW / "prices" / f"{name}.csv")
        if p.empty:
            continue
        p["date"] = pd.to_datetime(p["date"])
        p = p.sort_values("date")
        for n in (21, 63, 126):
            p[f"m{n}"] = p["close"] / p["close"].shift(n) - 1
        z = p.merge(bench, on="date", how="inner")
        for n in (21, 63, 126):
            z[f"rs{n}"] = z[f"m{n}"] - z[f"b{n}"]
        z["week"] = z["date"].dt.to_period("W-FRI")
        z = z.groupby("week", as_index=False).tail(1)
        z["series"] = name
        for n in (21, 63, 126):
            z[f"slope_rs{n}"] = z[f"rs{n}"].diff()
            z[f"accel_rs{n}"] = z[f"slope_rs{n}"].diff()
        rows.append(z[["date", "series", "rs21", "rs63", "rs126", "slope_rs21", "slope_rs63", "slope_rs126", "accel_rs21", "accel_rs63", "accel_rs126"]].tail(80))
    if rows:
        out = pd.concat(rows, ignore_index=True)
        upsert_csv(DERIVED / "relative_strength_weekly.csv", out, ["date", "series"])


def collect_tip() -> dict:
    return collect_price("TIP", "TIP", min_start="2024-01-01")


def compute_tip_haa() -> None:
    p = _read_csv(RAW / "prices" / "TIP.csv")
    if p.empty:
        return
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").drop_duplicates("date")
    # yfinance Adj Close is used as total-return cross-check series. Official iShares distributions
    # are maintained separately in data/raw/tip_distributions.csv when available.
    pxcol = "adj_close" if "adj_close" in p.columns and p["adj_close"].notna().sum() > 252 else "close"
    p["tr_proxy"] = p[pxcol]
    for n, label in [(21, "r1"), (63, "r3"), (126, "r6"), (252, "r12")]:
        p[label] = p["tr_proxy"] / p["tr_proxy"].shift(n) - 1
    p["haa_13612u"] = (12*p["r1"] + 4*p["r3"] + 2*p["r6"] + p["r12"]) / 4
    p["regime"] = np.where(
        p["haa_13612u"] <= 0,
        "Risk-Off",
        np.where((p["r1"] < 0) & (p["r3"] < 0), "Weak Risk-On", "Risk-On")
    )
    p["week"] = p["date"].dt.to_period("W-FRI")
    w = p.groupby("week", as_index=False).tail(1).copy()
    w["slope_haa"] = w["haa_13612u"].diff()
    w["accel_haa"] = w["slope_haa"].diff()
    out = w[["date", "r1", "r3", "r6", "r12", "haa_13612u", "regime", "slope_haa", "accel_haa"]].tail(104)
    upsert_csv(DERIVED / "tip_haa_weekly.csv", out, ["date"])


def write_coverage(status_rows: list[dict]) -> None:
    rows = []
    for name in list(KR_TICKERS) + ["TIP"]:
        path = RAW / "prices" / f"{name}.csv"
        df = _read_csv(path)
        rows.append({
            "series": name,
            "obs": int(len(df)),
            "start": None if df.empty else str(df["date"].min()),
            "end": None if df.empty else str(df["date"].max()),
            "ready_190": bool(len(df) >= 190),
            "ready_220": bool(len(df) >= 220),
            "ready_252": bool(len(df) >= 252),
        })
    pd.DataFrame(rows).to_csv(META / "price_coverage.csv", index=False)
    payload = {
        "generated_at_kst": datetime.now(KST).isoformat(),
        "collector_status": status_rows,
    }
    (META / "last_run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    status = []
    for name, ticker in KR_TICKERS.items():
        status.append(collect_price(name, ticker))
        p = _read_csv(RAW / "prices" / f"{name}.csv")
        if not p.empty:
            upsert_csv(DERIVED / "momentum_weekly.csv", calc_momentum(p, name), ["date", "series"])
    status.append(collect_tip())
    build_relative_strength()
    compute_tip_haa()
    write_coverage(status)


if __name__ == "__main__":
    main()
