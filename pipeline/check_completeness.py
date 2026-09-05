from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
META = DATA / "meta"
DERIVED = DATA / "derived"
RAW = DATA / "raw"
KST = timezone(timedelta(hours=9))

REGISTRY_JSON = META / "series_registry_min.json"
OUT_CSV = META / "completeness_series.csv"
OUT_JSON = META / "completeness_report.json"

PRICE_MAP = {
    "price_semi_it": "semiconductor",
    "price_auto": "auto",
    "price_battery": "battery",
    "price_bio": "bio_health",
    "price_financial": "financial",
    "price_ship_machinery": "shipbuilding",
    "price_steel": "steel_materials",
    "price_consumer": "consumer",
    "price_media_platform": "media_platform",
    "price_construction": "construction",
    "price_kbeauty": "k_beauty",
    "price_kfood": "k_food",
    "price_biosim_cdmo": "biosimilar_cdmo",
    "price_kospi200": "KOSPI200",
}


def safe_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_registry() -> pd.DataFrame:
    rows = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    reg = pd.DataFrame(rows)
    required = {"series_id", "dataset", "sector", "indicator", "status"}
    missing = required - set(reg.columns)
    if missing:
        raise RuntimeError(f"registry missing fields: {sorted(missing)}")
    if len(reg) != 80 or reg["series_id"].nunique() != 80:
        raise RuntimeError(f"registry must contain exactly 80 unique series, got rows={len(reg)}, unique={reg['series_id'].nunique()}")
    return reg


def fundamental_counts() -> pd.DataFrame:
    folder = RAW / "fundamentals"
    frames = []
    for path in sorted(folder.glob("*.csv")):
        f = safe_csv(path)
        if not f.empty and {"obs_date", "sector", "indicator"}.issubset(f.columns):
            f["_source_file"] = path.name
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    f = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["obs_date","sector","subsector","indicator","source"]:
        if c not in f.columns:
            f[c] = ""
    f["obs_date"] = pd.to_datetime(f["obs_date"], errors="coerce")
    keys = ["obs_date","sector","subsector","indicator","source"]
    f = f.drop_duplicates(keys, keep="last")
    return f


def main() -> None:
    reg = load_registry()
    price_cov = safe_csv(META / "price_coverage.csv")
    price_cov = price_cov.set_index("series") if not price_cov.empty else pd.DataFrame()
    fund = fundamental_counts()
    macro = safe_csv(RAW / "macro" / "macro_market_archive.csv")
    eps = safe_csv(RAW / "eps" / "eps_revisions_archive.csv")
    breadth = safe_csv(DERIVED / "breadth_bb20_2sigma_daily.csv")
    haa = safe_csv(DERIVED / "tip_haa_weekly.csv")
    momentum = safe_csv(DERIVED / "momentum_weekly.csv")
    rs = safe_csv(DERIVED / "relative_strength_weekly.csv")

    rows = []
    for _, r in reg.iterrows():
        sid = r["series_id"]
        ds = r["dataset"]
        obs = 0
        latest = None
        ready = False
        quality = "missing"
        reason = "no observations"

        if sid in PRICE_MAP:
            name = PRICE_MAP[sid]
            if not price_cov.empty and name in price_cov.index:
                pc = price_cov.loc[name]
                obs = int(pc["obs"])
                latest = str(pc["end"])
                ready = bool(pc["ready_220"])
                quality = "direct/market" if ready else "partial"
                reason = f"{obs} daily observations; 220 required"
        elif sid == "tip_tr":
            if not price_cov.empty and "TIP" in price_cov.index:
                pc = price_cov.loc["TIP"]
                obs = int(pc["obs"]); latest = str(pc["end"])
                ready = bool(pc["ready_252"]) and not haa.empty
                quality = "market total-return proxy" if ready else "partial"
                reason = f"TIP observations={obs}; HAA rows={len(haa)}"
        elif sid == "bb350":
            if not breadth.empty:
                obs = len(breadth); latest = str(breadth["date"].max())
                u = int(pd.to_numeric(breadth["universe_count"], errors="coerce").dropna().iloc[-1])
                ready = u >= 350
                quality = "exact" if ready else "partial-universe"
                reason = f"latest universe_count={u}; strict target=350"
        elif ds == "Macro_Market":
            z = macro[macro["series_id"].astype(str) == sid] if not macro.empty else pd.DataFrame()
            obs = len(z)
            if obs:
                latest = str(z["obs_date"].max())
                ready = obs >= 5
                quality = "proxy" if sid == "macro_kofr" else "direct/official-or-market"
                reason = f"{obs} stored observations"
        elif sid == "eps_sector":
            obs = len(eps)
            latest = str(eps["obs_date"].max()) if obs else None
            weeks = eps[eps["sector"].astype(str) == "시장전체"]["obs_date"].nunique() if obs else 0
            sectors = set(eps.loc[eps["sector"].astype(str) != "시장전체", "sector"].dropna().astype(str)) if obs else set()
            required = {"반도체/IT","자동차","2차전지","바이오·헬스케어","금융","조선·기계","미디어·플랫폼"}
            missing = sorted(required - sectors)
            ready = weeks >= 4 and not missing
            quality = "proxy-public" if obs else "missing"
            reason = f"market weeks={weeks}; missing EPS-required sectors={missing}"
        elif ds == "Fundamental":
            if not fund.empty:
                sector = str(r.get("sector", ""))
                indicator = str(r.get("indicator", ""))
                sector_aliases = [sector]
                if sector == "바이오·헬스케어": sector_aliases.append("바이오·헬스")
                z = fund[fund["sector"].astype(str).isin(sector_aliases)]
                tokens = [t for t in indicator.replace("·","/").replace("/"," ").replace("-"," ").split() if len(t) >= 2]
                if tokens:
                    mask = pd.Series(False, index=z.index)
                    for t in tokens:
                        mask = mask | z["indicator"].astype(str).str.contains(t, regex=False)
                    z = z[mask]
                obs = len(z)
                if obs:
                    latest_dt = z["obs_date"].max()
                    latest = latest_dt.strftime("%Y-%m-%d") if pd.notna(latest_dt) else None
                status = str(r.get("status", ""))
                standardized_proxy = ("proxy_active" in status) or ("active_partial_history" in status)
                ready = obs >= 3 or (obs >= 1 and standardized_proxy)
                quality = "direct-or-standardized-proxy" if ready else "partial"
                reason = f"matched observations={obs}; registry_status={status}"
        elif sid == "account_snapshots":
            ready = True
            quality = "private-archive"
            reason = "intentionally excluded from public GitHub; managed in ChatGPT Archive"
        elif sid == "derived_weekly":
            obs = min(len(momentum), len(rs))
            latest_candidates = []
            if not momentum.empty: latest_candidates.append(str(momentum["date"].max()))
            if not rs.empty: latest_candidates.append(str(rs["date"].max()))
            latest = max(latest_candidates) if latest_candidates else None
            ready = (not momentum.empty) and (not rs.empty) and (not haa.empty) and (not breadth.empty)
            quality = "price-derived-only" if ready else "partial"
            reason = "price/RS/HAA/breadth derived inputs present; full 60/40 sector score requires fundamental gate"

        rows.append({
            "series_id": sid,
            "dataset": ds,
            "sector": r.get("sector", ""),
            "indicator": r.get("indicator", ""),
            "observations": obs,
            "latest_observation": latest,
            "ready": bool(ready),
            "quality": quality,
            "reason": reason,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    failures = out[~out["ready"]]
    by_dataset = out.groupby("dataset")["ready"].agg(["count","sum"])
    report = {
        "generated_at_kst": datetime.now(KST).isoformat(),
        "registered_series": int(len(out)),
        "ready_series": int(out["ready"].sum()),
        "not_ready_series": int((~out["ready"]).sum()),
        "completion_pct": float(out["ready"].mean()*100 if len(out) else 0),
        "strict_complete": bool(failures.empty),
        "not_ready": failures[["series_id","dataset","sector","indicator","reason"]].to_dict("records"),
        "dataset_summary": by_dataset.reset_index().to_dict("records"),
        "privacy_note": "Personal account snapshots are deliberately kept outside the public GitHub repository and are counted as ready only when present in the private ChatGPT Archive.",
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
