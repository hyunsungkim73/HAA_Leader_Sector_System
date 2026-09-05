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
    "price_semi_it": "semiconductor", "price_auto": "auto", "price_battery": "battery",
    "price_bio": "bio_health", "price_financial": "financial", "price_ship_machinery": "shipbuilding",
    "price_steel": "steel_materials", "price_consumer": "consumer", "price_media_platform": "media_platform",
    "price_construction": "construction", "price_kbeauty": "k_beauty", "price_kfood": "k_food",
    "price_biosim_cdmo": "biosimilar_cdmo", "price_kospi200": "KOSPI200",
}

EPS_SERIES = {
    "semi_eps": "반도체/IT", "auto_eps": "자동차", "bio_eps": "바이오·헬스케어",
    "fin_eps": "금융", "ship_eps": "조선·기계", "plat_eps": "미디어·플랫폼",
}

FUND_KEYWORDS = {
    "semi_price": ["DDR", "NAND", "HBM"],
    "semi_spot_demand": ["현물 구매수요", "현물 거래", "spot demand"],
    "semi_inventory": ["재고 타이트니스", "재고일수", "채널재고"],
    "semi_supply_capex": ["공급/CAPEX", "가동률", "CAPEX"],
    "auto_inventory": ["days supply", "재고"],
    "auto_incentive": ["인센티브", "할인"],
    "auto_asp_mix": ["ASP", "믹스", "hybrid"],
    "auto_sales_orders": ["판매", "주문", "백로그"],
    "battery_ev_sales": ["EV 판매"],
    "battery_install": ["배터리 사용량", "설치량", "출하량"],
    "battery_util": ["가동률"],
    "battery_customer_inventory": ["EV days supply", "고객사 재고", "재고소진"],
    "battery_cell_price_margin": ["셀 시장가격", "RMB/Wh", "셀마진"],
    "bio_rx_share": ["시장점유율", "volume share", "처방"],
    "bio_sales_volume": ["신제품 매출", "Bioepis revenue", "판매량", "매출 비중"],
    "bio_orders": ["contract value", "신규수주", "수주잔고", "계약"],
    "bio_approval": ["FDA 승인", "출시 바이오시밀러", "허가"],
    "fin_nim": ["NIM"],
    "fin_loans": ["대출"],
    "fin_npl": ["NPL", "연체"],
    "fin_credit_cost": ["credit cost", "대손비용", "충당금"],
    "ship_price": ["신조선가지수"],
    "ship_orders": ["글로벌 신규수주"],
    "ship_backlog": ["수주잔고", "backlog"],
    "ship_mix": ["LNGC 수주잔고", "선종믹스", "H1 신규수주"],
    "steel_product_price": ["열연", "후판", "철근"],
    "steel_rawmat": ["PB분", "원료탄", "철광석 가격"],
    "steel_spread": ["HRC/"],
    "steel_inventory": ["사회재고", "완제품 재고", "유통재고"],
    "steel_output": ["철수 생산", "조강생산", "생산·가동률"],
    "cons_volume": ["구매건수", "판매량", "트래픽"],
    "cons_ticket": ["구매단가", "객단가"],
    "cons_inventory": ["재고액지수", "재고-판매"],
    "cons_promo": ["프로모션", "암묵가격", "할인강도"],
    "cons_margin": ["원가-마진", "마진스프레드", "원재료 가격"],
    "plat_mau": ["MAU", "DAU"],
    "plat_ads": ["광고/구독", "광고수요", "광고 매출"],
    "plat_gmv": ["GTV", "GMV"],
    "plat_monetize": ["수익화", "Financial Platform 매출", "Commerce 매출"],
    "const_demand": ["분양 물량", "청약경쟁률"],
    "const_unsold": ["미분양"],
    "const_orders": ["건설수주", "신규수주"],
    "const_pf": ["PF"],
    "const_cost": ["공사비", "원가율"],
}


def safe_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_registry() -> pd.DataFrame:
    rows = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    reg = pd.DataFrame(rows)
    if len(reg) != 80 or reg["series_id"].nunique() != 80:
        raise RuntimeError(f"registry must contain exactly 80 unique series; rows={len(reg)} unique={reg['series_id'].nunique()}")
    return reg


def load_fundamentals() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "fundamentals").glob("*.csv")):
        f = safe_csv(path)
        if not f.empty and {"obs_date", "sector", "indicator"}.issubset(f.columns):
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    f = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["obs_date", "sector", "subsector", "indicator", "source"]:
        if c not in f.columns: f[c] = ""
    f["obs_date"] = pd.to_datetime(f["obs_date"], errors="coerce")
    return f.drop_duplicates(["obs_date", "sector", "subsector", "indicator", "source"], keep="last")


def match_fund(fund: pd.DataFrame, sid: str, sector: str) -> pd.DataFrame:
    if fund.empty: return pd.DataFrame()
    aliases = [sector]
    if sector == "바이오·헬스케어": aliases.append("바이오·헬스")
    z = fund[fund["sector"].astype(str).isin(aliases)].copy()
    kws = FUND_KEYWORDS.get(sid, [])
    if not kws: return pd.DataFrame()
    mask = pd.Series(False, index=z.index)
    for kw in kws:
        mask |= z["indicator"].astype(str).str.contains(kw, regex=False, case=False)
        if "notes" in z.columns:
            mask |= z["notes"].fillna("").astype(str).str.contains(kw, regex=False, case=False)
    return z[mask]


def main() -> None:
    reg = load_registry()
    price_cov = safe_csv(META / "price_coverage.csv")
    price_cov = price_cov.set_index("series") if not price_cov.empty else pd.DataFrame()
    fund = load_fundamentals()
    macro = safe_csv(RAW / "macro" / "macro_market_archive.csv")
    eps = safe_csv(RAW / "eps" / "eps_revisions_archive.csv")
    breadth = safe_csv(DERIVED / "breadth_bb20_2sigma_daily.csv")
    haa = safe_csv(DERIVED / "tip_haa_weekly.csv")
    momentum = safe_csv(DERIVED / "momentum_weekly.csv")
    rs = safe_csv(DERIVED / "relative_strength_weekly.csv")

    rows = []
    for _, r in reg.iterrows():
        sid, ds, sector = r["series_id"], r["dataset"], str(r.get("sector", ""))
        obs, latest, ready, quality, reason = 0, None, False, "missing", "no observations"

        if sid in PRICE_MAP:
            name = PRICE_MAP[sid]
            if not price_cov.empty and name in price_cov.index:
                pc = price_cov.loc[name]; obs = int(pc["obs"]); latest = str(pc["end"])
                ready = bool(pc["ready_220"]); quality = "direct/market" if ready else "partial"
                reason = f"{obs} daily observations; 220 required"
        elif sid == "tip_tr":
            if not price_cov.empty and "TIP" in price_cov.index:
                pc = price_cov.loc["TIP"]; obs = int(pc["obs"]); latest = str(pc["end"])
                ready = bool(pc["ready_252"]) and not haa.empty
                quality = "market total-return proxy" if ready else "partial"; reason = f"TIP observations={obs}; HAA rows={len(haa)}"
        elif sid == "bb350":
            if not breadth.empty:
                obs = len(breadth); latest = str(breadth["date"].max())
                u = int(pd.to_numeric(breadth["universe_count"], errors="coerce").dropna().iloc[-1])
                ready = u >= 350 and obs >= 20; quality = "exact" if ready else "partial-universe"
                reason = f"breadth rows={obs}; latest universe_count={u}; strict target=350"
        elif ds == "Macro_Market":
            z = macro[macro["series_id"].astype(str) == sid] if not macro.empty else pd.DataFrame(); obs = len(z)
            if obs:
                latest = str(z["obs_date"].max())
                required = 90 if sid not in {"macro_kofr", "macro_cd91", "macro_kr3y", "macro_kr10y"} else 65
                ready = obs >= required; quality = "proxy" if sid == "macro_kofr" else "direct/official-or-market"
                reason = f"{obs} stored observations; {required} required for 3-month direction"
        elif sid == "eps_sector":
            obs = len(eps); latest = str(eps["obs_date"].max()) if obs else None
            required_sectors = ["반도체/IT", "자동차", "2차전지", "바이오·헬스케어", "금융", "조선·기계", "미디어·플랫폼"]
            counts = {s: int((eps["sector"].astype(str) == s).sum()) if obs else 0 for s in required_sectors}
            market_weeks = int(eps[eps["sector"].astype(str) == "시장전체"]["obs_date"].nunique()) if obs else 0
            missing = {s:n for s,n in counts.items() if n < 3}
            ready = market_weeks >= 4 and not missing; quality = "proxy-public" if obs else "missing"
            reason = f"market weeks={market_weeks}; sector observations={counts}; need >=3 each"
        elif sid in EPS_SERIES:
            s = EPS_SERIES[sid]; z = eps[eps["sector"].astype(str) == s] if not eps.empty else pd.DataFrame(); obs = len(z)
            if obs: latest = str(z["obs_date"].max())
            ready = obs >= 3; quality = "proxy-public" if obs else "missing"; reason = f"sector EPS observations={obs}; 3 required"
        elif ds == "Fundamental":
            z = match_fund(fund, sid, sector); obs = len(z)
            if obs:
                latest_dt = z["obs_date"].max(); latest = latest_dt.strftime("%Y-%m-%d") if pd.notna(latest_dt) else None
            ready = obs >= 3; quality = "direct-or-standardized-proxy" if ready else "partial"
            reason = f"matched native/proxy observations={obs}; 3 required for direction"
        elif sid == "account_snapshots":
            ready = True; quality = "private-archive"; reason = "intentionally excluded from public GitHub; managed in private ChatGPT Archive"
        elif sid == "derived_weekly":
            obs = min(len(momentum), len(rs)); latest_candidates = []
            if not momentum.empty: latest_candidates.append(str(momentum["date"].max()))
            if not rs.empty: latest_candidates.append(str(rs["date"].max()))
            latest = max(latest_candidates) if latest_candidates else None
            ready = (not momentum.empty) and (not rs.empty) and (not haa.empty) and (not breadth.empty)
            quality = "price-derived-core" if ready else "partial"; reason = "price/RS/HAA/breadth inputs present; final 60/40 scoring still depends on fundamental gates"

        rows.append({"series_id":sid,"dataset":ds,"sector":sector,"indicator":r.get("indicator", ""),"observations":obs,"latest_observation":latest,"ready":bool(ready),"quality":quality,"reason":reason})

    out = pd.DataFrame(rows); out.to_csv(OUT_CSV, index=False)
    failures = out[~out["ready"]]; by_dataset = out.groupby("dataset")["ready"].agg(["count","sum"])
    report = {
        "generated_at_kst": datetime.now(KST).isoformat(), "registered_series": int(len(out)),
        "ready_series": int(out["ready"].sum()), "not_ready_series": int((~out["ready"]).sum()),
        "completion_pct": float(out["ready"].mean()*100 if len(out) else 0), "strict_complete": bool(failures.empty),
        "not_ready": failures[["series_id","dataset","sector","indicator","reason"]].to_dict("records"),
        "dataset_summary": by_dataset.reset_index().to_dict("records"),
        "privacy_note": "Personal account snapshots remain outside the public GitHub repository and are managed in the private ChatGPT Archive."
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
