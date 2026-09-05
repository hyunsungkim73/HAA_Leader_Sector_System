from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
RAW = ROOT / "data" / "raw"

RATE_IDS = ["macro_kr3y", "macro_kr10y", "macro_kofr", "macro_cd91"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def direct_rate_counts(macro: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sid in RATE_IDS:
        if macro.empty or "series_id" not in macro.columns:
            counts[sid] = 0
            continue
        z = macro[macro["series_id"].astype(str) == sid].copy()
        if "source_type" not in z.columns:
            counts[sid] = 0
            continue
        proxy = z["source_type"].fillna("").astype(str).str.lower().str.contains("proxy", regex=False)
        counts[sid] = int((~proxy).sum())
    return counts


def main() -> None:
    comp = read_json(META / "completeness_report.json")
    series = read_csv(META / "completeness_series.csv")
    macro = read_csv(RAW / "macro" / "macro_market_archive.csv")

    rows = series.set_index("series_id") if not series.empty and "series_id" in series.columns else pd.DataFrame()

    bb_ready = False
    bb_quality = None
    bb_reason = "missing bb350 row"
    if not rows.empty and "bb350" in rows.index:
        bb = rows.loc["bb350"]
        bb_ready = as_bool(bb.get("ready", False))
        bb_quality = str(bb.get("quality", ""))
        bb_reason = str(bb.get("reason", ""))
    stage2_complete = bb_ready and bb_quality == "exact"

    direct_counts = direct_rate_counts(macro)
    rate_ready: dict[str, bool] = {}
    rate_reasons: dict[str, str] = {}
    for sid in RATE_IDS:
        if not rows.empty and sid in rows.index:
            r = rows.loc[sid]
            rate_ready[sid] = as_bool(r.get("ready", False))
            rate_reasons[sid] = str(r.get("reason", ""))
        else:
            rate_ready[sid] = False
            rate_reasons[sid] = "missing completeness row"
    stage3_complete = all(rate_ready.values()) and all(direct_counts[sid] >= 65 for sid in RATE_IDS)

    summary = {r.get("dataset"): r for r in comp.get("dataset_summary", [])}
    fund = summary.get("Fundamental", {})
    fundamental_count = int(fund.get("count", 0) or 0)
    fundamental_ready = int(fund.get("sum", 0) or 0)
    stage4_complete = fundamental_count == 50 and fundamental_ready == 50

    status = {
        "stage_2_bb350": {
            "requirement": "exact KOSPI200+KOSDAQ150 BB350",
            "ready": bb_ready,
            "quality": bb_quality,
            "reason": bb_reason,
            "complete": stage2_complete,
        },
        "stage_3_rates": {
            "requirement": "65 direct observations each; proxy rows excluded",
            "target_each": 65,
            "direct_counts": direct_counts,
            "ready": rate_ready,
            "reasons": rate_reasons,
            "complete": stage3_complete,
        },
        "stage_4_fundamentals": {
            "requirement": "Fundamental 50/50 ready",
            "count": fundamental_count,
            "ready_count": fundamental_ready,
            "complete": stage4_complete,
        },
        "overall": {
            "ready_series": comp.get("ready_series"),
            "registered_series": comp.get("registered_series"),
            "completion_pct": comp.get("completion_pct"),
            "strict_complete": comp.get("strict_complete"),
            "data_gates_complete": stage2_complete and stage3_complete and stage4_complete,
        },
    }
    out = META / "parallel_completion_status.json"
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
