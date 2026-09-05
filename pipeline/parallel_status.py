from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def main() -> None:
    comp = read_json(META / "completeness_report.json")
    breadth = read_json(META / "breadth_last_run.json")
    macro = read_csv(RAW / "macro" / "macro_market_archive.csv")

    latest_universe = None
    if breadth:
        latest_universe = breadth.get("latest", {}).get("universe_count")

    macro_counts = {}
    for sid in ["macro_kr3y", "macro_kr10y", "macro_kofr", "macro_cd91"]:
        n = int((macro["series_id"].astype(str) == sid).sum()) if not macro.empty and "series_id" in macro.columns else 0
        macro_counts[sid] = n

    fundamental_missing = []
    for x in comp.get("not_ready", []):
        if x.get("dataset") == "Fundamental":
            fundamental_missing.append(x.get("series_id"))

    status = {
        "stage_2_bb350": {
            "target": 350,
            "latest_universe_count": latest_universe,
            "complete": latest_universe == 350,
        },
        "stage_3_rates": {
            "target_each": 65,
            "counts": macro_counts,
            "complete": all(v >= 65 for v in macro_counts.values()),
        },
        "stage_4_fundamentals": {
            "target_missing": 0,
            "missing_count": len(fundamental_missing),
            "missing_series": fundamental_missing,
            "complete": len(fundamental_missing) == 0,
        },
        "overall": {
            "ready_series": comp.get("ready_series"),
            "registered_series": comp.get("registered_series"),
            "completion_pct": comp.get("completion_pct"),
            "strict_complete": comp.get("strict_complete"),
        },
    }
    out = META / "parallel_completion_status.json"
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
