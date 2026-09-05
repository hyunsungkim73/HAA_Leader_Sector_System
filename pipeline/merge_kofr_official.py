from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "data" / "raw" / "macro" / "macro_market_archive.csv"
KOFR = ROOT / "data" / "raw" / "macro" / "kofr_ksd_official.csv"
KST = timezone(timedelta(hours=9))
COLS = ["obs_date","series_id","category","indicator","value","unit","frequency","source","source_type","loaded_at","notes"]


def main() -> None:
    if not KOFR.exists() or not KOFR.stat().st_size:
        raise RuntimeError(f"missing official KOFR seed: {KOFR}")
    k = pd.read_csv(KOFR)
    if not {"obs_date", "value"}.issubset(k.columns):
        raise RuntimeError("official KOFR seed must contain obs_date,value")
    k["obs_date"] = pd.to_datetime(k["obs_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    k["value"] = pd.to_numeric(k["value"], errors="coerce")
    k = k.dropna(subset=["obs_date", "value"]).drop_duplicates("obs_date", keep="last")
    if len(k) < 65:
        raise RuntimeError(f"official KOFR observations={len(k)}; at least 65 required")
    k["series_id"] = "macro_kofr"
    k["category"] = "방어자산"
    k["indicator"] = "KOFR"
    k["unit"] = "%"
    k["frequency"] = "daily"
    k["source"] = "KSD KOFR official export https://www.kofr.kr/rate/rate.jsp"
    k["source_type"] = "official"
    k["loaded_at"] = datetime.now(KST).isoformat()
    k["notes"] = "official KSD KOFR Excel export; source file KOFR_20260906.xlsx"
    k = k[COLS]

    old = pd.read_csv(MACRO) if MACRO.exists() and MACRO.stat().st_size else pd.DataFrame(columns=COLS)
    old_counts = old.groupby("series_id").size().to_dict() if not old.empty else {}
    z = pd.concat([old, k], ignore_index=True)
    z["obs_date"] = pd.to_datetime(z["obs_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    z["value"] = pd.to_numeric(z["value"], errors="coerce")
    z = z.dropna(subset=["obs_date", "series_id", "value"]).drop_duplicates(["obs_date", "series_id"], keep="last")
    z = z.sort_values(["series_id", "obs_date"])
    new_counts = z.groupby("series_id").size().to_dict() if not z.empty else {}
    for sid, n in old_counts.items():
        if sid != "macro_kofr" and new_counts.get(sid, 0) < n:
            raise RuntimeError(f"macro archive regression for {sid}: old={n} new={new_counts.get(sid, 0)}")
    direct = z[(z["series_id"] == "macro_kofr") & (~z["source_type"].fillna("").str.lower().str.contains("proxy", regex=False))]
    if len(direct) < 65:
        raise RuntimeError(f"KOFR direct observations after merge={len(direct)}; at least 65 required")
    z.to_csv(MACRO, index=False)
    print(f"KOFR_OFFICIAL_MERGE_OK direct={len(direct)} min={direct['obs_date'].min()} max={direct['obs_date'].max()}")


if __name__ == "__main__":
    main()
