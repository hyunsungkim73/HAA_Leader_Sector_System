from __future__ import annotations

import re
import numpy as np
import pandas as pd

import collect_breadth as cb


def normalize_six_digit_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        n = int(value)
        return f"{n:06d}" if 0 <= n <= 999999 else None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        n = int(value)
        return f"{n:06d}" if 0 <= n <= 999999 else None
    s = str(value).strip()
    m = re.fullmatch(r"(\d{1,6})(?:\.0+)?", s)
    return m.group(1).zfill(6) if m else None


cb.normalize_six_digit_code = normalize_six_digit_code

if __name__ == "__main__":
    cb.main()
