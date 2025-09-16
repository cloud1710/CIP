"""
rfm_labeling.py
Ánh xạ bộ (R,F,M) sang 'RFM_Level' theo rule rfm_level_v2.
"""

from __future__ import annotations
import pandas as pd

def rfm_level_v2(row) -> str:
    R, F, M = int(row["R"]), int(row["F"]), int(row["M"])

    if (R >= 4) and (F >= 4) and (M >= 4):
        return "STARS"
    if (R == 5) and (F == 1) and (M <= 2):
        return "NEW"
    if (M >= 4) and (R >= 3):
        return "BIG SPENDER"
    if (F == 5) and (R >= 3):
        return "LOYAL"
    if (R >= 4) and (F <= 2) and (M <= 3):
        return "ACTIVE"
    if (R == 1):
        return "LOST"
    if (R >= 3) and (F <= 2) and (M <= 2):
        return "LIGHT"
    return "REGULARS"


def apply_rfm_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm cột 'RFM_Level' dựa trên cột R,F,M đã tồn tại.
    """
    work = df.copy()
    if not {"R","F","M"}.issubset(work.columns):
        raise ValueError("Thiếu cột R,F,M. Hãy chạy compute_rfm_scores trước.")
    work["RFM_Level"] = work.apply(rfm_level_v2, axis=1)
    return work