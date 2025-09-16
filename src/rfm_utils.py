"""
rfm_utils.py
Các hàm tiện ích dùng chung cho RFM (không bắt buộc).
"""

from __future__ import annotations
import pandas as pd

def add_rfm_code(df: pd.DataFrame,
                 r_col: str = "R",
                 f_col: str = "F",
                 m_col: str = "M",
                 code_col: str = "RFM_Segment") -> pd.DataFrame:
    work = df.copy()
    if not {r_col, f_col, m_col}.issubset(work.columns):
        raise ValueError("Thiếu R/F/M để tạo code.")
    work[code_col] = work[[r_col, f_col, m_col]].astype(str).agg("".join, axis=1)
    return work


def distribution_table(df: pd.DataFrame,
                       col: str) -> pd.DataFrame:
    """
    Thống kê count & % cho 1 cột phân loại.
    """
    counts = df[col].value_counts(dropna=False)
    pct = (counts / counts.sum() * 100).round(2)
    out = (pd.DataFrame({"count": counts, "pct": pct})
           .reset_index()
           .rename(columns={"index": col}))
    return out
