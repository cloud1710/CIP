"""
rfm_rule_scoring.py
Chấm điểm R, F, M theo rule:
- R: bins nghiệp vụ (mặc định: (-1,7,30,90,180,inf]) -> 5..1
- F, M: winsor 99% + chia quintile 20/40/60/80 -> 1..5
Sinh thêm: RFM_Segment (chuỗi) & RFM_Score (tổng 3 số).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Optional

def _winsor(series: pd.Series, p: float = 0.99) -> pd.Series:
    if series.empty:
        return series
    up = series.quantile(p)
    return series.clip(upper=up)


def _quantile_breaks(series: pd.Series,
                     probs: Sequence[float] = (0.2, 0.4, 0.6, 0.8)) -> pd.Series:
    return series.quantile(probs)


def _qscore(value: float, q: pd.Series) -> int:
    # q có 4 phần tử (20/40/60/80). Trả về 1..5
    if value <= q.iloc[0]:
        return 1
    elif value <= q.iloc[1]:
        return 2
    elif value <= q.iloc[2]:
        return 3
    elif value <= q.iloc[3]:
        return 4
    return 5


def compute_rfm_scores(rfm_df: pd.DataFrame,
                       recency_col: str = "Recency",
                       frequency_col: str = "Frequency",
                       monetary_col: str = "Monetary",
                       recency_bins: Optional[Sequence[int]] = None,
                       winsor_p: float = 0.99,
                       r_labels: Sequence[int] = (5,4,3,2,1),
                       add_segment: bool = True) -> pd.DataFrame:
    """
    Thêm cột R,F,M,RFM_Segment,RFM_Score vào dataframe RFM gốc.

    Parameters
    ----------
    recency_bins:
        Danh sách biên cho pd.cut. Mặc định [-1, 7, 30, 90, 180, inf].
    winsor_p:
        Ngưỡng winsor upper (99% = 0.99).
    r_labels:
        Nhãn điểm R tương ứng số khoảng (mặc định 5..1).
    add_segment:
        Nếu True tạo cột 'RFM_Segment'.

    Returns
    -------
    pd.DataFrame
    """
    if recency_bins is None:
        recency_bins = [-1, 7, 30, 90, 180, float("inf")]

    df = rfm_df.copy()

    # R score
    df["R"] = pd.cut(df[recency_col],
                     bins=recency_bins,
                     labels=r_labels,
                     include_lowest=True).astype(int)

    # Winsor + quantile cho F, M
    Fw = _winsor(df[frequency_col], winsor_p)
    Mw = _winsor(df[monetary_col], winsor_p)

    qF = _quantile_breaks(Fw)
    qM = _quantile_breaks(Mw)

    df["F"] = Fw.apply(lambda v: _qscore(v, qF)).astype(int)
    df["M"] = Mw.apply(lambda v: _qscore(v, qM)).astype(int)

    if add_segment:
        df["RFM_Segment"] = df[["R","F","M"]].astype(str).agg("".join, axis=1)

    df["RFM_Score"] = df[["R","F","M"]].sum(axis=1).astype(int)
    return df
