"""
rfm_base.py
Xây dựng bảng RFM cơ sở từ orders_full.csv (đÃ chuẩn hoá).

Giả định file orders có các cột:
- member_number (ID khách)
- order_id (mã đơn)
- date (ngày, kiểu datetime hoặc string dd-MM-YYYY)
- gross_sales (doanh thu dòng / đơn)
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional

DEFAULT_COLS = {
    "customer": "member_number",
    "order": "order_id",
    "date": "date",
    "monetary": "gross_sales"
}

def load_orders(path: str | Path,
                parse_dates: bool = True,
                date_col: str = "date",
                dayfirst: bool = True) -> pd.DataFrame:
    """
    Đọc file orders_full.csv và ép kiểu ngày nếu cần.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    return df


def build_rfm_snapshot(df: pd.DataFrame,
                       snapshot_date: Optional[pd.Timestamp] = None,
                       customer_col: str = DEFAULT_COLS["customer"],
                       order_col: str = DEFAULT_COLS["order"],
                       date_col: str = DEFAULT_COLS["date"],
                       monetary_col: str = DEFAULT_COLS["monetary"],
                       rename_customer_to: str = "customer_id") -> pd.DataFrame:
    """
    Tạo bảng RFM gốc với các cột:
    - customer_id
    - Recency (số ngày từ lần mua cuối đến snapshot_date)
    - Frequency (số đơn distinct)
    - Monetary (tổng doanh thu)

    Parameters
    ----------
    snapshot_date:
        Nếu None → dùng max(date_col).
    rename_customer_to:
        Chuẩn hoá tên cột ID khách cuối.
    """
    work = df.copy()

    if snapshot_date is None:
        snapshot_date = pd.to_datetime(work[date_col].max()).normalize()

    grouped = (work
               .groupby(customer_col)
               .agg(
                    LastPurchase=(date_col, "max"),
                    Frequency=(order_col, "nunique"),
                    Monetary=(monetary_col, "sum")
               )
               .reset_index())

    grouped["Recency"] = (snapshot_date - grouped["LastPurchase"].dt.normalize()).dt.days
    grouped = grouped.rename(columns={customer_col: rename_customer_to})
    grouped = grouped[[rename_customer_to, "Recency", "Frequency", "Monetary"]]

    # Bảo đảm kiểu số chuẩn
    grouped["Frequency"] = grouped["Frequency"].astype(int)
    grouped["Monetary"] = grouped["Monetary"].astype(float).round(2)
    grouped["Recency"] = grouped["Recency"].astype(int)

    return grouped