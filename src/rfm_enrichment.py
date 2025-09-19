# ===========================
# File: src/rfm_enrichment.py
# ===========================
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

def compute_recent_aov_last3(
    orders_df: pd.DataFrame,
    customer_col: str = "customer_id",
    order_col: str = "order_id",
    amount_pref: tuple = ("gross_sales","net_sales","amount","total","value"),
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Tính AOV 3 đơn gần nhất / khách hàng.
    Trả về DataFrame: [customer_id, recent_aov_3]
    """
    if orders_df.empty:
        return pd.DataFrame(columns=[customer_col, "recent_aov_3"])

    df = orders_df.copy()

    # Chuẩn cột customer
    if customer_col not in df.columns:
        # cố gắng đoán
        cand = next((c for c in ["customer_id","member_number","cust_id","user_id"] if c in df.columns), None)
        if cand is None:
            return pd.DataFrame(columns=[customer_col,"recent_aov_3"])
        if cand != customer_col:
            df = df.rename(columns={cand: customer_col})

    # Bảo đảm order_col
    if order_col not in df.columns:
        return pd.DataFrame(columns=[customer_col, "recent_aov_3"])

    # Chọn cột giá trị
    val_col = next((c for c in amount_pref if c in df.columns), None)

    if val_col:
        order_val = df.groupby([customer_col, order_col])[val_col].sum().reset_index(name="order_value")
    else:
        # fallback: mỗi order = 1
        order_val = df.groupby([customer_col, order_col]).size().reset_index(name="order_value")

    # Ngày đơn
    if date_col in df.columns:
        order_dates = df.groupby([customer_col, order_col])[date_col].min().reset_index()
        order_dates[date_col] = pd.to_datetime(order_dates[date_col], errors="coerce")
        order_val = order_val.merge(order_dates, on=[customer_col, order_col], how="left")
        order_val = order_val.sort_values([customer_col, date_col], ascending=[True, False])
    else:
        order_val["__tmp_date"] = pd.NaT
        order_val = order_val.sort_values([customer_col, order_col], ascending=[True, False])

    order_val["__rank"] = order_val.groupby(customer_col).cumcount() + 1
    recent3 = order_val[order_val["__rank"] <= 3]
    recent_aov = recent3.groupby(customer_col)["order_value"].mean().reset_index(name="recent_aov_3")
    return recent_aov


def _compute_global_norms(rfm: pd.DataFrame) -> Tuple[float,float,float,float,float,float]:
    return (
        rfm["Recency"].min(), rfm["Recency"].max(),
        rfm["Frequency"].min(), rfm["Frequency"].max(),
        rfm["Monetary"].min(), rfm["Monetary"].max()
    )


def _add_normalized(rfm: pd.DataFrame, norms: Tuple[float,float,float,float,float,float]) -> pd.DataFrame:
    rmin, rmax, fmin, fmax, mmin, mmax = norms
    out = rfm.copy()
    out["R_n"] = 1 - (out["Recency"] - rmin) / (rmax - rmin + 1e-9)
    out["F_n"] = (out["Frequency"] - fmin) / (fmax - fmin + 1e-9)
    out["M_n"] = (out["Monetary"] - mmin) / (mmax - mmin + 1e-9)
    for c in ["R_n","F_n","M_n"]:
        out[c] = out[c].clip(0,1)
    return out


def _determine_priority_dimension(row, delta: float) -> str:
    dims = {"R": row["R_n"], "F": row["F_n"], "M": row["M_n"]}
    sorted_dims = sorted(dims.items(), key=lambda x: x[1])
    lowest, low_val = sorted_dims[0]
    highest, high_val = sorted_dims[-1]
    if (high_val - low_val) < delta:
        return "Balanced"
    return {"R":"Recency","F":"Frequency","M":"Monetary"}[lowest]


def _assign_upgrade_goal(row) -> str:
    seg = row.get("RFM_Level")
    F_n = row.get("F_n", 0)
    M_n = row.get("M_n", 0)
    if seg == "REGULARS":
        if F_n > 0.70 and M_n > 0.70: return "LOYAL / BIG SPENDER candidate"
        if F_n > 0.75: return "LOYAL"
        if M_n > 0.80: return "BIG SPENDER"
    elif seg == "BIG SPENDER":
        if F_n < 0.55: return "Boost Frequency → STARS path"
        if F_n >= 0.65: return "STARS"
    elif seg == "LOYAL":
        if M_n >= 0.75: return "STARS"
    elif seg == "LIGHT":
        return "ACTIVE (≥2–3 orders / 30d)"
    elif seg == "NEW":
        return "LIGHT → ACTIVE (2nd order ≤30d)"
    elif seg == "LOST":
        if M_n >= 0.70: return "Recover → BIG SPENDER path"
        return "Reactivate → ACTIVE"
    return ""


def _detect_misalignment(row) -> str:
    seg = row.get("RFM_Level")
    cid = row.get("cluster_gmm")
    # Tuỳ bạn map thực tế: ví dụ cluster 4 = Dormant High Value, cluster 2 = Premium Active
    if seg in ("BIG SPENDER","STARS") and cid in (4,):
        return "VIP Drift risk"
    if seg == "REGULARS" and cid in (2,):
        return "Potential upgrade missed"
    return ""


def enrich_rfm_with_metrics(
    rfm_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    *,
    customer_col: str = "customer_id",
    priority_delta: float = 0.15,
    recency_drift_flag: float = 0.30,
    aov_compression_threshold: float = 0.85,
    next_review_days: int = 14,
    spread_review: bool = False
) -> pd.DataFrame:
    """
    Làm giàu RFM với:
      - recent_aov_3
      - normalized R_n, F_n, M_n
      - drift_recency (Recency vs median cluster)
      - monetary_compress_flag
      - priority_dim
      - upgrade_goal_dynamic
      - segment_cluster_misalignment
      - next_review_date (isoformat)
    """
    if rfm_df.empty:
        return rfm_df.copy()

    rfm = rfm_df.copy()

    # AOV 3 đơn gần nhất
    recent_aov = compute_recent_aov_last3(orders_df, customer_col=customer_col)
    rfm = rfm.merge(recent_aov, on=customer_col, how="left")

    # Normalize
    norms = _compute_global_norms(rfm)
    rfm = _add_normalized(rfm, norms)

    # Expected Recency theo cluster
    if "cluster_gmm" in rfm.columns:
        cluster_expected = (
            rfm.groupby("cluster_gmm")["Recency"].median()
               .rename("Recency_expected_cluster")
        )
        rfm = rfm.merge(cluster_expected, on="cluster_gmm", how="left")
    else:
        rfm["Recency_expected_cluster"] = np.nan

    # Tính các chỉ số dòng
    drift_list = []
    compress_list = []
    recent_aov_list = []
    lifetime_aov_list = []
    priority_list = []
    upgrade_list = []
    misalign_list = []

    for _, row in rfm.iterrows():
        # Drift recency
        drift_val = np.nan
        exp_rec = row.get("Recency_expected_cluster")
        if pd.notna(exp_rec) and exp_rec > 0:
            drift_val = (row["Recency"] - exp_rec) / exp_rec

        # AOV
        lifetime_aov = row["Monetary"] / max(row["Frequency"], 1)
          # recent
        r_aov = row.get("recent_aov_3")
        if pd.isna(r_aov):
            r_aov = lifetime_aov
        compress_flag = (r_aov / (lifetime_aov + 1e-9)) < aov_compression_threshold

        # Priority dimension
        priority_dim = _determine_priority_dimension(row, priority_delta)

        # Upgrade goal
        upgrade_goal = _assign_upgrade_goal(row)

        # Misalignment
        mis = _detect_misalignment(row)

        drift_list.append(drift_val)
        compress_list.append(compress_flag)
        recent_aov_list.append(r_aov)
        lifetime_aov_list.append(lifetime_aov)
        priority_list.append(priority_dim)
        upgrade_list.append(upgrade_goal)
        misalign_list.append(mis)

    rfm["drift_recency"] = drift_list
    rfm["monetary_compress_flag"] = compress_list
    rfm["recent_aov_3"] = recent_aov_list
    rfm["lifetime_aov"] = lifetime_aov_list
    rfm["priority_dim"] = priority_list
    rfm["upgrade_goal_dynamic"] = upgrade_list
    rfm["segment_cluster_misalignment"] = misalign_list

    # Next review date
    base_day = (datetime.today() + timedelta(days=next_review_days)).date()
    if spread_review:
        # phân bổ đều 0–4 ngày
        rfm = rfm.sort_values(customer_col).reset_index(drop=True)
        rfm["next_review_date"] = [
            (base_day + timedelta(days=int(i % 5))).isoformat()
            for i in range(len(rfm))
        ]
    else:
        rfm["next_review_date"] = base_day.isoformat()

    # Flag drift cảnh báo
    rfm["drift_recency_flag"] = (rfm["drift_recency"] > recency_drift_flag).fillna(False)

    return rfm


# ===========================
# (OPTIONAL) Test nhanh khi chạy riêng
# ===========================
if __name__ == "__main__":
    # Ví dụ giả lập
    data = {
        "customer_id":[1,2,3],
        "Recency":[10,40,5],
        "Frequency":[5,2,9],
        "Monetary":[500000,120000,950000],
        "RFM_Level":["REGULARS","LOST","BIG SPENDER"],
        "cluster_gmm":[2,4,2]
    }
    rfm_sample = pd.DataFrame(data)
    orders_sample = pd.DataFrame({
        "customer_id":[1,1,1,2,3,3,3],
        "order_id":[11,12,13,21,31,32,33],
        "gross_sales":[200000,150000,180000,120000,400000,300000,250000],
        "date":pd.date_range("2025-09-01", periods=7, freq="D")
    })
    enriched = enrich_rfm_with_metrics(rfm_sample, orders_sample)
    print(enriched)