"""
rfm_kmeans.py
Pipeline KMeans với:
- Biến đổi: R -> -log1p, F -> log1p, M -> log1p
- StandardScaler
- Trọng số (w_R, w_F, w_M)
- Lựa chọn k qua inertia (Elbow) + silhouette
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

R_COL = "Recency"
F_COL = "Frequency"
M_COL = "Monetary"

def make_preprocess_transformer() -> ColumnTransformer:
    """
    Tạo ColumnTransformer: R:-log1p, F:log1p, M:log1p + impute median.
    """
    log_tf      = FunctionTransformer(np.log1p, validate=False)
    r_logneg_tf = FunctionTransformer(lambda x: -np.log1p(x), validate=False)

    preprocess = ColumnTransformer([
        ("recency_logneg", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rlogneg", r_logneg_tf)
        ]), [R_COL]),
        ("freq_log", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("log", log_tf)
        ]), [F_COL]),
        ("mon_log", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("log", log_tf)
        ]), [M_COL]),
    ])
    return preprocess


def make_kmeans_pipeline(k: int,
                         w_R: float = 1.5,
                         w_F: float = 1.0,
                         w_M: float = 1.2,
                         n_init: int = 20,
                         random_state: int = 42) -> Pipeline:
    """
    Xây dựng full pipeline KMeans với trọng số sau scaler.
    """
    preprocess = make_preprocess_transformer()
    weights = np.array([w_R, w_F, w_M], dtype=float)
    weight_tf = FunctionTransformer(lambda Z: Z * weights, validate=False)

    pipe = Pipeline([
        ("prep", preprocess),
        ("scaler", StandardScaler()),
        ("weight", weight_tf),
        ("kmeans", KMeans(n_clusters=k, init="k-means++",
                          n_init=n_init, random_state=random_state))
    ])
    return pipe


def kmeans_sweep(rfm_df: pd.DataFrame,
                 k_range: Iterable[int] = range(2, 11),
                 w_R: float = 1.5,
                 w_F: float = 1.0,
                 w_M: float = 1.2) -> pd.DataFrame:
    """
    Chạy nhiều k, trả về DataFrame gồm:
    k, inertia, silhouette
    """
    X = rfm_df[[R_COL, F_COL, M_COL]].copy()
    inertias = []
    sils = []

    for k in k_range:
        pipe = make_kmeans_pipeline(k, w_R=w_R, w_F=w_F, w_M=w_M)
        labels = pipe.fit_predict(X)
        Z = pipe[:-1].transform(X)
        inertias.append(pipe.named_steps["kmeans"].inertia_)
        try:
            sils.append(silhouette_score(Z, labels))
        except Exception:
            sils.append(np.nan)

    return pd.DataFrame({
        "k": list(k_range),
        "inertia": inertias,
        "silhouette": sils
    })


def fit_kmeans_final(rfm_df: pd.DataFrame,
                     k: int,
                     w_R: float = 1.5,
                     w_F: float = 1.0,
                     w_M: float = 1.2) -> Tuple[pd.Series, Pipeline, np.ndarray]:
    """
    Huấn luyện cuối cùng với k chọn.
    Return:
      labels (pd.Series),
      pipeline (để transform/ dự đoán lại),
      Z (numpy array sau transform & weight)
    """
    X = rfm_df[[R_COL, F_COL, M_COL]].copy()
    pipe = make_kmeans_pipeline(k, w_R=w_R, w_F=w_F, w_M=w_M)
    labels = pipe.fit_predict(X)
    Z = pipe[:-1].transform(X)
    return pd.Series(labels, index=X.index, name="cluster_kmeans"), pipe, Z


def profile_clusters_kmeans(rfm_df: pd.DataFrame,
                            labels: pd.Series,
                            customer_col: str = "customer_id") -> pd.DataFrame:
    """
    Tạo profile mean/median + count cho từng cụm kmeans.
    """
    base = rfm_df.copy()
    base = base.join(labels)

    stats = (base
             .groupby("cluster_kmeans")[[R_COL, F_COL, M_COL]]
             .agg(["mean", "median"])
             .round(2))

    counts = (base.groupby("cluster_kmeans")
              .size()
              .to_frame("count"))

    profile = pd.concat([stats, counts], axis=1)
    return profile