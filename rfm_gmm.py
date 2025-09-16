"""
rfm_gmm.py
Gaussian Mixture Model cho RFM với:
- Transform giống KMeans (R:-log1p, F/M:log1p) + scaler + trọng số
- Sweep k với BIC / AIC / ICL / Silhouette
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

R_COL = "Recency"
F_COL = "Frequency"
M_COL = "Monetary"

def _make_preprocess() -> ColumnTransformer:
    log_tf      = FunctionTransformer(np.log1p, validate=False)
    r_logneg_tf = FunctionTransformer(lambda x: -np.log1p(x), validate=False)
    return ColumnTransformer([
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


def make_gmm_pipeline(k: int,
                      w_R: float = 1.5,
                      w_F: float = 1.0,
                      w_M: float = 1.2,
                      cov: str = "full",
                      reg: float = 1e-6,
                      seed: int = 42,
                      n_init: int = 5) -> Pipeline:
    preprocess = _make_preprocess()
    weights = np.array([w_R, w_F, w_M])
    weight_tf = FunctionTransformer(lambda Z: Z * weights, validate=False)
    pipe = Pipeline([
        ("prep", preprocess),
        ("scaler", StandardScaler()),
        ("weight", weight_tf),
        ("gmm", GaussianMixture(n_components=k,
                                covariance_type=cov,
                                reg_covar=reg,
                                n_init=n_init,
                                random_state=seed))
    ])
    return pipe


def _icl_score(gm: GaussianMixture, Z: np.ndarray, eps: float = 1e-12) -> float:
    tau = gm.predict_proba(Z)
    entropy = (tau * np.log(tau + eps)).sum()  # <= 0
    return gm.bic(Z) + entropy  # lower better


def gmm_sweep(rfm_df: pd.DataFrame,
              k_values: Iterable[int] = range(2, 11),
              w_R: float = 1.5,
              w_F: float = 1.0,
              w_M: float = 1.2,
              cov: str = "full",
              reg: float = 1e-6,
              seed: int = 42) -> pd.DataFrame:
    """
    Chạy nhiều k => tính BIC, AIC, ICL, Silhouette (+ min cluster size %).
    """
    X = rfm_df[[R_COL, F_COL, M_COL]].copy()
    rows = []
    N = len(X)

    for k in k_values:
        pipe = make_gmm_pipeline(k, w_R=w_R, w_F=w_F, w_M=w_M,
                                 cov=cov, reg=reg, seed=seed)
        pipe.fit(X)
        Z = pipe[:-1].transform(X)
        gm = pipe.named_steps["gmm"]
        labels = gm.predict(Z)

        bic = gm.bic(Z)
        aic = gm.aic(Z)
        icl = _icl_score(gm, Z)
        try:
            sil = silhouette_score(Z, labels)
        except Exception:
            sil = np.nan

        sizes = np.bincount(labels, minlength=k)
        min_size = sizes.min()
        min_pct = min_size / N * 100

        rows.append((k, bic, aic, icl, sil, sizes, min_size, min_pct))

    res = pd.DataFrame(rows, columns=[
        "k", "BIC", "AIC", "ICL", "Silhouette", "sizes", "min_size", "min_pct"
    ]).set_index("k")
    return res


def fit_gmm_final(rfm_df: pd.DataFrame,
                  k: int,
                  w_R: float = 1.5,
                  w_F: float = 1.0,
                  w_M: float = 1.2,
                  cov: str = "full",
                  reg: float = 1e-6,
                  seed: int = 42) -> Tuple[pd.Series, Pipeline, np.ndarray]:
    """
    Fit mô hình GMM cuối cùng cho k chọn.
    Return:
      labels (Series cluster_gmm),
      pipeline,
      Z (data sau transform)
    """
    X = rfm_df[[R_COL, F_COL, M_COL]].copy()
    pipe = make_gmm_pipeline(k, w_R=w_R, w_F=w_F, w_M=w_M,
                             cov=cov, reg=reg, seed=seed)
    pipe.fit(X)
    Z = pipe[:-1].transform(X)
    gm = pipe.named_steps["gmm"]
    labels = gm.predict(Z)
    return pd.Series(labels, index=X.index, name="cluster_gmm"), pipe, Z


def profile_clusters_gmm(rfm_df: pd.DataFrame,
                         labels: pd.Series) -> pd.DataFrame:
    """
    Profile mean/median + count.
    """
    base = rfm_df.copy().join(labels)
    stats = (base
             .groupby("cluster_gmm")[[R_COL, F_COL, M_COL]]
             .agg(["mean", "median"])
             .round(2))
    counts = base.groupby("cluster_gmm").size().to_frame("count")
    profile = pd.concat([stats, counts], axis=1)
    return profile