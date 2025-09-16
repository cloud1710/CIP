# scripts/train_gmm.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.rfm_base import load_orders, build_rfm_snapshot
from src.rfm_rule_scoring import compute_rfm_scores
from src.model_io import (
    compute_dataframe_hash,
    build_metadata,
    save_clustering_package
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/orders_full.csv")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--outdir", default="models/gmm")
    parser.add_argument("--log1p", action="store_true", help="Dùng log1p cho Recency/Frequency/Monetary")
    args = parser.parse_args()

    print(f"[INFO] Train GMM version={args.version}, k={args.k}")

    orders = load_orders(pathlib.Path(args.data))
    rfm = build_rfm_snapshot(orders)
    rfm = compute_rfm_scores(rfm)

    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy().astype(float)

    pipeline_steps = []
    if args.log1p:
        X = np.log1p(X)
        pipeline_steps.append("log1p")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pipeline_steps.append("standard_scale")

    gmm = GaussianMixture(n_components=args.k, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    profile = (pd.concat([
        rfm[features].reset_index(drop=True),
        pd.Series(labels, name="cluster")
    ], axis=1)
        .groupby("cluster")
        .agg(["mean", "median", "count"])
    )

    fingerprint = compute_dataframe_hash(rfm, cols=features)
    metadata = build_metadata(
        model_version=args.version,
        n_components=args.k,
        features=features,
        data_fingerprint=fingerprint,
        extras={
            "pipeline_steps": pipeline_steps,
            "original_rows": len(rfm)
        }
    )

    out_dir = pathlib.Path(args.outdir) / f"gmm_rfm_{args.version}"

    # Lưu model là dict (dễ pickle)
    model_obj = {
        "gmm": gmm,
        "scaler": scaler,
        "pipeline_steps": pipeline_steps
    }

    save_clustering_package(
        out_dir=out_dir,
        model=model_obj,
        metadata=metadata,
        labels=pd.Series(labels, name="cluster"),
        feature_df=rfm.set_index("customer_id") if "customer_id" in rfm.columns else rfm,
        profile=profile
    )

    seg_master = rfm.copy()
    seg_master["cluster"] = labels
    seg_master.to_csv(out_dir / "segmentation_master.csv", index=False)

    print(f"[OK] Saved package -> {out_dir}")
    print("[DONE]")

if __name__ == "__main__":
    main()
