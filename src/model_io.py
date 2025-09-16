# src/model_io.py
import hashlib
import json
import joblib
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Iterable, Union

def compute_dataframe_hash(df: pd.DataFrame, cols: Union[Iterable, None] = None):
    if cols is not None:
        sub = df[list(cols)].copy()
    else:
        sub = df.copy()
    sub = sub.sort_index(axis=1).sort_index()
    raw = sub.to_csv(index=True).encode("utf-8")
    import hashlib as _hl
    return _hl.md5(raw).hexdigest()

def build_metadata(model_version: str,
                   n_components: int,
                   features,
                   data_fingerprint: str,
                   extras: dict = None):
    meta = {
        "model_type": "GMM",
        "version": model_version,
        "n_components": n_components,
        "features": list(features),
        "trained_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "data_fingerprint": data_fingerprint
    }
    if extras:
        meta.update(extras)
    return meta

def save_clustering_package(out_dir: Path,
                            model,
                            metadata: dict,
                            labels: pd.Series,
                            feature_df: pd.DataFrame,
                            profile: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Thử dump model; nếu fail vì unpicklable lambda -> thu gọn
    model_path = out_dir / "model.pkl"
    try:
        joblib.dump(model, model_path)
    except Exception as e:
        # Thử fallback nếu model là dict
        fallback_path = out_dir / "model_light.pkl"
        print(f"[WARN] Không pickle được model đầy đủ: {e}")
        if isinstance(model, dict):
            safe_dict = {}
            for k, v in model.items():
                try:
                    joblib.dump(v, out_dir / f"_tmp_{k}.pkl")
                    safe_dict[k] = v
                except Exception:
                    print(f"[WARN] Bỏ qua key không pickle được: {k} ({type(v)})")
            joblib.dump(safe_dict, fallback_path)
            metadata["model_pickle_notice"] = "Original model had unpicklable parts (lambda). Saved reduced dict."
        else:
            raise

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if "customer_id" in feature_df.columns:
        cid = feature_df["customer_id"]
    else:
        # fallback: index
        cid = feature_df.index

    assign_df = pd.DataFrame({
        "customer_id": cid,
        "cluster": labels.values if hasattr(labels, "values") else labels
    })
    assign_df.to_csv(out_dir / "cluster_assignments.csv", index=False)

    profile.to_csv(out_dir / "profile.csv")

    return True

def load_clustering_model(model_dir: Path):
    model_path = model_dir / "model.pkl"
    light_path = model_dir / "model_light.pkl"
    meta_path = model_dir / "metadata.json"
    assign_path = model_dir / "cluster_assignments.csv"
    profile_path = model_dir / "profile.csv"

    if model_path.exists():
        model = joblib.load(model_path)
    elif light_path.exists():
        model = joblib.load(light_path)
    else:
        raise FileNotFoundError("Không tìm thấy model.pkl hoặc model_light.pkl")

    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    labels_df = pd.DataFrame()
    if assign_path.exists():
        labels_df = pd.read_csv(assign_path)
        if "customer_id" in labels_df.columns:
            labels_df = labels_df.set_index("customer_id")

    profile_df = pd.DataFrame()
    if profile_path.exists():
        profile_df = pd.read_csv(profile_path, index_col=0)

    return model, metadata, labels_df, profile_df
