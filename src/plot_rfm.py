"""
plot_rfm.py
Hàm vẽ cho RFM (matplotlib / plotly). Chỉ import bên trong hàm để tránh
lỗi khi môi trường thiếu thư viện đồ hoạ.
"""

from __future__ import annotations
import pandas as pd

def plot_histograms(rfm_df: pd.DataFrame,
                    cols = ("Recency","Frequency","Monetary"),
                    bins: int = 20):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 3*len(cols)))
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        ax.hist(rfm_df[c].dropna(), bins=bins, edgecolor="black")
        ax.set_title(f"Distribution of {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_rfm_treemap(level_summary: pd.DataFrame,
                     label_col: str = "RFM_Level",
                     count_col: str = "Count",
                     extra_cols = ("RecencyMean","FrequencyMean","MonetaryMean")):
    import matplotlib.pyplot as plt
    import squarify

    df = level_summary.copy()
    # Tạo nhãn tự động nếu chưa có
    if "LabelText" not in df.columns:
        def _mk(row):
            return (f"{row[label_col]}\n"
                    f"R:{int(row['RecencyMean'])} "
                    f"F:{int(row['FrequencyMean'])} "
                    f"M:{int(row['MonetaryMean'])}\n"
                    f"N={int(row[count_col])} ({row['Percent']}%)")
        if {"RecencyMean","FrequencyMean","MonetaryMean","Percent"}.issubset(df.columns):
            df["LabelText"] = df.apply(_mk, axis=1)
        else:
            df["LabelText"] = df[label_col]

    fig = plt.figure(figsize=(12,8))
    squarify.plot(
        sizes=df[count_col],
        label=df["LabelText"],
        alpha=0.9,
        pad=True
    )
    plt.title("Customer Segments Treemap", fontsize=18, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    return fig


def plot_elbow_silhouette(results_df: pd.DataFrame):
    """
    Dành cho KMeans sweep: results_df gồm cột k, inertia, silhouette
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(results_df["k"], results_df["inertia"], marker="o")
    ax[0].set_title("Elbow")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Inertia")
    ax[0].grid(alpha=.3)

    ax[1].plot(results_df["k"], results_df["silhouette"], marker="o", color="darkgreen")
    ax[1].set_title("Silhouette")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Score")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def plot_gmm_model_selection(gmm_df: pd.DataFrame):
    """
    gmm_df: index=k, cột BIC,AIC,ICL,Silhouette,min_pct
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13,5))

    ax[0].plot(gmm_df.index, gmm_df["BIC"], marker="o", label="BIC")
    ax[0].plot(gmm_df.index, gmm_df["ICL"], marker="o", label="ICL")
    ax[0].plot(gmm_df.index, gmm_df["AIC"], marker="o", label="AIC", alpha=.6)
    ax[0].set_title("Model selection (lower better)")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Score")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    ax2 = ax[1]
    ax2.plot(gmm_df.index, gmm_df["Silhouette"], marker="o", label="Silhouette")
    ax2.set_title("Silhouette vs k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette")
    ax2.grid(alpha=.3)
    ax2b = ax2.twinx()
    ax2b.plot(gmm_df.index, gmm_df["min_pct"], marker="s", ls=":", color="red", label="Min size %")
    ax2b.set_ylabel("Min size (%)")

    # Gộp legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, loc="best")

    fig.tight_layout()
    return fig


def plot_cluster_bubble(profile_df: pd.DataFrame,
                        cluster_col: str,
                        recency_mean_col: str = ("Recency","mean"),
                        frequency_mean_col: str = ("Frequency","mean"),
                        monetary_mean_col: str = ("Monetary","mean"),
                        size_col: str = "count",
                        size_mode: str = "frequency",
                        title: str = "Cluster Bubble Chart"):
    """
    profile_df: MultiIndex columns (Recency,mean)... + count
    size_mode: 'frequency' -> kích thước = FrequencyMean, 'count' -> = count
    """
    import plotly.express as px

    # Chuẩn hoá cột nếu MultiIndex (sau groupby agg)
    df = profile_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df_flat = {}
        for c in df.columns:
            if isinstance(c, tuple):
                df_flat["_".join([str(x) for x in c])] = df[c]
            else:
                df_flat[str(c)] = df[c]
        df = pd.DataFrame(df_flat)

    # Map tên
    r_col = "Recency_mean" if "Recency_mean" in df.columns else "Recency_mean"
    f_col = "Frequency_mean"
    m_col = "Monetary_mean"

    if size_mode == "count":
        size_use = size_col
    else:
        size_use = f_col

    df["ClusterName"] = df.index.map(lambda i: f"{cluster_col} {i}")

    fig = px.scatter(
        df,
        x=r_col,
        y=m_col,
        size=size_use,
        color="ClusterName",
        hover_data=[f_col, size_col],
        size_max=60,
        template="plotly_white",
        title=title
    )
    fig.update_layout(
        xaxis_title="Recency (mean, days – thấp là mới)",
        yaxis_title="Monetary (mean)"
    )
    return fig
