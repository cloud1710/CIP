import streamlit as st
from pathlib import Path
import pandas as pd
import io
import numpy as np
import plotly.express as px
import html

from src.rfm_base import load_orders, build_rfm_snapshot
from src.rfm_rule_scoring import compute_rfm_scores
from src.rfm_labeling import apply_rfm_level

try:
    from src.cluster_profile import load_artifacts
except Exception as e:
    load_artifacts = None
    _cluster_import_error = str(e)

st.set_page_config(page_title="Phân cụm khách hàng - Segmentation", layout="wide")
st.title("🔀 Phân tích & Phân cụm khách hàng (k=5)")

DATA_PATH = Path("data/orders_full.csv")
GMM_DIR = Path("models/gmm/gmm_rfm_v1")
REQUIRED_BASE_COLS = {"customer_id", "order_id"}

BUBBLE_SHOW_TEXT = True
BUBBLE_ADD_REFERENCE_LINES = True
BUBBLE_REVERSE_RECENCY_AXIS = False
BUBBLE_SIZE_MAX = 60
BUBBLE_TITLE_PREFIX = "Trung bình các cụm RFM"

SHOW_RM_SCATTER = True
CUSTOMER_SCATTER_MAX_POINTS = 15000
CUSTOMER_SCATTER_RANDOM_STATE = 42
CUSTOMER_SCATTER_OPACITY = 0.55
SHOW_RM_MEDIAN_LINES = True

# Sidebar
with st.sidebar:
    st.header("⚙️ Tuỳ chọn dữ liệu")
    uploaded_file = st.file_uploader("Tải lên CSV", type=["csv"])
    c1, c2 = st.columns(2)
    with c1:
        reset_default = st.button("Dùng mặc định", type="secondary")
    with c2:
        show_guide = st.toggle("Hướng dẫn", value=False)
    if show_guide:
        st.markdown(
            "- File cần có cột: customer_id, order_id, order_date, giá trị đơn hàng.\n"
            "- Nếu tên cột khác chuẩn, cần chỉnh lại trong mã nguồn."
        )

@st.cache_data
def load_raw_default():
    return load_orders(DATA_PATH)

@st.cache_data
def build_rfm_full_default():
    raw = load_raw_default()
    rfm_base = build_rfm_snapshot(raw)
    rfm_scored = compute_rfm_scores(rfm_base)
    rfm_final = apply_rfm_level(rfm_scored)
    return rfm_final

@st.cache_data
def load_artifacts_cached(dir_path: Path):
    if load_artifacts is None:
        raise RuntimeError(f"Không import được cluster_profile: {_cluster_import_error}")
    return load_artifacts(dir_path)

def try_build_rfm_from_uploaded(bytes_data: bytes):
    try:
        df = pd.read_csv(io.BytesIO(bytes_data))
    except Exception as e:
        raise ValueError(f"Không đọc được CSV: {e}")
    missing = REQUIRED_BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}")
    try:
        snap = build_rfm_snapshot(df)
        scored = compute_rfm_scores(snap)
        final = apply_rfm_level(scored)
        return final
    except Exception as e:
        raise ValueError(f"Lỗi xử lý RFM trên file tải lên: {e}")

if reset_default and "uploaded_orders_bytes" in st.session_state:
    del st.session_state["uploaded_orders_bytes"]
if uploaded_file is not None:
    st.session_state["uploaded_orders_bytes"] = uploaded_file.getvalue()

uploaded_error = None
if "uploaded_orders_bytes" in st.session_state:
    try:
        rfm = try_build_rfm_from_uploaded(st.session_state["uploaded_orders_bytes"])
    except Exception as e:
        uploaded_error = str(e)
        rfm = build_rfm_full_default()
else:
    rfm = build_rfm_full_default()
if uploaded_error:
    st.warning(f"Lỗi file tải lên → dùng dữ liệu mặc định. Chi tiết: {uploaded_error}")

# Load GMM artifacts
labels_df = profile_df = meta = gmm_model = mapping = scaler = None
gmm_loaded_ok = False
try:
    if GMM_DIR.exists():
        gmm_model, scaler, labels_df, profile_df, meta, mapping = load_artifacts_cached(GMM_DIR)
        if labels_df.index.name != "customer_id" and "customer_id" in labels_df.columns:
            labels_df = labels_df.set_index("customer_id")
        if "customer_id" in rfm.columns:
            rfm = rfm.set_index("customer_id").join(labels_df, how="left").reset_index()
        else:
            rfm = rfm.join(labels_df, how="left")
        gmm_loaded_ok = True
    else:
        st.warning(f"Chưa có thư mục artifacts: {GMM_DIR}. Hãy huấn luyện trước.")
except Exception as e:
    st.warning(f"Không load được artifacts GMM: {e}")

# ---- Mapping hành động cụm cho tab Predict ----
def build_cluster_action_map(rfm_df: pd.DataFrame):
    if not gmm_loaded_ok or "cluster_gmm" not in rfm_df.columns:
        return {}
    agg_tmp = (
        rfm_df.groupby("cluster_gmm")
              .agg(
                  RecencyMean=("Recency","mean"),
                  FrequencyMean=("Frequency","mean"),
                  MonetaryMean=("Monetary","mean"),
              )
              .reset_index()
    )
    if agg_tmp.empty:
        return {}
    r_med_all = agg_tmp["RecencyMean"].median()
    f_med_all = agg_tmp["FrequencyMean"].median()
    m_med_all = agg_tmp["MonetaryMean"].median()
    def classify(row):
        r_cat = "Mới" if row["RecencyMean"] <= r_med_all else "Lâu"
        f_cat = "F cao" if row["FrequencyMean"] >= f_med_all else "F thấp"
        m_cat = "Chi tiêu cao" if row["MonetaryMean"] >= m_med_all else "Chi tiêu thấp"
        if r_cat == "Mới" and f_cat == "F cao" and m_cat == "Chi tiêu cao":
            return "VIP / Core","Giữ chân & mở rộng CLV"
        elif r_cat == "Mới" and f_cat == "F cao":
            return "Hoạt động tần suất cao","Tăng AOV qua bundle"
        elif r_cat == "Mới" and m_cat == "Chi tiêu cao":
            return "Big Ticket mới","Khuyến khích lặp lại sớm"
        elif r_cat == "Mới":
            return "Mới thử nghiệm","Onboarding + ưu đãi đơn 2"
        elif f_cat == "F cao" and m_cat == "Chi tiêu cao":
            return "Nguy cơ bỏ - giá trị cao","Win-back cá nhân hoá"
        elif f_cat == "F cao":
            return "F cao giá trị thấp","Cross-sell margin cao"
        elif m_cat == "Chi tiêu cao":
            return "Ngủ quên giá trị cao","Reactivation mạnh"
        else:
            return "Ngủ quên giá trị thấp","Re-engagement nhẹ"
    cluster_map = {}
    for _, rrow in agg_tmp.iterrows():
        _, action = classify(rrow)
        cluster_map[int(rrow["cluster_gmm"])] = action
    return cluster_map

cluster_action_map = build_cluster_action_map(rfm)

def big(n):
    try:
        return f"{n:,.0f}"
    except:
        return n

def build_distribution(df: pd.DataFrame, col: str):
    dist = df[col].value_counts(dropna=False).to_frame("Count")
    dist["Percent"] = (dist["Count"] / dist["Count"].sum() * 100).round(2)
    return dist

def aggregate_rfm_level_for_treemap(df: pd.DataFrame,
                                    monetary_col: str = "Monetary") -> pd.DataFrame:
    work = df.copy()
    if "RFM_Level" not in work.columns:
        work = apply_rfm_level(work)
    agg_dict = {"Rows": ("RFM_Level","size")}
    agg_dict["Count"] = ("customer_id","nunique") if "customer_id" in work.columns else ("RFM_Level","size")
    if monetary_col in work.columns:
        agg_dict["Monetary_Sum"] = (monetary_col,"sum")
        agg_dict["Monetary_Avg"] = (monetary_col,"mean")
    agg = work.groupby("RFM_Level", as_index=False).agg(**agg_dict)
    total_rows = agg["Rows"].sum()
    agg["Share_%"] = agg["Rows"] / total_rows * 100
    for col in ["Monetary_Sum","Monetary_Avg"]:
        if col not in agg.columns:
            agg[col] = 0
    return agg.sort_values("Rows", ascending=False)

DEFAULT_SEGMENT_COLORS = {
    "STARS": "#1b7837","BIG SPENDER": "#00429d","LOYAL": "#73a2c6","ACTIVE": "#4daf4a",
    "NEW": "#ffcc00","LIGHT": "#f29e4c","REGULARS": "#9e9e9e","LOST": "#d73027","OTHER": "#cccccc"
}
SEGMENT_DISPLAY_COLORS = {
    "STARS":"#1b7837","BIG SPENDER":"#00429d","LOYAL":"#73a2c6",
    "ACTIVE":"#4daf4a","NEW":"#ffcc00","LIGHT":"#f29e4c",
    "REGULARS":"#9e9e9e","LOST":"#d73027","OTHER":"#607d8b"
}
SEGMENT_STRATEGY_MAP = {
    "LOST": ["Win-back voucher / ưu đãi", "Khảo sát lý do rời bỏ", "Flash sale tái kích hoạt"],
    "REGULARS": ["Ưu đãi duy trì nhẹ", "Theo dõi nâng cấp", "Giữ trải nghiệm ổn định"],
    "BIG SPENDER": ["CSKH ưu tiên", "Gợi ý combo/subscription", "Ưu đãi cá nhân hoá"],
    "STARS": ["Chăm sóc VIP", "Upsell cao cấp", "Referral thưởng cao"],
    "LIGHT": ["Combo nhỏ tăng AOV", "Content nuôi dưỡng", "Ưu đãi nhỏ nhưng đều"],
    "ACTIVE": ["Ưu đãi kích hoạt", "Remarketing đa kênh", "Upsell nhẹ"],
    "LOYAL": ["Tích điểm / gamification", "Referral program", "Ưu tiên thử sản phẩm mới"],
    "NEW": ["Email chào mừng + voucher", "Onboarding sản phẩm chủ lực", "Nhắc quay lại sớm"],
    "OTHER": ["Theo dõi thêm hành vi", "Điều chỉnh tiêu chí phân nhóm"]
}

def plot_rfm_treemap_fixed(agg_df: pd.DataFrame,
                           title: str = "Treemap phân bố nhóm RFM",
                           base_font_boost: int = 2,
                           clamp_size: bool = True):
    import squarify, matplotlib.pyplot as plt, matplotlib as mpl
    df = agg_df.copy()
    df = df[df["Count"] > 0]
    if df.empty:
        raise ValueError("Không có dữ liệu treemap.")
    sizes = df["Count"].astype(float).tolist()
    normed = squarify.normalize_sizes(sizes, 100, 60)
    rects = squarify.squarify(normed, 0, 0, 100, 60)
    fig_w, fig_h, dpi_val = 9.2, 5.6, 110
    if clamp_size:
        fig_w = min(fig_w, 12); fig_h = min(fig_h, 8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi_val)
    ax.set_xlim(0,100); ax.set_ylim(0,60); ax.axis("off")
    def hex_to_rgb(h): h = h.replace("#",""); return tuple(int(h[i:i+2],16) for i in (0,2,4))
    def luminance(rgb):
        r,g,b = [v/255 for v in rgb]
        def ch(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
        rl,gl,bl = ch(r), ch(g), ch(b)
        return 0.2126*rl + 0.7152*gl + 0.0722*bl
    for i, rect in enumerate(rects):
        seg = df.iloc[i]["RFM_Level"]
        share = df.iloc[i]["Share_%"]
        color_hex = DEFAULT_SEGMENT_COLORS.get(seg, "#777777")
        rgb = hex_to_rgb(color_hex)
        text_color = "black" if luminance(rgb) > 0.6 else "white"
        x,y,dx,dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        ax.add_patch(mpl.patches.Rectangle((x,y), dx, dy,
                                           facecolor=color_hex, edgecolor="white", linewidth=1))
        area = dx*dy
        if area < 45: fz_raw = 8
        elif area < 80: fz_raw = 9 + base_font_boost - 1
        elif area < 140: fz_raw = 10 + base_font_boost - 1
        elif area < 250: fz_raw = 11 + base_font_boost
        elif area < 400: fz_raw = 12 + base_font_boost
        else: fz_raw = 14 + base_font_boost
        if area < 45 and seg != "NEW":
            continue
        if seg in {"LOST","REGULARS","BIG SPENDER"}:
            fz = fz_raw
        elif seg in {"LIGHT","STARS"}:
            fz = max(6, round(fz_raw*(2/3)))
        else:
            fz = max(6, round(fz_raw*0.5))
        ax.text(x+dx/2, y+dy/2, f"{seg}\n{share:.1f}%", ha="center", va="center",
                color=text_color, fontsize=fz, fontweight="bold", linespacing=0.9)
    ax.set_title(title, fontweight="bold", fontsize=17)
    ax.text(0.0, -0.05, "Kích thước = Số lượng khách hàng", ha="left", va="center",
            transform=ax.transAxes, fontsize=8, color="#555")
    return fig

tab_rule, tab_cluster, tab_predict = st.tabs([
    "📊 Tập luật khách hàng",
    "🧩 Phân cụm khách hàng",
    "🧮 Dự đoán khách hàng mới"
])

# TAB RULE
with tab_rule:
    st.subheader("Tổng quan RFM")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Số khách hàng", big(rfm["customer_id"].nunique()))
    with c2: st.metric("Tổng Monetary", big(rfm["Monetary"].sum()))
    with c3: st.metric("Recency (trung vị)", big(rfm["Recency"].median()))
    with c4: st.metric("Frequency (trung vị)", big(rfm["Frequency"].median()))

    st.markdown("### Phân phối RFM theo cấp độ")
    if "RFM_Level" in rfm.columns:
        dist = build_distribution(rfm, "RFM_Level").rename(columns={"Count":"Số lượng","Percent":"Tỷ lệ (%)"})
        st.dataframe(dist, use_container_width=True)
    else:
        st.warning("Thiếu trường RFM_Level.")

    st.markdown("### Top khách hàng (Monetary cao)")
    topN = st.slider("Số khách hiển thị", 5, 50, 10)
    st.dataframe(rfm.sort_values("Monetary", ascending=False).head(topN), use_container_width=True)

    st.markdown("### Treemap nhóm RFM")
    agg_rfm = aggregate_rfm_level_for_treemap(rfm) if "RFM_Level" in rfm.columns else pd.DataFrame()
    if agg_rfm.empty:
        st.info("Không có dữ liệu để hiển thị treemap.")
    else:
        try:
            fig = plot_rfm_treemap_fixed(agg_rfm)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi vẽ treemap: {e}")

    st.markdown("### Gợi ý chiến lược theo Segment (tham khảo)")
    strat_df = pd.DataFrame([(k, "; ".join(v)) for k,v in SEGMENT_STRATEGY_MAP.items()],
                            columns=["Segment","Chiến thuật gợi ý"])
    st.dataframe(strat_df, use_container_width=True)

# TAB CLUSTER
with tab_cluster:
    st.subheader("Phân cụm khách hàng (GMM k=5)")
    if not gmm_loaded_ok:
        st.warning(
            "Chưa có hoặc chưa load được nhãn GMM.\n\n"
            "Chạy lệnh:\n"
            "python -m src.cluster_profile fit --orders data/orders_full.csv\n\n"
            f"Artifacts: {GMM_DIR}"
        )
    else:
        if "cluster_gmm" not in rfm.columns:
            st.error("Không có cột cluster_gmm sau khi join labels.")
        else:
            agg = (
                rfm.groupby("cluster_gmm")
                   .agg(
                       RecencyMean=("Recency","mean"),
                       FrequencyMean=("Frequency","mean"),
                       MonetaryMean=("Monetary","mean"),
                       Count=("customer_id","size")
                   )
                   .reset_index()
            )
            agg = agg.sort_values("cluster_gmm")
            agg["Cluster"] = agg["cluster_gmm"].apply(lambda v: f"Cụm {int(v)}")
            cluster_order = [f"Cụm {i}" for i in sorted(agg['cluster_gmm'].unique())]
            agg["BubbleSize"] = agg["FrequencyMean"].clip(lower=0.0001)

            # Phân loại + hành động (đã dùng để tạo bảng & mapping)
            r_med_all = agg["RecencyMean"].median()
            f_med_all = agg["FrequencyMean"].median()
            m_med_all = agg["MonetaryMean"].median()
            def classify(row):
                r_cat = "Mới" if row["RecencyMean"] <= r_med_all else "Lâu"
                f_cat = "F cao" if row["FrequencyMean"] >= f_med_all else "F thấp"
                m_cat = "Chi tiêu cao" if row["MonetaryMean"] >= m_med_all else "Chi tiêu thấp"
                if r_cat == "Mới" and f_cat == "F cao" and m_cat == "Chi tiêu cao":
                    return "VIP / Core","Giữ chân & mở rộng CLV"
                elif r_cat == "Mới" and f_cat == "F cao":
                    return "Hoạt động tần suất cao","Tăng AOV qua bundle"
                elif r_cat == "Mới" and m_cat == "Chi tiêu cao":
                    return "Big Ticket mới","Khuyến khích lặp lại sớm"
                elif r_cat == "Mới":
                    return "Mới thử nghiệm","Onboarding + ưu đãi đơn 2"
                elif f_cat == "F cao" and m_cat == "Chi tiêu cao":
                    return "Nguy cơ bỏ - giá trị cao","Win-back cá nhân hoá"
                elif f_cat == "F cao":
                    return "F cao giá trị thấp","Cross-sell margin cao"
                elif m_cat == "Chi tiêu cao":
                    return "Ngủ quên giá trị cao","Reactivation mạnh"
                else:
                    return "Ngủ quên giá trị thấp","Re-engagement nhẹ"
            classes, actions = [], []
            for _, rowc in agg.iterrows():
                c,a = classify(rowc); classes.append(c); actions.append(a)
            agg["Phân loại"] = classes
            agg["Hành động gợi ý"] = actions

            if profile_df is not None:
                prof_cols = []
                if "cluster_marketing_name" in profile_df.columns: prof_cols.append("cluster_marketing_name")
                if "cluster_label_desc" in profile_df.columns: prof_cols.append("cluster_label_desc")
                prof_show = profile_df.copy()
                prof_show = prof_show[prof_cols] if prof_cols else None
                if prof_show is not None:
                    prof_show = prof_show.reset_index().rename(columns={"index":"cluster_gmm"})
                    agg = agg.merge(prof_show, on="cluster_gmm", how="left")

            fig_cluster = px.scatter(
                agg, x="RecencyMean", y="MonetaryMean",
                size="BubbleSize", color="Cluster",
                category_orders={"Cluster": cluster_order},
                hover_name="Cluster",
                hover_data={
                    "RecencyMean":":.1f","MonetaryMean":":.1f","FrequencyMean":":.2f",
                    "Count": True,"BubbleSize": False,"Phân loại": True
                    # ĐÃ GỠ 'ValueScore' để tránh lỗi vì không tồn tại cột
                },
                size_max=BUBBLE_SIZE_MAX,
                template="plotly_white",
                title=f"{BUBBLE_TITLE_PREFIX} (k=5 - GMM)"
            )
            fig_cluster.update_layout(
                xaxis_title="Recency (trung bình, ngày – thấp = mới hơn)",
                yaxis_title="Monetary (trung bình)"
            )
            if BUBBLE_REVERSE_RECENCY_AXIS:
                fig_cluster.update_layout(xaxis=dict(autorange="reversed"))
            if BUBBLE_ADD_REFERENCE_LINES:
                r_med = agg["RecencyMean"].median()
                m_med = agg["MonetaryMean"].median()
                fig_cluster.add_vline(x=r_med, line_dash="dot", line_color="#888",
                                      annotation_text="Recency trung vị", annotation_position="top")
                fig_cluster.add_hline(y=m_med, line_dash="dot", line_color="#888",
                                      annotation_text="Monetary trung vị", annotation_position="bottom right")
            if BUBBLE_SHOW_TEXT:
                fig_cluster.update_traces(text=agg["Cluster"], textposition="top center",
                                          marker_line_width=1, marker_line_color="#444")
            st.plotly_chart(fig_cluster, use_container_width=True)

            if SHOW_RM_SCATTER:
                st.markdown("### Phân bố khách hàng (Recency ↔ Monetary)")
                cust_df = rfm.dropna(subset=["Recency","Monetary","cluster_gmm"]).copy()
                total_pts = len(cust_df)
                sampled_flag = False
                if total_pts > CUSTOMER_SCATTER_MAX_POINTS:
                    cust_df = cust_df.sample(CUSTOMER_SCATTER_MAX_POINTS,
                                             random_state=CUSTOMER_SCATTER_RANDOM_STATE)
                    sampled_flag = True
                cust_df["Cluster"] = cust_df["cluster_gmm"].apply(lambda v: f"Cụm {int(v)}")
                color_map = {}
                for tr in fig_cluster.data:
                    name = getattr(tr, "name", None)
                    if name and name.startswith("Cụm"):
                        mk_color = getattr(tr.marker, "color", None)
                        if isinstance(mk_color, str):
                            color_map[name] = mk_color
                scatter_title = "Phân bố khách hàng: Recency (X) vs Monetary (Y)"
                if sampled_flag:
                    scatter_title += f" (Lấy mẫu {len(cust_df)}/{total_pts})"
                fig_rm = px.scatter(
                    cust_df, x="Recency", y="Monetary",
                    color="Cluster",
                    category_orders={"Cluster": cluster_order},
                    color_discrete_map=color_map if color_map else None,
                    opacity=CUSTOMER_SCATTER_OPACITY,
                    template="plotly_white",
                    hover_data={"customer_id": True,"Recency":":.1f","Monetary":":.2f","cluster_gmm": True},
                    title=scatter_title
                )
                if SHOW_RM_MEDIAN_LINES and not cust_df.empty:
                    median_r = cust_df["Recency"].median()
                    median_m = cust_df["Monetary"].median()
                    fig_rm.add_vline(x=median_r, line_dash="dot", line_color="#666",
                                     annotation_text="Recency trung vị", annotation_position="top")
                    fig_rm.add_hline(y=median_m, line_dash="dot", line_color="#666",
                                     annotation_text="Monetary trung vị", annotation_position="bottom right")
                fig_rm.update_layout(
                    xaxis_title="Recency (ngày từ lần mua gần nhất)",
                    yaxis_title="Monetary (tổng chi tiêu)",
                    legend_title="Cụm",
                    margin=dict(l=20,r=20,t=60,b=40)
                )
                if BUBBLE_REVERSE_RECENCY_AXIS:
                    fig_rm.update_layout(xaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_rm, use_container_width=True)

            st.markdown("### Định nghĩa & phân tích cụm (GMM k=5)")
            display_cols = [
                "Cluster","Count","RecencyMean","FrequencyMean","MonetaryMean",
                "Phân loại","Hành động gợi ý"
            ]
            extra_inserted = False
            if "cluster_marketing_name" in agg.columns:
                display_cols.insert(1,"cluster_marketing_name"); extra_inserted = True
            elif "cluster_label_desc" in agg.columns:
                display_cols.insert(1,"cluster_label_desc"); extra_inserted = True

            cluster_strat_df = agg[display_cols].copy()
            for c_round, dec in [("RecencyMean",1),("FrequencyMean",2),("MonetaryMean",1)]:
                if c_round in cluster_strat_df.columns:
                    cluster_strat_df[c_round] = cluster_strat_df[c_round].round(dec)
            rename_cols = {
                "Count":"Số lượng KH","RecencyMean":"Recency TB",
                "FrequencyMean":"Frequency TB","MonetaryMean":"Monetary TB",
                "Hành động gợi ý":"Hành động gợi ý"
            }
            if extra_inserted:
                if "cluster_marketing_name" in cluster_strat_df.columns:
                    rename_cols["cluster_marketing_name"] = "Tên marketing"
                if "cluster_label_desc" in cluster_strat_df.columns:
                    rename_cols["cluster_label_desc"] = "Mô tả cụm"
            cluster_show = cluster_strat_df.rename(columns=rename_cols)
            st.dataframe(cluster_show, use_container_width=True)

# TAB PREDICT
with tab_predict:
    st.subheader("🧮 Dự đoán khách hàng mới (RFM + Cluster)")
    st.markdown("Điều chỉnh các slider để xem R,F,M, Segment và dự đoán Cluster GMM.")

    if "predict_initialized" not in st.session_state:
        sample_row = rfm.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]
        st.session_state.recency_val = int(sample_row["Recency"])
        st.session_state.frequency_val = int(sample_row["Frequency"])
        st.session_state.monetary_val = int(sample_row["Monetary"])
        st.session_state.predict_initialized = True

    recency_max = int(max(7, rfm["Recency"].max()))
    freq_max = int(max(5, rfm["Frequency"].max()))
    monetary_max = int(max(50, rfm["Monetary"].max()))

    col_left, col_right = st.columns([1,1])
    with col_left:
        recency_val = st.slider("Recency (ngày từ lần mua gần nhất)", 1, recency_max, int(st.session_state.recency_val))
        frequency_val = st.slider("Frequency (số đơn)", 1, freq_max, int(st.session_state.frequency_val))
        monetary_val = st.slider("Monetary (tổng chi tiêu)", 1, monetary_max, int(st.session_state.monetary_val))

    st.session_state.recency_val = recency_val
    st.session_state.frequency_val = frequency_val
    st.session_state.monetary_val = monetary_val

    st.markdown("""
    <style>
      .result-box {
        background:#E7F3FF;
        border:1px solid #d0e3f5;
        border-radius:16px;
        padding:22px 24px 20px 24px;
        margin-top:0;
        box-shadow:0 2px 6px rgba(0,0,0,0.06);
        min-height:360px;
      }
      .result-box h4 {
        margin:0 0 14px 0;
        font-size:18px;
        font-weight:700;
        color:#0d3d66;
      }
      .segment-badge {
        font-weight:800;
        text-transform:uppercase;
        font-size:24px;
        letter-spacing:.6px;
        margin:4px 0 14px 0;
        display:inline-block;
      }
      .score-badges {
        display:flex;
        gap:10px;
        flex-wrap:wrap;
        margin:4px 0 18px 0;
      }
      .score-item {
        background:#ffffff;
        padding:10px 16px;
        border-radius:12px;
        font-size:13.5px;
        font-weight:600;
        min-width:110px;
        text-align:center;
        border:1px solid #e2e8f0;
      }
      .kv {
        display:grid;
        grid-template-columns:150px 1fr;
        row-gap:8px;
        column-gap:14px;
        font-size:14px;
        margin:0 0 14px 0;
      }
      .kv div.key { font-weight:600; color:#0f3554; }
      .tactics-list { margin:4px 0 2px 20px; padding:0; }
      .tactics-list li { margin:5px 0; font-size:14px; }
    </style>
    """, unsafe_allow_html=True)

    def score_new_customer(base_rfm: pd.DataFrame,
                           recency: float,
                           frequency: float,
                           monetary: float):
        required = ["customer_id","Recency","Frequency","Monetary"]
        for c in required:
            if c not in base_rfm.columns:
                raise ValueError(f"Thiếu cột {c} trong RFM hiện tại.")
        temp = base_rfm[required].copy()
        new_row = pd.DataFrame([{
            "customer_id": "__NEW__",
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary
        }])
        combined = pd.concat([temp, new_row], ignore_index=True)
        scored_all = compute_rfm_scores(combined)
        labeled_all = apply_rfm_level(scored_all)
        new_scored = labeled_all[labeled_all["customer_id"] == "__NEW__"].copy()
        if new_scored.empty:
            raise RuntimeError("Không tìm thấy bản ghi khách mới sau khi tính.")
        return new_scored.iloc[-1]

    def predict_cluster_gmm(recency, frequency, monetary):
        if not gmm_loaded_ok or gmm_model is None:
            return None
        try:
            feats = np.array([[recency, frequency, monetary]], dtype=float)
            feats_scaled = scaler.transform(feats) if scaler is not None else feats
            probs = gmm_model.predict_proba(feats_scaled)[0]
            cluster_id = int(np.argmax(probs))
            confidence = float(np.max(probs))
            return {"cluster_id": cluster_id, "confidence": confidence}
        except Exception as e:
            return {"error": str(e)}

    def describe_cluster(cid: int):
        if profile_df is not None:
            if cid in getattr(profile_df, "index", []):
                row = profile_df.loc[cid]
                parts = []
                if "cluster_marketing_name" in row: parts.append(f"{row['cluster_marketing_name']}")
                if "cluster_label_desc" in row: parts.append(f"{row['cluster_label_desc']}")
                if parts: return " - ".join(parts)
            if profile_df is not None and "cluster" in profile_df.columns:
                tmp = profile_df[profile_df["cluster"] == cid]
                if not tmp.empty:
                    row = tmp.iloc[0]
                    parts = []
                    if "cluster_marketing_name" in row: parts.append(f"{row['cluster_marketing_name']}")
                    if "cluster_label_desc" in row: parts.append(f"{row['cluster_label_desc']}")
                    if parts: return " - ".join(parts)
        return f"Cụm {cid}"

    try:
        row = score_new_customer(rfm,
                                 st.session_state.recency_val,
                                 st.session_state.frequency_val,
                                 st.session_state.monetary_val)
    except Exception as e:
        with col_right:
            st.error(f"Lỗi tính RFM cho khách mới: {e}")
        row = None

    if row is not None:
        r_score = int(row["R"])
        f_score = int(row["F"])
        m_score = int(row["M"])
        rfm_level = row.get("RFM_Level","OTHER")
        rfm_segment = row.get("RFM_Segment", f"{r_score}{f_score}{m_score}")

        cluster_pred = predict_cluster_gmm(
            st.session_state.recency_val,
            st.session_state.frequency_val,
            st.session_state.monetary_val
        )
        cluster_id = None
        cluster_desc = "Không có model GMM"
        confidence_pct = "-"
        if cluster_pred is None:
            pass
        elif "error" in cluster_pred:
            cluster_desc = f"Lỗi dự đoán: {cluster_pred['error']}"
        else:
            cluster_id = cluster_pred["cluster_id"]
            cluster_desc = describe_cluster(cluster_id)
            confidence_pct = f"{cluster_pred['confidence']*100:.1f}%"

        segment_tactics = SEGMENT_STRATEGY_MAP.get(rfm_level, ["(Chưa có chiến lược)"])
        segment_tactics_html = "".join(f"<li>{html.escape(t)}</li>" for t in segment_tactics)

        cluster_action = cluster_action_map.get(cluster_id) if cluster_id is not None else None
        cluster_tactics_html = f"<li>{html.escape(cluster_action)}</li>" if cluster_action else "<li>(Không có gợi ý)</li>"

        seg_color = SEGMENT_DISPLAY_COLORS.get(rfm_level, "#607d8b")

        result_html = f"""
        <div class='result-box'>
          <h4>Kết quả phân tích khách hàng mới</h4>
          <div class='segment-badge' style='color:{seg_color};'>{html.escape(rfm_level)}</div>
          <div class='score-badges'>
            <div class='score-item'>R: {r_score}</div>
            <div class='score-item'>F: {f_score}</div>
            <div class='score-item'>M: {m_score}</div>
            <div class='score-item'>Segment: {html.escape(rfm_segment)}</div>
          </div>
          <div class='kv'>
            <div class='key'>Cluster GMM</div><div>{'-' if cluster_id is None else cluster_id}</div>
            <div class='key'>Mô tả cluster</div><div>{html.escape(cluster_desc)}</div>
            <div class='key'>Độ tin cậy</div><div>{confidence_pct}</div>
          </div>
          <div style='margin:4px 0 6px 0; font-weight:600;'>Chiến thuật đề xuất (theo Segment)</div>
          <ul class='tactics-list'>
            {segment_tactics_html}
          </ul>
          <div style='margin:10px 0 6px 0; font-weight:600;'>Chiến thuật đề xuất (theo Cluster GMM)</div>
          <ul class='tactics-list'>
            {cluster_tactics_html}
          </ul>
        </div>
        """
        with col_right:
            st.markdown(result_html, unsafe_allow_html=True)

# Footer
st.markdown(
    "<div style='text-align:left; color:#666; font-size:13px; margin-top:30px;'>© 2025 Đồ án tốt nghiệp lớp DL07_K306 - RFM Segmentation - Nhóm J</div>",
    unsafe_allow_html=True
)
