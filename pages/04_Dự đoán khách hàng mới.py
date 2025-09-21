import streamlit as st
import pandas as pd
import numpy as np
import html
from pathlib import Path

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="Dự đoán khách hàng mới", layout="wide")
st.title("Dự đoán khách hàng mới dựa trên các giá trị giả định")

# ============== CONSTANTS ==============
DATA_PATH = Path("data/orders_full.csv")
FEATURES = ["Recency","Frequency","Monetary"]
N_COMPONENTS = 5
RANDOM_STATE = 42

# ============== IMPORT MODULES ==============
try:
    from src.rfm_base import load_orders, build_rfm_snapshot
    from src.rfm_rule_scoring import compute_rfm_scores
    from src.rfm_labeling import apply_rfm_level
except Exception as e:
    st.error(f"Lỗi import module RFM: {e}")
    st.stop()

# ============== LOAD RFM BASE ==============
@st.cache_data
def load_rfm_base(csv_path: Path):
    raw = load_orders(csv_path)
    snap = build_rfm_snapshot(raw)
    scored = compute_rfm_scores(snap)
    labeled = apply_rfm_level(scored)
    return labeled

try:
    rfm_base = load_rfm_base(DATA_PATH)
except Exception as e:
    st.error(f"Lỗi xây RFM từ orders_full.csv: {e}")
    st.stop()

if rfm_base.empty:
    st.warning("Dataset RFM trống.")
    st.stop()

# ============== STYLE ==============
st.markdown("""
<style>
  .predict-box {
    background:#F2F8FF;
    border:1px solid #d6e6f5;
    border-radius:18px;
    padding:22px 24px;
    box-shadow:0 2px 5px rgba(0,0,0,0.04);
    font-family:system-ui,Roboto,Arial,sans-serif;
  }
  .predict-box h4 {
    margin:0 0 16px 0;
    font-size:18px;
    font-weight:600;
    color:#0d3d66;
  }
  .badges-row {
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    margin-bottom:10px;
    width:100%;
  }
  .badges-row.equal3 .badge {
    flex:1 1 calc(33.333% - 10px);
    min-width:0;
  }
  .badges-row.equal3 { align-items:stretch; }
  .badges-row.equal3 .badge {
    display:flex;
    align-items:center;
    justify-content:center;
  }
  .badge {
    background:#ffffff;
    padding:10px 14px;
    border-radius:12px;
    font-size:13.5px;
    font-weight:600;
    border:1px solid #e1e8f0;
    text-align:center;
    box-sizing:border-box;
  }
  .info-box {
    background:#FFF9DC;
    border:1px solid #E8D28A;
    border-radius:12px;
    padding:16px 18px;
    margin-top:16px;
    font-size:14.5px;
    line-height:1.55;
  }
  .info-box h5 {
    margin:0 0 14px 0;
    font-size:18px; /* = Kết quả dự đoán */
    font-weight:600;
    color:#0d3d66;
  }
  .info-row { margin:4px 0; }
  .info-row b {
    color:#0d3d66;
    font-weight:600;
  }
  .section-title {
    margin:22px 0 12px 0;
    font-weight:600;
    font-size:18px; /* = Kết quả dự đoán */
    color:#0d3d66;
  }
  ul.tactics { margin:4px 0 4px 20px; padding:0; }
  ul.tactics li {
    margin:4px 0;
    font-size:14.5px; /* = size Segment value (ví dụ STARS) */
    font-weight:500;
  }
  footer { margin-top:40px; color:#666; font-size:12.5px; text-align:left; }
</style>
""", unsafe_allow_html=True)

# ============== INTRO TEXT ==============
st.markdown("Dựa trên R-F-M và mô hình GMM (k=5) để đánh giá Segment và Cluster của khách hàng mới này")

# ============== HELPERS ==============
def classify_cluster(recency_mean, frequency_mean, monetary_mean,
                     r_med, f_med, m_med):
    r_cat = "Mới" if recency_mean <= r_med else "Lâu"
    f_cat = "F cao" if frequency_mean >= f_med else "F thấp"
    m_cat = "Chi tiêu cao" if monetary_mean >= m_med else "Chi tiêu thấp"
    if r_cat == "Mới" and f_cat == "F cao" and m_cat == "Chi tiêu cao":
        return "STARS","Upsell & giữ chân"
    elif r_cat == "Mới" and f_cat == "F cao":
        return "ACTIVE","Duy trì tần suất"
    elif r_cat == "Mới" and m_cat == "Chi tiêu cao":
        return "BIG SPENDER","Khuyến khích lặp lại"
    elif r_cat == "Mới":
        return "NEW","Onboarding"
    elif f_cat == "F cao" and m_cat == "Chi tiêu cao":
        return "LOYAL","Tăng CLV"
    elif f_cat == "F cao":
        return "REGULARS","Cross-sell"
    elif m_cat == "Chi tiêu cao":
        return "BIG SPENDER","Reactivation giá trị"
    else:
        return "LIGHT","Nuôi dưỡng"

def compute_cluster_profile(df_with_cluster: pd.DataFrame):
    prof = (
        df_with_cluster.groupby("cluster_gmm")
        .agg(
            RecencyMean=("Recency","mean"),
            FrequencyMean=("Frequency","mean"),
            MonetaryMean=("Monetary","mean"),
            Count=("customer_id","nunique")
        ).reset_index()
    )
    if prof.empty:
        return prof
    r_inv = (prof["RecencyMean"].max() - prof["RecencyMean"]) + 1e-9
    r_norm = (r_inv - r_inv.min())/(r_inv.max()-r_inv.min()+1e-9)
    f_norm = (prof["FrequencyMean"]-prof["FrequencyMean"].min())/(prof["FrequencyMean"].max()-prof["FrequencyMean"].min()+1e-9)
    m_norm = (prof["MonetaryMean"]-prof["MonetaryMean"].min())/(prof["MonetaryMean"].max()-prof["MonetaryMean"].min()+1e-9)
    prof["ValueScore"] = (r_norm + f_norm + m_norm)/3
    r_med = prof["RecencyMean"].median()
    f_med = prof["FrequencyMean"].median()
    m_med = prof["MonetaryMean"].median()
    labels, actions = [], []
    for _, row in prof.iterrows():
        lb, ac = classify_cluster(row["RecencyMean"], row["FrequencyMean"], row["MonetaryMean"],
                                  r_med, f_med, m_med)
        labels.append(lb); actions.append(ac)
    prof["Phân loại"] = labels
    prof["Hành động gợi ý"] = actions
    return prof

def generate_mapping_by_monetary_desc(df: pd.DataFrame,
                                      internal_col="internal_component",
                                      monetary_col="Monetary"):
    tmp = (
        df.groupby(internal_col, as_index=False)
          .agg(MonetaryMean=(monetary_col,"mean"))
          .sort_values("MonetaryMean", ascending=False)
          .reset_index(drop=True)
    )
    mapping = {}
    for public_id, row in tmp.iterrows():
        mapping[int(row[internal_col])] = int(public_id)
    return mapping

def refit_gmm(full_rfm_for_fit: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    train = full_rfm_for_fit[["customer_id"] + FEATURES].copy()
    scaler = StandardScaler()
    X = train[FEATURES].values.astype(float)
    Xs = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type="full",
                          random_state=RANDOM_STATE)
    gmm.fit(Xs)
    probs = gmm.predict_proba(Xs)
    internal = probs.argmax(axis=1)
    train["internal_component"] = internal
    mapping = generate_mapping_by_monetary_desc(train, "internal_component","Monetary")
    train["cluster_gmm"] = [mapping.get(ic, ic) for ic in internal]
    profile = compute_cluster_profile(train)
    return gmm, scaler, train, profile, probs

def score_new_customer_dynamic(base_rfm: pd.DataFrame, r, f, m):
    temp = base_rfm[["customer_id","Recency","Frequency","Monetary"]].copy()
    new_row = pd.DataFrame([{
        "customer_id":"__NEW__",
        "Recency": r,
        "Frequency": f,
        "Monetary": m
    }])
    joined = pd.concat([temp, new_row], ignore_index=True)
    scored = compute_rfm_scores(joined)
    labeled = apply_rfm_level(scored)
    return labeled[labeled["customer_id"]=="__NEW__"].iloc[-1]

STRATEGY_LIBRARY = {
    "LOST": [
        "Win-back voucher / quà sinh nhật",
        "Khảo sát lý do rời bỏ",
        "Flash sale tái kích hoạt"
    ],
    "REGULARS": [
        "Ưu đãi duy trì nhẹ",
        "Theo dõi nâng cấp sang LOYAL / BIG SPENDER",
        "Giữ trải nghiệm ổn định"
    ],
    "BIG SPENDER": [
        "CSKH ưu tiên / hotline riêng",
        "Gợi ý combo hoặc subscription",
        "Ưu đãi cá nhân hoá giữ chân"
    ],
    "STARS": [
        "Chăm sóc VIP / sự kiện riêng",
        "Upsell & cross-sell cao cấp",
        "Referral thưởng cao"
    ],
    "LIGHT": [
        "Combo nhỏ tăng giá trị đơn",
        "Content nuôi dưỡng / review",
        "Ưu đãi nhỏ nhưng đều"
    ],
    "ACTIVE": [
        "Ưu đãi kích hoạt (Mua 2 tặng 1)",
        "Remarketing email / push",
        "Upsell nhẹ sản phẩm liên quan"
    ],
    "LOYAL": [
        "Tích điểm / gamification",
        "Referral program",
        "Ưu tiên thử sản phẩm mới"
    ],
    "NEW": [
        "Email cảm ơn + voucher đơn 2",
        "Onboarding: sp phổ biến",
        "Nhắc quay lại trong 30 ngày"
    ],
    "OTHER": [
        "Theo dõi thêm hành vi",
        "Điều chỉnh tiêu chí phân nhóm",
        "Kiểm soát chi phí chăm sóc"
    ]
}

def get_strategy_list(segment_name: str):
    key = (segment_name or "OTHER").strip().upper()
    return STRATEGY_LIBRARY.get(key, STRATEGY_LIBRARY["OTHER"])

# Đổi Stale -> Older
def build_cluster_descriptor(r, f, m, r_med, f_med, m_med):
    rec_part = "Fresh" if r <= r_med else "Older"
    f_part = "HighFreq" if f >= f_med else "LowFreq"
    m_part = "HighValue" if m >= m_med else "LowValue"
    return " | ".join([rec_part, f_part, m_part])

# ============== DEFAULTS ==============
DEFAULT_RECENCY = 33
DEFAULT_FREQUENCY = 3
DEFAULT_MONETARY = 109

if "initialized" not in st.session_state:
    st.session_state.recency_val = DEFAULT_RECENCY
    st.session_state.frequency_val = DEFAULT_FREQUENCY
    st.session_state.monetary_val = DEFAULT_MONETARY
    st.session_state.initialized = True

# ============== SLIDERS ==============
recency_max = int(max(7, rfm_base["Recency"].max(), DEFAULT_RECENCY))
freq_max = int(max(5, rfm_base["Frequency"].max(), DEFAULT_FREQUENCY))
monetary_max = int(max(50, rfm_base["Monetary"].max(), DEFAULT_MONETARY))

c_left, c_right = st.columns([1,1])
with c_left:
    recency_val = st.slider("Recency (ngày từ lần mua gần nhất)", 1, recency_max, int(st.session_state.recency_val))
    frequency_val = st.slider("Frequency (số đơn)", 1, freq_max, int(st.session_state.frequency_val))
    monetary_val = st.slider("Monetary (tổng chi tiêu)", 1, monetary_max, int(st.session_state.monetary_val))
    run_btn = st.button("Tính toán và dự đoán", type="primary")

st.session_state.recency_val = recency_val
st.session_state.frequency_val = frequency_val
st.session_state.monetary_val = monetary_val

if "auto_run_done" not in st.session_state:
    run_btn = True
    st.session_state.auto_run_done = True

r_med_all = rfm_base["Recency"].median()
f_med_all = rfm_base["Frequency"].median()
m_med_all = rfm_base["Monetary"].median()

if run_btn:
    try:
        new_customer_row = pd.DataFrame([{
            "customer_id":"__NEW__",
            "Recency": recency_val,
            "Frequency": frequency_val,
            "Monetary": monetary_val
        }])
        fit_df = pd.concat([rfm_base[["customer_id","Recency","Frequency","Monetary"]], new_customer_row], ignore_index=True)
        gmm, scaler, train_with_clusters, profile, probs_all = refit_gmm(fit_df)

        new_row_cluster = train_with_clusters[train_with_clusters["customer_id"]=="__NEW__"].iloc[-1]
        public_cluster = int(new_row_cluster["cluster_gmm"])

        try:
            scored_new = score_new_customer_dynamic(rfm_base, recency_val, frequency_val, monetary_val)
            r_score = int(scored_new["R"]); f_score = int(scored_new["F"]); m_score = int(scored_new["M"])
            rfm_level = scored_new.get("RFM_Level","OTHER")
        except Exception:
            r_score=f_score=m_score="-"; rfm_level="OTHER"

        probs_new = probs_all[-1]
        confidence = float(np.max(probs_new))

        value_score_txt = "-"
        if not profile.empty and public_cluster in profile["cluster_gmm"].values:
            crow = profile[profile["cluster_gmm"]==public_cluster].iloc[0]
            value_score_txt = f"{crow['ValueScore']:.3f}"

        cluster_descriptor = build_cluster_descriptor(recency_val, frequency_val, monetary_val,
                                                      r_med_all, f_med_all, m_med_all)

        strategy_list = get_strategy_list(rfm_level)
        strategy_html = "".join(f"<li>{html.escape(s)}</li>" for s in strategy_list)

        badges_row1 = f"""
        <div class="badges-row equal3">
          <div class="badge">R: {r_score}</div>
          <div class="badge">F: {f_score}</div>
          <div class="badge">M: {m_score}</div>
        </div>
        """

        info_html = f"""
        <div class="info-box">
          <h5>Thông tin cụ thể</h5>
          <div class="info-row"><b>Segment:</b> {html.escape(str(rfm_level))}</div>
          <div class="info-row"><b>Cluster GMM:</b> {public_cluster}</div>
          <div class="info-row"><b>Cluster Desc:</b> {html.escape(cluster_descriptor)}</div>
          <div class="info-row"><b>Độ tin cậy:</b> {confidence*100:.1f}%</div>
          <div class="info-row"><b>Value Score:</b> {value_score_txt}</div>
        </div>
        """

        predict_html = f"""
        <div class="predict-box">
          <h4>Kết quả dự đoán</h4>
          {badges_row1}
          {info_html}
          <div class="section-title">Chiến lược gợi ý (Theo Segment)</div>
          <ul class="tactics">{strategy_html}</ul>
        </div>
        """

        with c_right:
            st.markdown(predict_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Lỗi tính toán / dự đoán: {e}")

st.markdown("<footer>© 2025 Đồ án tốt nghiệp lớp DL07_K306 - RFM Segmentation - Nhóm J</footer>", unsafe_allow_html=True)