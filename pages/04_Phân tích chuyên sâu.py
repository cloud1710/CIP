import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.rfm_base import load_orders, build_rfm_snapshot
from src.rfm_rule_scoring import compute_rfm_scores
from src.rfm_labeling import apply_rfm_level

# --- TÍCH HỢP CLUSTER PROFILE MỚI ---
try:
    from src.cluster_profile import load_artifacts  # (model, scaler, labels_df, profile_df, meta, mapping)
except Exception as e:
    load_artifacts = None
    _cluster_import_error = str(e)

try:
    from src.recommendation import recommend_actions
except Exception:
    def recommend_actions(rfm_level: str, cluster=None, monetary=None):
        return {"goal": "N/A", "tactics": [], "notes": []}

st.set_page_config(page_title="04 - Customer", layout="wide")
st.title("👤 Phân tích Khách hàng chuyên sâu")

DATA_PATH = Path("data/orders_full.csv")
GMM_DIR = Path("models/gmm/gmm_rfm_v1")

# ================== SEGMENT KNOWLEDGE BASE ==================
segment_catalog = {
    "LOST": {
        "definition": "Khách hàng lâu không quay lại, Recency cao vượt ngưỡng.",
        "base_goal": "Tái kích hoạt hoặc xác định lý do rời bỏ.",
        "base_strategies": [
            "Win-back voucher / quà sinh nhật",
            "Flash sale tái kích hoạt",
            "Khảo sát lý do rời bỏ"
        ],
        "kpi_focus": ["Reactivation Rate", "Open Rate", "Return Purchase"],
        "upgrade_path": "Chuyển thành ACTIVE rồi REGULARS / LOYAL",
        "risk_signals": ["Recency cao", "Frequency giảm", "Không phản hồi chiến dịch"]
    },
    "REGULARS": {
        "definition": "Khách mua đều đặn, hành vi ổn định.",
        "base_goal": "Duy trì tần suất và tăng giá trị đơn hàng.",
        "base_strategies": [
            "Ưu đãi duy trì nhẹ",
            "Theo dõi nâng cấp lên LOYAL / BIG SPENDER",
            "Tối ưu trải nghiệm"
        ],
        "kpi_focus": ["Repeat Rate", "AOV", "Frequency"],
        "upgrade_path": "Nâng lên LOYAL hoặc BIG SPENDER",
        "risk_signals": ["Tần suất giảm tuần/tháng", "Giảm giá trị đơn"]
    },
    "BIG SPENDER": {
        "definition": "Chi tiêu lớn, giá trị cao trong kỳ.",
        "base_goal": "Gia tăng vòng đời và giảm rủi ro giảm chi tiêu.",
        "base_strategies": [
            "CSKH ưu tiên / hotline riêng",
            "Gợi ý combo / subscription",
            "Ưu đãi cá nhân hoá giữ chân"
        ],
        "kpi_focus": ["CLV", "AOV", "Retention"],
        "upgrade_path": "Kết hợp thành STARS (nếu tần suất tăng)",
        "risk_signals": ["Giảm mạnh AOV", "Khoảng cách mua kéo dài"]
    },
    "STARS": {
        "definition": "Tần suất cao & chi tiêu cao – nhóm lõi rất giá trị.",
        "base_goal": "Khóa chặt trung thành và khai thác referral.",
        "base_strategies": [
            "Chăm sóc VIP / event độc quyền",
            "Upsell & cross-sell cao cấp",
            "Referral thưởng cao"
        ],
        "kpi_focus": ["Referral Rate", "CLV", "Upsell Rate"],
        "upgrade_path": "Duy trì đỉnh giá trị, giảm rủi ro bão hòa",
        "risk_signals": ["Giảm tần suất bất thường", "Không phản hồi ưu đãi cao cấp"]
    },
    "LIGHT": {
        "definition": "Mua thưa và chi tiêu thấp.",
        "base_goal": "Tăng tần suất và giá trị giỏ.",
        "base_strategies": [
            "Combo nhỏ tăng giá trị đơn",
            "Content nuôi dưỡng & review",
            "Ưu đãi nhỏ nhưng đều"
        ],
        "kpi_focus": ["Frequency", "AOV", "Activation"],
        "upgrade_path": "Nâng thành REGULARS rồi LOYAL",
        "risk_signals": ["Không quay lại sau chiến dịch", "AOV không tăng"]
    },
    "ACTIVE": {
        "definition": "Vừa quay lại gần đây, tần suất còn thấp.",
        "base_goal": "Khóa nhịp mua lặp lại 2–3 lần liên tiếp.",
        "base_strategies": [
            "Ưu đãi kích hoạt (Mua 2 tặng 1)",
            "Remarketing email / push",
            "Upsell nhẹ sản phẩm liên quan"
        ],
        "kpi_focus": ["Second Purchase Rate", "Repeat Cycle Time"],
        "upgrade_path": "Đẩy thành REGULARS hoặc LOYAL",
        "risk_signals": ["Không có đơn thứ 2 trong 30 ngày"]
    },
    "LOYAL": {
        "definition": "Trung thành, mua lặp lại ổn định.",
        "base_goal": "Duy trì và tăng CLV / AOV.",
        "base_strategies": [
            "Tích điểm / gamification",
            "Referral program",
            "Ưu tiên thử sản phẩm mới"
        ],
        "kpi_focus": ["Retention", "Referral", "CLV"],
        "upgrade_path": "Phát triển thành STARS / BIG SPENDER",
        "risk_signals": ["Giảm tần suất dần", "Không dùng điểm thưởng"]
    },
    "NEW": {
        "definition": "Khách hàng mới – mua lần đầu.",
        "base_goal": "Kích hoạt mua lần thứ 2 nhanh.",
        "base_strategies": [
            "Email cảm ơn + voucher đơn 2",
            "Onboarding giới thiệu sản phẩm bán chạy",
            "Nhắc quay lại trong 30 ngày"
        ],
        "kpi_focus": ["Second Purchase Rate", "Onboarding Completion"],
        "upgrade_path": "ACTIVE rồi REGULARS",
        "risk_signals": ["Không mua lại <30 ngày", "Không mở email onboarding"]
    },
    "OTHER": {
        "definition": "Nhóm nhỏ / chưa rõ đặc trưng.",
        "base_goal": "Thu thập thêm dữ liệu hành vi.",
        "base_strategies": [
            "Theo dõi thêm hành vi",
            "Điều chỉnh tiêu chí phân nhóm",
            "Kiểm soát chi phí chăm sóc"
        ],
        "kpi_focus": ["Data Completeness"],
        "upgrade_path": "Phân bổ lại sang nhóm chính",
        "risk_signals": ["Khối lượng thấp", "Nhiễu nhãn"]
    }
}

# ================== CACHE PIPELINE ==================
@st.cache_data
def build_rfm():
    raw = load_orders(DATA_PATH)
    snap = build_rfm_snapshot(raw)
    scored = compute_rfm_scores(snap)
    final = apply_rfm_level(scored)
    return raw, final

@st.cache_data
def load_artifacts_cached(dir_path: Path):
    if load_artifacts is None:
        raise RuntimeError(f"Không import được cluster_profile: {_cluster_import_error}")
    return load_artifacts(dir_path)  # (model, scaler, labels_df, profile_df, meta, mapping)

def join_clusters(rfm_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    lbl = labels_df.copy()
    if lbl.index.name != "customer_id":
        if "customer_id" in lbl.columns:
            lbl = lbl.set_index("customer_id")
        else:
            raise ValueError("labels_df không có customer_id")
    if "customer_id" in rfm_df.columns:
        merged = rfm_df.set_index("customer_id").join(lbl, how="left")
        return merged.reset_index()
    else:
        return rfm_df.join(lbl, how="left")

# ================== LOAD DATA ==================
try:
    orders, rfm_base = build_rfm()
except Exception as e:
    st.error(f"Lỗi dựng RFM: {e}")
    st.stop()

try:
    model, scaler, labels_df, profile_df, meta, mapping = load_artifacts_cached(GMM_DIR)
except FileNotFoundError:
    st.warning(
        f"Chưa có artifacts GMM tại {GMM_DIR}. Hãy chạy:\n\n"
        "python -m src.cluster_profile fit --orders data/orders_full.csv"
    )
    st.stop()
except Exception as e:
    st.error(f"Lỗi load artifacts GMM: {e}")
    st.stop()

try:
    rfm_all = join_clusters(rfm_base, labels_df)
except Exception as e:
    st.error(f"Lỗi join labels vào RFM: {e}")
    st.stop()

if "cluster_gmm" not in rfm_all.columns:
    st.error("Không tìm thấy cột cluster_gmm sau join. Kiểm tra artifacts.")
    st.stop()

# ================== INPUT ==================
st.subheader("Tìm và phân tích khách hàng trong bộ dữ liệu")
st.markdown("Lưu ý: Chỉ chấp nhận ID trong khoảng 1000 – 5000.")
input_id_raw = st.text_input("Nhập customer_id (Enter để xác nhận):", value="")
if not input_id_raw.strip():
    st.info("Vui lòng nhập customer_id để tra cứu.")
    st.stop()
if not input_id_raw.isdigit():
    st.error("customer_id phải là số.")
    st.stop()
input_id = int(input_id_raw)
if not (1000 <= input_id <= 5000):
    st.error("Khách hàng không thuộc dữ liệu cho phép (ngoài khoảng 1000–5000).")
    st.stop()
cust_df = rfm_all[rfm_all["customer_id"].astype(int) == input_id]
if cust_df.empty:
    st.error("Không tìm thấy khách hàng trong dữ liệu.")
    st.stop()
row = cust_df.iloc[0]
cluster_id = row.get("cluster_gmm", None)

# Chuẩn bị lịch sử đơn hàng
cust_id_col = None
for cand in ["member_number", "customer_id"]:
    if cand in orders.columns:
        cust_id_col = cand
        break
if cust_id_col is not None:
    cust_orders = orders[orders[cust_id_col].astype(str) == str(row["customer_id"])].copy()
else:
    cust_orders = pd.DataFrame()

# ================== GLOBAL STATS ==================
rec_median = rfm_all["Recency"].median()
freq_median = rfm_all["Frequency"].median()
mon_median = rfm_all["Monetary"].median()

# ================== STYLE (UPDATED) ==================
st.markdown("""
<style>
  :root {
      --font-base: 17px;
      --font-small: 13.5px;
      --font-medium: 15.5px;
      --font-metric-value: 22px;
      --line-base: 1.55;
      --line-tight: 1.35;
  }
  .rfm-section * {
      font-size: var(--font-base) !important;
      line-height: var(--line-base);
  }
  div[data-testid="stMetric"] {
      background: #f7faff;
      padding: 12px 14px;
      border-radius: 10px;
      box-shadow: 0 0 0 1px #e1ebf7, 0 2px 4px rgba(0,0,0,0.04);
      min-height: 92px;
  }
  div[data-testid="stMetricLabel"] {
      font-size: var(--font-small) !important;
      color:#2f5f92 !important;
      font-weight:600 !important;
      letter-spacing:.2px;
  }
  div[data-testid="stMetricValue"] {
      font-size: var(--font-metric-value) !important;
      font-weight:600 !important;
      color:#0f4f85 !important;
      line-height: var(--line-tight);
  }
  .snapshot-wrapper {
      background:#eef5fd;
      padding:16px 18px 10px 18px;
      border-radius:12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      margin-bottom:14px;
      border:1px solid #d8e6f6;
  }
  .snapshot-title {
      font-size:20px;
      font-weight:650;
      margin:0 0 10px 0;
      color:#0d4d92;
      letter-spacing:.3px;
  }
  .blue-box {
      background:#f0f7ff;
      border-radius:12px;
      padding:18px 22px 14px 22px;
      margin-bottom:26px;
      font-size:var(--font-medium);
      line-height:1.58;
      box-shadow:0 2px 6px rgba(0,0,0,0.05);
      border:1px solid #d2e5f7;
  }
  .blue-box h4 {
      margin:0 0 14px 0;
      font-size:19px;
      font-weight:650;
      color:#0d4d92;
      line-height:1.3;
      letter-spacing:.4px;
  }
  .blue-box ul {
      margin:8px 0 4px 20px;
      padding:0;
      list-style:disc;
  }
  .blue-box li {
      margin:6px 0 8px 0;
      padding-left:2px;
  }
  .blue-box li:last-child { margin-bottom:4px; }
  .pill {
      display:inline-block;
      background:#1976d2;
      color:#fff;
      padding:3px 9px 4px 9px;
      border-radius:14px;
      font-size:12px;
      font-weight:500;
      margin-right:5px;
      margin-bottom:6px;
      line-height:1.2;
      box-shadow:0 1px 2px rgba(0,0,0,0.15);
  }
  .priority-badge {
      font-size:12px;
      padding:3px 7px 3px 7px;
      border-radius:12px;
      background:#ffffff;
      border:1px solid #1976d2;
      color:#1976d2;
      margin-left:8px;
      font-weight:500;
  }
  .cat-Reactivation { background:#d32f2f !important; }
  .cat-Onboarding { background:#0288d1 !important; }
  .cat-Monetize { background:#6a1b9a !important; }
  .cat-Growth { background:#2e7d32 !important; }
  .cat-Retention { background:#ef6c00 !important; }
  .cat-General { background:#546e7a !important; }
  .blue-box a { color:#0d54ad; text-decoration:none; }
  .blue-box a:hover { text-decoration:underline; }
</style>
""", unsafe_allow_html=True)

# ================== RADAR ==================
def make_rfm_radar(rval, fval, mval):
    categories = ["R", "F", "M"]
    values = [rval, fval, mval]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='RFM Score',
        line=dict(color="#1E88E5")
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5], dtick=1)),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=10),
        title="R-F-M Radar (1–5)"
    )
    return fig

# ================== QUALITATIVE ==================
def qualitative_recency(rec_d, median):
    if rec_d <= 0.6 * median: return "rất mới"
    if rec_d <= median: return "khá mới"
    if rec_d <= 1.4 * median: return "xa dần"
    return "rất xa"
def qualitative_freq(fv, median):
    if fv >= 1.8 * median: return "tần suất vượt trội"
    if fv >= 1.2 * median: return "tần suất cao"
    if fv >= 0.8 * median: return "tần suất trung bình"
    return "tần suất thấp"
def qualitative_mon(mv, median):
    if mv >= 2.2 * median: return "chi tiêu cực cao"
    if mv >= 1.4 * median: return "chi tiêu cao"
    if mv >= 0.9 * median: return "chi tiêu trung bình"
    return "chi tiêu thấp"
q_rec = qualitative_recency(row["Recency"], rec_median)
q_freq = qualitative_freq(row["Frequency"], freq_median)
q_mon = qualitative_mon(row["Monetary"], mon_median)

# ================== FETCH CLUSTER PROFILE ROW ==================
def fetch_cluster_row(profile, cid):
    if cid is None or pd.isna(cid):
        return None
    prof = profile.copy()
    if isinstance(prof.columns, pd.MultiIndex):
        prof.columns = ["_".join(map(str, c)) for c in prof.columns]
    if cid in prof.index:
        return prof.loc[cid]
    try:
        str_map = {str(i): i for i in prof.index}
        if str(cid) in str_map:
            return prof.loc[str_map[str(cid)]]
    except Exception:
        pass
    return None

# ================== ĐỘ LỆCH SO VỚI CLUSTER ==================
def cluster_deviation_text():
    clrow = fetch_cluster_row(profile_df, cluster_id)
    if clrow is None:
        return ""
    cr = clrow.get("Recency_mean")
    cf = clrow.get("Frequency_mean")
    cm = clrow.get("Monetary_mean")
    if any(pd.isna([cr, cf, cm])):
        return ""
    dev_r = row["Recency"] - cr
    dev_f = row["Frequency"] - cf
    dev_m = row["Monetary"] - cm
    def fmt(v, invert=False):
        if invert:
            if v < -0.5: return f"mới hơn {abs(v):.1f}d"
            if v > 0.5: return f"lâu hơn {v:.1f}d"
            return "tương đương"
        else:
            if v > 0.5: return f"cao hơn {v:.1f}"
            if v < -0.5: return f"thấp hơn {abs(v):.1f}"
            return "tương đương"
    return f"Recency {fmt(dev_r, invert=True)}, Frequency {fmt(dev_f)}, Monetary {fmt(dev_m)}"

cluster_dev_txt = cluster_deviation_text()

# ================== SEGMENT LOOKUP ==================
seg_key = row["RFM_Level"] if row["RFM_Level"] in segment_catalog else "OTHER"
seg_info = segment_catalog.get(seg_key, segment_catalog["OTHER"])

# ================== PERSONALIZATION ENGINE ==================
def derive_personalized_plan(row, seg_info, cluster_dev_txt):
    base_goal = seg_info["base_goal"]
    base_strategies = seg_info["base_strategies"][:]
    kpis = seg_info.get("kpi_focus", [])
    upgrade_path = seg_info.get("upgrade_path", "")
    risk_signals = seg_info.get("risk_signals", [])

    dynamic_tactics = []
    if seg_key in ("LOST", "LIGHT") and row["Recency"] > rec_median:
        dynamic_tactics.append("Chuỗi reactivation 3 bước (Email → SMS → Push)")
    if seg_key == "NEW" and row["Frequency"] == 1:
        dynamic_tactics.append("Trigger ưu đãi đơn hàng 2 trong 7 ngày")
    if seg_key in ("BIG SPENDER", "STARS") and row["Monetary"] > 2 * mon_median:
        dynamic_tactics.append("Khảo sát hài lòng + ưu đãi tri ân cá nhân")
    if seg_key == "LOYAL" and row["Frequency"] >= 1.5 * freq_median:
        dynamic_tactics.append("Đề xuất referral (mã giới thiệu)")
    if "mới hơn" in cluster_dev_txt and seg_key not in ("NEW", "ACTIVE"):
        dynamic_tactics.append("Tận dụng tương tác gần: bundle cao cấp / upsell")
    if row["Monetary"] > 2 * mon_median and row["Frequency"] < freq_median:
        dynamic_tactics.append("Giảm rào cản mua lại: gợi ý sản phẩm nhỏ để tạo nhịp")

    mod_rec = recommend_actions(
        rfm_level=row["RFM_Level"],
        cluster=int(cluster_id) if cluster_id is not None and pd.notna(cluster_id) else None,
        monetary=row["Monetary"]
    )

    combined = base_strategies + mod_rec.get("tactics", []) + dynamic_tactics
    combined = [t for t in combined if t and "(Missing recommendation module)" not in t]

    seen = set()
    final_tactics = []
    for t in combined:
        if t not in seen:
            final_tactics.append(t)
            seen.add(t)

    def classify_action(txt):
        lower = txt.lower()
        if any(k in lower for k in ["reactivation", "win-back", "tái", "kích hoạt"]):
            return "Reactivation"
        if any(k in lower for k in ["referral", "giới thiệu"]):
            return "Growth"
        if any(k in lower for k in ["upsell", "cross", "bundle", "combo"]):
            return "Monetize"
        if any(k in lower for k in ["onboarding", "đơn hàng 2", "mới"]):
            return "Onboarding"
        if any(k in lower for k in ["ưu đãi", "duy trì", "giữ chân", "retention"]):
            return "Retention"
        return "General"

    def priority_score(txt):
        cat = classify_action(txt)
        weight = {
            "Reactivation": 90 if seg_key in ("LOST", "LIGHT") else 70,
            "Onboarding": 85 if seg_key in ("NEW", "ACTIVE") else 60,
            "Monetize": 80 if seg_key in ("BIG SPENDER", "STARS", "LOYAL") else 55,
            "Growth": 75,
            "Retention": 70,
            "General": 55
        }.get(cat, 55)
        if "bundle" in txt.lower(): weight += 5
        if "survey" in txt.lower() or "khảo sát" in txt.lower(): weight -= 5
        return min(100, weight)

    enriched = [{
        "tactic": t,
        "category": classify_action(t),
        "priority": priority_score(t)
    } for t in final_tactics]
    enriched = sorted(enriched, key=lambda x: x["priority"], reverse=True)

    goal = mod_rec.get("goal", base_goal)
    if goal == "N/A":
        goal = base_goal

    return {
        "goal": goal,
        "kpis": kpis,
        "upgrade_path": upgrade_path,
        "risk_signals": risk_signals,
        "tactics": enriched,
        "notes": [n for n in mod_rec.get("notes", []) if "(Missing recommendation" not in n]
    }

personalized_plan = derive_personalized_plan(row, seg_info, cluster_dev_txt)

# ================== PREFERENCE UTILITIES (NEW) ==================
CATEGORY_ICONS = {
    "Beverages": "🥤",
    "Drink": "🥤",
    "Food": "🍱",
    "Snack": "🍪",
    "Personal Care": "🧴",
    "Cosmetics": "💄",
    "Beauty": "💄",
    "Household": "🏠",
    "Home": "🏠",
    "Electronics": "🔌",
    "Device": "🔌",
    "Fashion": "👕",
    "Apparel": "👕",
    "Health": "💊",
    "Book": "📚",
    "Sports": "🏃",
    "Baby": "🍼",
    "Pet": "🐾"
}

def icon_for_category(cat: str) -> str:
    if not isinstance(cat, str) or not cat.strip():
        return "📦"
    lower = cat.lower()
    for key, ic in CATEGORY_ICONS.items():
        if key.lower() in lower:
            return ic
    return "📦"

def extract_customer_preferences(cust_orders: pd.DataFrame, top_n: int = 3):
    if cust_orders is None or cust_orders.empty:
        return [], []
    prod_col_candidates = ["product_name", "sku_name", "product_title", "product_id"]
    prod_col = next((c for c in prod_col_candidates if c in cust_orders.columns), None)
    cat_col_candidates = ["category", "category_name", "department", "cat_name"]
    cat_col = next((c for c in cat_col_candidates if c in cust_orders.columns), None)
    value_col = "gross_sales" if "gross_sales" in cust_orders.columns else None

    top_products = []
    if prod_col:
        prod_df = cust_orders[[prod_col] + ([value_col] if value_col else [])].copy()
        if value_col:
            prod_rank = (prod_df.groupby(prod_col, dropna=True)[value_col]
                         .sum()
                         .sort_values(ascending=False)
                         .head(top_n))
            for name, val in prod_rank.items():
                top_products.append(f"{name} ({val:,.0f})")
        else:
            prod_rank = (prod_df.groupby(prod_col, dropna=True)
                         .size()
                         .sort_values(ascending=False)
                         .head(top_n))
            for name, cnt in prod_rank.items():
                top_products.append(f"{name} ({cnt}x)")

    top_categories = []
    if cat_col:
        cat_df = cust_orders[[cat_col] + ([value_col] if value_col else [])].copy()
        if value_col:
            cat_rank = (cat_df.groupby(cat_col, dropna=True)[value_col]
                        .agg(['sum', 'count'])
                        .sort_values("sum", ascending=False)
                        .head(top_n))
            for idx, r in cat_rank.iterrows():
                top_categories.append((idx, r["sum"], r["count"]))
        else:
            cat_rank = (cat_df.groupby(cat_col, dropna=True)
                        .size()
                        .sort_values(ascending=False)
                        .head(top_n))
            for idx, cnt in cat_rank.items():
                top_categories.append((idx, None, cnt))

    return top_products, top_categories

# ================== SNAPSHOT ==================
st.markdown('<div class="rfm-section">', unsafe_allow_html=True)
st.markdown('<div class="snapshot-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="snapshot-title">RFM Snapshot</div>', unsafe_allow_html=True)

marketing_name = None
label_desc = None
if profile_df is not None and cluster_id is not None and pd.notna(cluster_id):
    clrow = fetch_cluster_row(profile_df, cluster_id)
    if clrow is not None:
        marketing_name = clrow.get("cluster_marketing_name")
        label_desc = clrow.get("cluster_label_desc")

top_cols = st.columns([1,1,1,1,1,1])
with top_cols[0]: st.metric("Customer ID", row["customer_id"])
with top_cols[1]: st.metric("Recency (days)", int(row["Recency"]))
with top_cols[2]: st.metric("Frequency", int(row["Frequency"]))
with top_cols[3]: st.metric("Monetary", f"{row['Monetary']:,.0f}")
with top_cols[4]: st.metric("RFM Level", row["RFM_Level"])
if "cluster_confidence" in row:
    with top_cols[5]:
        st.metric("Cluster Conf.", f"{row['cluster_confidence']:.2f}" if pd.notna(row['cluster_confidence']) else "—")

bottom_cols = st.columns([1,1,1,1,1])
with bottom_cols[0]: st.metric("R score", int(row["R"]))
with bottom_cols[1]: st.metric("F score", int(row["F"]))
with bottom_cols[2]: st.metric("M score", int(row["M"]))
with bottom_cols[3]: st.metric("Cluster GMM", cluster_id if pd.notna(cluster_id) else "N/A")
with bottom_cols[4]:
    if marketing_name:
        st.metric("Cluster Name", marketing_name)
    elif label_desc:
        st.metric("Cluster Desc", str(label_desc)[:18])
    else:
        st.metric("Cluster Name", "—")
st.markdown('</div>', unsafe_allow_html=True)

# ================== RADAR + BOXES ==================
st.markdown("---")
st.markdown("### Customer Insights")

def render_analysis_box():
    pct_rec_better = (rfm_all["Recency"] < row["Recency"]).mean() * 100
    pct_freq = (rfm_all["Frequency"] <= row["Frequency"]).mean() * 100
    pct_mon = (rfm_all["Monetary"] <= row["Monetary"]).mean() * 100
    items = []
    cluster_name_line = ""
    if marketing_name:
        cluster_name_line = f" (Cluster: {marketing_name})"
    elif label_desc:
        cluster_name_line = f" (Cluster: {label_desc})"
    items.append(f"Segment: <b>{seg_key}</b>{cluster_name_line} – {seg_info['definition']}")
    items.append(f"Recency: {int(row['Recency'])} ngày → {q_rec} (Median {int(rec_median)}; {pct_rec_better:.1f}% khách mới hơn).")
    items.append(f"Frequency: {int(row['Frequency'])} → {q_freq} (Median {freq_median:.1f}; Percentile ≈ {pct_freq:.1f}%).")
    items.append(f"Monetary: {row['Monetary']:.0f} → {q_mon} (Median {mon_median:.0f}; Percentile ≈ {pct_mon:.1f}%).")
    if cluster_dev_txt:
        items.append(f"So với cụm: {cluster_dev_txt}.")
    if seg_key in ("LOST","LIGHT"):
        items.append("Nguy cơ giảm tương tác → ưu tiên kích hoạt lại.")
    elif seg_key in ("STARS","BIG SPENDER","LOYAL"):
        items.append("Giá trị cao – tối ưu giữ chân & CLV.")
    elif seg_key == "NEW":
        items.append("Cần đảm bảo mua lần 2 nhanh (≤30 ngày).")

    top_products, top_categories = extract_customer_preferences(cust_orders, top_n=3)
    if top_products:
        items.append("Sản phẩm thường mua: " + ", ".join(top_products))
    else:
        items.append("Sản phẩm thường mua: (không đủ dữ liệu)")
    if top_categories:
        cat_parts = []
        for cat, val, cnt in top_categories:
            ic = icon_for_category(str(cat))
            if val is not None:
                cat_parts.append(f"{ic} {cat} ({val:,.0f}; {cnt}x)")
            else:
                cat_parts.append(f"{ic} {cat} ({cnt}x)")
        items.append("Ngành hàng ưu thế: " + ", ".join(cat_parts))
    else:
        items.append("Ngành hàng ưu thế: (không đủ dữ liệu)")

    html_items = "".join(f"<li>{x}</li>" for x in items)
    return f"""
    <div class="blue-box">
      <h4>Phân tích đặc điểm</h4>
      <ul>{html_items}</ul>
    </div>
    """

def render_strategy_box(plan):
    tactic_html_parts = []
    for t in plan["tactics"]:
        cat = t["category"]
        pr = t["priority"]
        tactic_html_parts.append(
            f"<li>{t['tactic']} <span class='priority-badge'>{pr}</span> "
            f"<span class='pill cat-{cat}'>{cat}</span></li>"
        )
    kpi_html = " ".join(f"<span class='pill'>{k}</span>" for k in plan["kpis"])
    risk_html = " ".join(f"<span class='pill' style='background:#b71c1c'>{r}</span>" for r in plan["risk_signals"])
    notes_html = ""
    if plan.get("notes"):
        notes_html = "<p><i>Notes: " + " | ".join(plan["notes"]) + "</i></p>"
    return f"""
    <div class="blue-box">
      <h4>Kế hoạch hành động cá nhân</h4>
      <p><b>Goal:</b> {plan['goal']}</p>
      <p><b>KPIs:</b> {kpi_html if kpi_html else '—'}</p>
      <p><b>Nguy cơ theo dõi:</b> {risk_html if risk_html else '—'}</p>
      <p><b>Đường nâng cấp:</b> {plan['upgrade_path']}</p>
      <p><b>Chiến thuật ưu tiên:</b></p>
      <ul>{''.join(tactic_html_parts)}</ul>
      {notes_html}
    </div>
    """

radar_col, explain_col = st.columns([1.05, 1])
with radar_col:
    st.plotly_chart(make_rfm_radar(int(row["R"]), int(row["F"]), int(row["M"])), use_container_width=True)
with explain_col:
    st.markdown(render_analysis_box(), unsafe_allow_html=True)
    st.markdown(render_strategy_box(personalized_plan), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close rfm-section

# ================== PROBABILITY COMPONENT (NẾU CÓ) ==================
prob_cols = [c for c in rfm_all.columns if c.startswith("prob_")]
if prob_cols and cluster_id is not None and pd.notna(cluster_id):
    st.markdown("### Phân bổ Xác suất Thành phần (GMM)")
    row_probs = row[prob_cols].dropna()
    if not row_probs.empty:
        prob_df = pd.DataFrame({
            "Component": [c.replace("prob_", "") for c in row_probs.index],
            "Probability": row_probs.values
        })
        prob_fig = px.bar(prob_df, x="Component", y="Probability", text="Probability",
                          template="plotly_white", title="Component Membership Probabilities")
        prob_fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
        prob_fig.update_yaxes(range=[0, 1])
        st.plotly_chart(prob_fig, use_container_width=True)
    else:
        st.caption("Không có giá trị probability hợp lệ.")

# ================== LỊCH SỬ MUA HÀNG ==================
st.markdown("### Lịch sử mua hàng")
if not cust_orders.empty and {"date"}.issubset(cust_orders.columns):
    value_col = "gross_sales" if "gross_sales" in cust_orders.columns else None
    cust_orders["_date"] = pd.to_datetime(cust_orders["date"], errors="coerce")
    if value_col:
        daily = (cust_orders
                 .groupby(cust_orders["_date"].dt.date)
                 .agg(metric_val=(value_col, "sum"),
                      orders=("order_id", "nunique")))
        st.line_chart(daily["metric_val"])
    else:
        daily = (cust_orders
                 .groupby(cust_orders["_date"].dt.date)
                 .agg(orders=("order_id", "nunique")))
        st.line_chart(daily["orders"])
else:
    st.info("Không đủ cột (date) để vẽ lịch sử hoặc không có đơn hàng.")

# ================== SO SÁNH VỚI TRUNG BÌNH CỤM ==================
st.markdown("### So sánh với Trung bình Cụm")
cl_profile_row = fetch_cluster_row(profile_df, cluster_id)
if cluster_id is not None and pd.notna(cluster_id) and cl_profile_row is not None:
    compare = pd.DataFrame({
        "Metric": ["Recency", "Frequency", "Monetary"],
        "ClusterMean": [
            cl_profile_row.get("Recency_mean"),
            cl_profile_row.get("Frequency_mean"),
            cl_profile_row.get("Monetary_mean")
        ],
        "Customer": [row["Recency"], row["Frequency"], row["Monetary"]]
    })
    long_cmp = compare.melt(id_vars="Metric", value_vars=["ClusterMean", "Customer"],
                            var_name="Type", value_name="Value")
    name_map = {"Recency": "Recency (days ↓ tốt)", "Frequency": "Frequency", "Monetary": "Monetary"}
    long_cmp["MetricLabel"] = long_cmp["Metric"].map(name_map)
    fig_group = px.bar(
        long_cmp,
        x="MetricLabel", y="Value", color="Type",
        barmode="group",
        text="Value",
        template="plotly_white",
        title="Customer vs Cluster Mean"
    )
    fig_group.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_group.update_layout(yaxis_title="Value", legend_title="")
    st.plotly_chart(fig_group, use_container_width=True)

    compare["DiffPctRaw"] = (compare["Customer"] - compare["ClusterMean"]) / (compare["ClusterMean"] + 1e-9) * 100
    def adj(row_):
        if row_["Metric"] == "Recency":
            return -row_["DiffPctRaw"]  # Recency thấp hơn là tốt → đảo dấu
        return row_["DiffPctRaw"]
    compare["DiffPctAdj"] = compare.apply(adj, axis=1)
    compare["Direction"] = np.where(compare["DiffPctAdj"] >= 0, "Better / Higher", "Worse / Lower")
    fig_diff = px.bar(
        compare,
        x="DiffPctAdj",
        y="Metric",
        color="Direction",
        orientation="h",
        text=compare["DiffPctAdj"].map(lambda v: f"{v:+.1f}%"),
        template="plotly_white",
        title="Chênh lệch % so với Cụm (đã chuẩn hoá: Recency đảo dấu)"
    )
    fig_diff.add_vline(x=0, line_color="#666", line_dash="dash")
    fig_diff.update_layout(xaxis_title="Adjusted % Difference (Positive = Performance Better)",
                           yaxis_title="")
    st.plotly_chart(fig_diff, use_container_width=True)
    with st.expander("Chi tiết số liệu so sánh"):
        st.dataframe(compare[["Metric", "ClusterMean", "Customer", "DiffPctRaw", "DiffPctAdj"]].round(3))
else:
    st.info("Cluster profile không khả dụng.")

# ================== RAW ORDERS ==================
with st.expander("Chi tiết đơn hàng (top 50 gần nhất)"):
    if not cust_orders.empty and "date" in cust_orders.columns:
        cust_orders = cust_orders.sort_values("date", ascending=False)
    st.dataframe(cust_orders.head(50))

# ============ FOOTER ============
st.markdown(
    "<div style='text-align:left; color:#666; font-size:13px; margin-top:30px;'>© 2025 Đồ án tốt nghiệp lớp DL07_K306 - RFM Segmentation - Nhóm J</div>",
    unsafe_allow_html=True
)
