import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.rfm_base import load_orders, build_rfm_snapshot
from src.rfm_rule_scoring import compute_rfm_scores
from src.rfm_labeling import apply_rfm_level

try:
    from src.combo_recommender import (
        prepare_line_items,
        build_cooccurrence,
        build_customer_profile,
        recommend_combos_for_customer
    )
    _combo_available = True
except Exception:
    _combo_available = False

try:
    from src.cluster_profile import load_artifacts
except Exception as e:
    load_artifacts = None
    _cluster_import_error = str(e)

try:
    from src.recommendation import recommend_actions
except Exception:
    def recommend_actions(rfm_level: str, cluster=None, monetary=None):
        return {"goal": "N/A", "tactics": [], "notes": []}

st.set_page_config(page_title="04 - Customer", layout="wide")

DATA_PATH = Path("data/orders_full.csv")
GMM_DIR = Path("models/gmm/gmm_rfm_v1")

# ================= Segment Catalog =================
segment_catalog = {
    "LOST":{"definition":"Khách hàng lâu không quay lại, Recency cao vượt ngưỡng.","base_goal":"Tái kích hoạt hoặc xác định lý do rời bỏ.","base_strategies":["Win-back voucher / quà sinh nhật","Flash sale tái kích hoạt","Khảo sát lý do rời bỏ"],"kpi_focus":["Reactivation Rate","Open Rate","Return Purchase"],"upgrade_path":"Chuyển thành ACTIVE rồi REGULARS / LOYAL","risk_signals":["Recency cao","Frequency giảm","Không phản hồi chiến dịch"]},
    "REGULARS":{"definition":"Khách mua đều đặn, hành vi ổn định.","base_goal":"Duy trì tần suất và tăng giá trị đơn hàng.","base_strategies":["Ưu đãi duy trì nhẹ","Theo dõi nâng cấp lên LOYAL / BIG SPENDER","Tối ưu trải nghiệm"],"kpi_focus":["Repeat Rate","AOV","Frequency"],"upgrade_path":"Nâng lên LOYAL hoặc BIG SPENDER","risk_signals":["Tần suất giảm tuần/tháng","Giảm giá trị đơn"]},
    "BIG SPENDER":{"definition":"Chi tiêu lớn, giá trị cao trong kỳ.","base_goal":"Gia tăng vòng đời và giảm rủi ro giảm chi tiêu.","base_strategies":["CSKH ưu tiên / hotline riêng","Gợi ý combo / subscription","Ưu đãi cá nhân hoá giữ chân"],"kpi_focus":["CLV","AOV","Retention"],"upgrade_path":"Kết hợp thành STARS (nếu tần suất tăng)","risk_signals":["Giảm mạnh AOV","Khoảng cách mua kéo dài"]},
    "STARS":{"definition":"Tần suất cao & chi tiêu cao – nhóm lõi rất giá trị.","base_goal":"Khóa chặt trung thành và khai thác referral.","base_strategies":["Chăm sóc VIP / event độc quyền","Upsell & cross-sell cao cấp","Referral thưởng cao"],"kpi_focus":["Referral Rate","CLV","Upsell Rate"],"upgrade_path":"Duy trì đỉnh giá trị, giảm rủi ro bão hòa","risk_signals":["Giảm tần suất bất thường","Không phản hồi ưu đãi cao cấp"]},
    "LIGHT":{"definition":"Mua thưa và chi tiêu thấp.","base_goal":"Tăng tần suất và giá trị giỏ.","base_strategies":["Combo nhỏ tăng giá trị đơn","Content nuôi dưỡng & review","Ưu đãi nhỏ nhưng đều"],"kpi_focus":["Frequency","AOV","Activation"],"upgrade_path":"Nâng thành REGULARS rồi LOYAL","risk_signals":["Không quay lại sau chiến dịch","AOV không tăng"]},
    "ACTIVE":{"definition":"Vừa quay lại gần đây, tần suất còn thấp.","base_goal":"Khóa nhịp mua lặp lại 2–3 lần liên tiếp.","base_strategies":["Ưu đãi kích hoạt (Mua 2 tặng 1)","Remarketing email / push","Upsell nhẹ sản phẩm liên quan"],"kpi_focus":["Second Purchase Rate","Repeat Cycle Time"],"upgrade_path":"Đẩy thành REGULARS hoặc LOYAL","risk_signals":["Không có đơn thứ 2 trong 30 ngày"]},
    "LOYAL":{"definition":"Trung thành, mua lặp lại ổn định.","base_goal":"Duy trì và tăng CLV / AOV.","base_strategies":["Tích điểm / gamification","Referral program","Ưu tiên thử sản phẩm mới"],"kpi_focus":["Retention","Referral","CLV"],"upgrade_path":"Phát triển thành STARS / BIG SPENDER","risk_signals":["Giảm tần suất dần","Không dùng điểm thưởng"]},
    "NEW":{"definition":"Khách hàng mới – mua lần đầu.","base_goal":"Kích hoạt mua lần thứ 2 nhanh.","base_strategies":["Email cảm ơn + voucher đơn 2","Onboarding giới thiệu sản phẩm bán chạy","Nhắc quay lại trong 30 ngày"],"kpi_focus":["Second Purchase Rate","Onboarding Completion"],"upgrade_path":"ACTIVE rồi REGULARS","risk_signals":["Không mua lại <30 ngày","Không mở email onboarding"]},
    "OTHER":{"definition":"Nhóm nhỏ / chưa rõ đặc trưng.","base_goal":"Thu thập thêm dữ liệu hành vi.","base_strategies":["Theo dõi thêm hành vi","Điều chỉnh tiêu chí phân nhóm","Kiểm soát chi phí chăm sóc"],"kpi_focus":["Data Completeness"],"upgrade_path":"Phân bổ lại sang nhóm chính","risk_signals":["Khối lượng thấp","Nhiễu nhãn"]}
}

RADAR_TARGET_HEIGHT = 250

# ================= Cached Functions =================
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
    return load_artifacts(dir_path)

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
    return rfm_df.join(lbl, how="left")

@st.cache_data(show_spinner=False)
def compute_combo_rules(raw_orders: pd.DataFrame, min_support_orders: int = 5):
    if not _combo_available:
        return pd.DataFrame(), {}
    df = raw_orders.copy()
    cust_col = next((c for c in ["customer_id","member_number","cust_id","user_id"] if c in df.columns), None)
    prod_col = next((c for c in ["product_name","product","sku_name","item_name","product_title","product_id"] if c in df.columns), None)
    if prod_col is None or "order_id" not in df.columns:
        return pd.DataFrame(), {"status":"missing_columns"}
    if cust_col != "customer_id":
        df = df.rename(columns={cust_col:"customer_id"})
    if prod_col != "product_name":
        df = df.rename(columns={prod_col:"product_name"})
    df = df[~df["product_name"].isna()]
    if df.empty or df["order_id"].nunique() == 0:
        return pd.DataFrame(), {"status":"no_data"}
    try:
        line = prepare_line_items(df, product_col="product_name")
        rules, prod_orders, total_orders = build_cooccurrence(
            line, product_col="product_name", min_support_orders=min_support_orders
        )
        return rules, {"status":"ok","total_orders":int(total_orders)}
    except Exception as e:
        return pd.DataFrame(), {"status":"error","error":str(e)}

# ================= Load Data =================
try:
    orders, rfm_base = build_rfm()
except Exception as e:
    st.error(f"Lỗi dựng RFM: {e}")
    st.stop()

try:
    model, scaler, labels_df, profile_df, meta, mapping = load_artifacts_cached(GMM_DIR)
except FileNotFoundError:
    st.warning("Thiếu artifacts GMM. Chạy: python -m src.cluster_profile fit --orders data/orders_full.csv")
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
    st.error("Không có cluster_gmm trong dữ liệu.")
    st.stop()

with st.spinner("Đang tính toán gợi ý combo..."):
    combo_rules, combo_meta = compute_combo_rules(orders, min_support_orders=5)

# ================= Input =================
st.title("👤 Phân tích Khách hàng chuyên sâu")
st.markdown("Nhập ID khách hàng (1000–5000).")
input_id_raw = st.text_input("customer_id:", value="")
if not input_id_raw.strip():
    st.stop()
if not input_id_raw.isdigit():
    st.error("customer_id phải là số.")
    st.stop()
input_id = int(input_id_raw)
if not (1000 <= input_id <= 5000):
    st.error("Ngoài phạm vi (1000–5000).")
    st.stop()

cust_df = rfm_all[rfm_all["customer_id"].astype(int) == input_id]
if cust_df.empty:
    st.error("Không tìm thấy khách hàng.")
    st.stop()
row = cust_df.iloc[0]
cluster_id = row.get("cluster_gmm", None)

# ================= Customer Orders =================
cust_id_col = next((c for c in ["member_number","customer_id"] if c in orders.columns), None)
if cust_id_col:
    cust_orders = orders[orders[cust_id_col].astype(str) == str(row["customer_id"])].copy()
else:
    cust_orders = pd.DataFrame()

# ================= Combo Recs =================
combo_recs = []
if _combo_available and not combo_rules.empty:
    try:
        orders_for_rec = orders.copy()
        if cust_id_col and cust_id_col != "customer_id":
            orders_for_rec = orders_for_rec.rename(columns={cust_id_col:"customer_id"})
        if "product_name" not in orders_for_rec.columns:
            for alt in ["product","sku_name","item_name","product_title","product_id"]:
                if alt in orders_for_rec.columns:
                    orders_for_rec = orders_for_rec.rename(columns={alt:"product_name"})
                    break
        if "product_name" in orders_for_rec.columns:
            combo_recs = recommend_combos_for_customer(
                cust_id=row["customer_id"],
                orders=orders_for_rec,
                rules_df=combo_rules,
                product_col="product_name",
                top_k=7,
                min_lift=1.05
            )
    except Exception:
        combo_recs = []

# ================= Medians =================
rec_median = rfm_all["Recency"].median()
freq_median = rfm_all["Frequency"].median()
mon_median = rfm_all["Monetary"].median()

# ================= Styles =================
st.markdown("""
<style>
  .segment-header {border-radius:14px; padding:18px 22px 14px 22px; margin: 6px 0 18px 0; display:flex; align-items:center; justify-content:space-between; box-shadow:0 2px 6px rgba(0,0,0,0.07); color:#fff; font-family:'Inter','Segoe UI',system-ui;}
  .segment-header h2 { font-size:26px; font-weight:700; margin:0;}
  .segment-badge { font-size:16px; font-weight:600; padding:6px 16px; background:rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.40); border-radius:24px;}
  .metric-card, .cluster-card, .radar-wrap { background:#f6f9fe; border:1px solid #dbe8f7; border-radius:14px; box-shadow:0 1px 4px rgba(0,0,0,0.06);}
  .metric-card { padding:18px 20px 18px 20px; display:flex; flex-direction:column; gap:14px; height:100%;}
  .metric-card h4 { margin:0; font-size:16px; font-weight:660; color:#0d4d92; letter-spacing:.3px;}
  .rfm-flex { display:flex; gap:16px; }
  .rfm-col { flex:1; display:flex; flex-direction:column; gap:10px; }
  .metric-item { background:#ffffff; border:1px solid #e2ecf6; border-radius:11px; padding:10px 10px 8px 10px; text-align:center; display:flex; flex-direction:column; justify-content:center; min-height:72px;}
  .metric-item span.label { font-size:12px; color:#4372a3; font-weight:500; margin-bottom:4px; letter-spacing:.25px;}
  .metric-item span.value { font-size:20px; font-weight:600; color:#0f4f85; line-height:1.05;}
  .cluster-card { padding:16px 18px 14px 18px; display:flex; flex-direction:column; gap:12px; height:100%;}
  .cluster-card h4 { margin:0; font-size:16px; font-weight:660; color:#0d4d92; letter-spacing:.3px;}
  .cluster-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }
  .c-box { background:#ffffff; border:1px solid #e3edf7; border-radius:10px; padding:8px 10px 6px 10px; display:flex; flex-direction:column; justify-content:center; min-height:60px; text-align:center;}
  .c-box .label { font-size:11px; color:#4372a3; font-weight:500; margin-bottom:3px; letter-spacing:.25px;}
  .c-box .value { font-size:18px; font-weight:600; color:#0f4f85; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .c-desc { grid-column:1 / span 2; min-height:100px; text-align:left; align-items:flex-start; padding:10px 12px 8px 12px;}
  .c-desc .value { font-size:13px; font-weight:600; color:#0f4f85; line-height:1.25; white-space:normal; display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden;}
  .radar-wrap { padding:16px 14px 10px 14px; display:flex; flex-direction:column; gap:8px; height:100%;}
  .radar-title { font-size:16px; font-weight:660; color:#0d4d92; margin:0; letter-spacing:.3px;}
  .blue-box { background:#f0f7ff; border-radius:14px; padding:22px 26px 16px 26px; margin:10px 0 26px 0; font-size:15.2px; line-height:1.58; box-shadow:0 2px 5px rgba(0,0,0,0.05); border:1px solid #d2e5f7;}
  .blue-box h4 { margin:0 0 14px 0; font-size:20px; font-weight:700; color:#0d4d92;}
  .blue-box ul { margin:4px 0 4px 22px; padding:0;}
  .blue-box li { margin:4px 0 8px 0;}
  .priority-badge { font-size:11px; padding:3px 7px; border-radius:10px; background:#ffffff; border:1px solid #1976d2; color:#1976d2; margin-left:6px; font-weight:500;}
  .pill { display:inline-block; background:#1976d2; color:#fff; padding:4px 10px 5px 10px; border-radius:16px; font-size:12px; font-weight:600; margin:3px 6px 6px 0; line-height:1.05; box-shadow:0 1px 2px rgba(0,0,0,0.15); position:relative; cursor:help;}
  .pill[data-tip]:hover::after { content:attr(data-tip); position:absolute; left:50%; transform:translateX(-50%); bottom:110%; background:#0d4d92; color:#fff; padding:8px 10px; border-radius:8px; width:230px; font-size:11.5px; line-height:1.4; box-shadow:0 4px 10px rgba(0,0,0,0.18); z-index:30; pointer-events:none; text-align:left;}
  .pill[data-tip]:hover::before { content:""; position:absolute; left:50%; transform:translateX(-50%); bottom:100%; border:6px solid transparent; border-top-color:#0d4d92;}
  .cat-Reactivation { background:#d32f2f !important; }
  .cat-Onboarding { background:#0288d1 !important; }
  .cat-Monetize { background:#6A1B9A !important; }
  .cat-Growth { background:#2e7d32 !important; }
  .cat-Retention { background:#ef6c00 !important; }
  .cat-General { background:#546e7a !important; }
  .risk-pill { background:#b71c1c !important; }
  .history-title { font-weight:600; margin:8px 0 10px 0; font-size:16px; color:#0d4d92;}
  .triple-wrapper { display:flex; gap:14px; width:100%; margin:10px 0 6px 0; }
  .pref-box { background:#ffffff; border:1px solid #dbe8f7; border-radius:14px; padding:14px 16px 12px 16px; flex:1; font-size:14.8px; line-height:1.5; box-shadow:0 1px 3px rgba(0,0,0,0.05); display:flex; flex-direction:column;}
  .pref-box h5 { margin:0 0 10px 0; font-size:15.5px; font-weight:650; color:#0d4d92; letter-spacing:.2px; }
  .pref-box ul { margin:0; padding-left:18px; }
  .pref-box li { margin:4px 0 6px 0; }
  .combo-list { margin:0; padding-left:18px; }
  .note-hover { font-size:12px; color:#555; margin:4px 0 0 2px; }
  @media (max-width:1100px){
     .triple-wrapper { flex-direction:column; }
  }
</style>
""", unsafe_allow_html=True)

# ================= Qualitative Helpers =================
def qualitative_recency(rec_d, median):
    if rec_d <= 0.6*median: return "rất mới"
    if rec_d <= median: return "khá mới"
    if rec_d <= 1.4*median: return "xa dần"
    return "rất xa"
def qualitative_freq(fv, median):
    if fv >= 1.8*median: return "tần suất vượt trội"
    if fv >= 1.2*median: return "tần suất cao"
    if fv >= 0.8*median: return "tần suất trung bình"
    return "tần suất thấp"
def qualitative_mon(mv, median):
    if mv >= 2.2*median: return "chi tiêu cực cao"
    if mv >= 1.4*median: return "chi tiêu cao"
    if mv >= 0.9*median: return "chi tiêu trung bình"
    return "chi tiêu thấp"

q_rec = qualitative_recency(row["Recency"], rec_median)
q_freq = qualitative_freq(row["Frequency"], freq_median)
q_mon = qualitative_mon(row["Monetary"], mon_median)

def fetch_cluster_row(profile, cid):
    if cid is None or pd.isna(cid): return None
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

def cluster_deviation_text():
    clrow = fetch_cluster_row(profile_df, cluster_id)
    if clrow is None: return ""
    cr = clrow.get("Recency_mean"); cf = clrow.get("Frequency_mean"); cm = clrow.get("Monetary_mean")
    if any(pd.isna([cr, cf, cm])): return ""
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

seg_key = row["RFM_Level"] if row["RFM_Level"] in segment_catalog else "OTHER"
seg_info = segment_catalog.get(seg_key, segment_catalog["OTHER"])

def derive_personalized_plan(row, seg_info, cluster_dev_txt):
    base_goal = seg_info["base_goal"]
    base_strategies = seg_info["base_strategies"][:]
    kpis = seg_info.get("kpi_focus", [])
    upgrade_path = seg_info.get("upgrade_path", "")
    risk_signals = seg_info.get("risk_signals", [])
    dynamic = []
    if seg_key in ("LOST","LIGHT") and row["Recency"] > rec_median:
        dynamic.append("Chuỗi reactivation 3 bước (Email → SMS → Push)")
    if seg_key == "NEW" and row["Frequency"] == 1:
        dynamic.append("Trigger ưu đãi đơn hàng 2 trong 7 ngày")
    if seg_key in ("BIG SPENDER","STARS") and row["Monetary"] > 2*mon_median:
        dynamic.append("Khảo sát hài lòng + ưu đãi tri ân cá nhân")
    if seg_key == "LOYAL" and row["Frequency"] >= 1.5*freq_median:
        dynamic.append("Đề xuất referral (mã giới thiệu)")
    if "mới hơn" in cluster_dev_txt and seg_key not in ("NEW","ACTIVE"):
        dynamic.append("Tận dụng tương tác gần: bundle cao cấp / upsell")
    if row["Monetary"] > 2*mon_median and row["Frequency"] < freq_median:
        dynamic.append("Giảm rào cản mua lại: gợi ý sản phẩm nhỏ để tạo nhịp")
    mod_rec = recommend_actions(
        rfm_level=row["RFM_Level"],
        cluster=int(cluster_id) if cluster_id is not None and pd.notna(cluster_id) else None,
        monetary=row["Monetary"]
    )
    combined = base_strategies + mod_rec.get("tactics", []) + dynamic
    combined = [t for t in combined if t and "(Missing recommendation module)" not in t]
    seen=set(); final=[]
    for t in combined:
        if t not in seen:
            final.append(t); seen.add(t)
    def classify(txt):
        lower = txt.lower()
        if any(k in lower for k in ["reactivation","win-back","tái","kích hoạt"]): return "Reactivation"
        if any(k in lower for k in ["referral","giới thiệu"]): return "Growth"
        if any(k in lower for k in ["upsell","cross","bundle","combo"]): return "Monetize"
        if any(k in lower for k in ["onboarding","đơn hàng 2","mới"]): return "Onboarding"
        if any(k in lower for k in ["ưu đãi","duy trì","giữ chân","retention"]): return "Retention"
        return "General"
    def priority(txt):
        cat = classify(txt)
        weight = {
            "Reactivation": 90 if seg_key in ("LOST","LIGHT") else 70,
            "Onboarding": 85 if seg_key in ("NEW","ACTIVE") else 60,
            "Monetize": 80 if seg_key in ("BIG SPENDER","STARS","LOYAL") else 55,
            "Growth": 75, "Retention": 70, "General": 55
        }[cat]
        if "bundle" in txt.lower(): weight +=5
        if "survey" in txt.lower() or "khảo sát" in txt.lower(): weight -=5
        return min(100, weight)
    enriched = [{"tactic":t,"category":classify(t),"priority":priority(t)} for t in final]
    enriched = sorted(enriched, key=lambda x:x["priority"], reverse=True)
    goal = mod_rec.get("goal", base_goal)
    if goal == "N/A": goal = base_goal
    return {
        "goal": goal,
        "kpis": kpis,
        "upgrade_path": upgrade_path,
        "risk_signals": risk_signals,
        "tactics": enriched,
        "notes": [n for n in mod_rec.get("notes",[]) if "(Missing recommendation" not in n]
    }
personalized_plan = derive_personalized_plan(row, seg_info, cluster_dev_txt)

DEFAULT_SEGMENT_COLORS = {
    "STARS":"#1b7837","BIG SPENDER":"#00429d","LOYAL":"#73a2c6",
    "ACTIVE":"#4daf4a","NEW":"#ffcc00","LIGHT":"#f29e4c",
    "REGULARS":"#9e9e9e","LOST":"#d73027","OTHER":"#607d8b"
}
seg_color = DEFAULT_SEGMENT_COLORS.get(seg_key, "#607d8b")

marketing_name = None; label_desc = None
if profile_df is not None and cluster_id is not None and pd.notna(cluster_id):
    clrow = fetch_cluster_row(profile_df, cluster_id)
    if clrow is not None:
        marketing_name = clrow.get("cluster_marketing_name")
        label_desc = clrow.get("cluster_label_desc")

# ================= Header =================
st.markdown(
    f"""
    <div class="segment-header" style="background:{seg_color};">
       <h2>Khách hàng #{row['customer_id']}</h2>
       <div class="segment-badge">{seg_key}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= Top Layout =================
col_left, col_right = st.columns([5,5])
with col_left:
    st.markdown(f"""
      <div class="metric-card">
        <h4>RFM Overview</h4>
          <div class="rfm-flex">
            <div class="rfm-col">
              <div class="metric-item"><span class="label">Recency</span><span class="value">{int(row['Recency'])}</span></div>
              <div class="metric-item"><span class="label">R Score</span><span class="value">{int(row['R'])}</span></div>
            </div>
            <div class="rfm-col">
              <div class="metric-item"><span class="label">Frequency</span><span class="value">{int(row['Frequency'])}</span></div>
              <div class="metric-item"><span class="label">F Score</span><span class="value">{int(row['F'])}</span></div>
            </div>
            <div class="rfm-col">
              <div class="metric-item"><span class="label">Monetary</span><span class="value">{row['Monetary']:,.0f}</span></div>
              <div class="metric-item"><span class="label">M Score</span><span class="value">{int(row['M'])}</span></div>
            </div>
          </div>
      </div>
    """, unsafe_allow_html=True)

with col_right:
    cluster_col, radar_col = st.columns([3,2])
    cluster_conf_val = row["cluster_confidence"] if "cluster_confidence" in row and pd.notna(row["cluster_confidence"]) else None
    full_desc = marketing_name or label_desc or "—"
    with cluster_col:
        st.markdown(f"""
          <div class="cluster-card">
            <h4>Cluster</h4>
            <div class="cluster-grid">
              <div class="c-box">
                <div class="label">Cluster GMM</div>
                <div class="value">{cluster_id if pd.notna(cluster_id) else 'N/A'}</div>
              </div>
              <div class="c-box">
                <div class="label">Confidence</div>
                <div class="value">{f"{cluster_conf_val:.2f}" if cluster_conf_val is not None else "—"}</div>
              </div>
              <div class="c-box c-desc">
                <div class="label">Cluster Desc</div>
                <div class="value">{full_desc}</div>
              </div>
            </div>
          </div>
        """, unsafe_allow_html=True)
    with radar_col:
        st.markdown("<div class='radar-wrap'><p class='radar-title'>Radar</p>", unsafe_allow_html=True)
        def make_rfm_radar(rval, fval, mval):
            categories = ["R","F","M"]
            values = [rval, fval, mval]
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name='',
                line=dict(color="#1E88E5", width=2)
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,5], dtick=1)),
                showlegend=False, margin=dict(l=10,r=10,t=10,b=10),
                title='', height=RADAR_TARGET_HEIGHT
            )
            return fig
        st.plotly_chart(make_rfm_radar(int(row["R"]), int(row["F"]), int(row["M"])), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ================= Phân tích đặc điểm =================
def build_analysis_points():
    pct_rec_better = (rfm_all["Recency"] < row["Recency"]).mean()*100
    pct_freq = (rfm_all["Frequency"] <= row["Frequency"]).mean()*100
    pct_mon = (rfm_all["Monetary"] <= row["Monetary"]).mean()*100
    items=[]
    cluster_name_line=""
    if marketing_name: cluster_name_line=f" (Cluster: {marketing_name})"
    elif label_desc: cluster_name_line=f" (Cluster: {label_desc})"
    items.append(f"Segment: <b>{seg_key}</b>{cluster_name_line} – {seg_info['definition']}")
    items.append(f"Recency: {int(row['Recency'])} ngày → {q_rec} (Median {int(rec_median)}; {pct_rec_better:.1f}% khách mới hơn).")
    items.append(f"Frequency: {int(row['Frequency'])} → {q_freq} (Median {freq_median:.1f}; ~{pct_freq:.1f}%).")
    items.append(f"Monetary: {row['Monetary']:.0f} → {q_mon} (Median {mon_median:.0f}; ~{pct_mon:.1f}%).")
    if cluster_dev_txt:
        items.append(f"So với cụm (tóm tắt): {cluster_dev_txt}.")
    if seg_key in ("LOST","LIGHT"):
        items.append("Nguy cơ giảm tương tác → ưu tiên kích hoạt lại.")
    elif seg_key in ("STARS","BIG SPENDER","LOYAL"):
        items.append("Giá trị cao – tập trung giữ chân & tăng CLV.")
    elif seg_key == "NEW":
        items.append("Cần đảm bảo mua lần 2 ≤ 30 ngày.")
    return items
analysis_html = "<ul>" + "".join(f"<li>{x}</li>" for x in build_analysis_points()) + "</ul>"
st.markdown(f"<div class='blue-box'><h4>Phân tích đặc điểm</h4>{analysis_html}</div>", unsafe_allow_html=True)

# ================= Lịch sử mua hàng (Full Width) =================
st.markdown("<div class='history-title'>Lịch sử mua hàng</div>", unsafe_allow_html=True)
if not cust_orders.empty and "date" in cust_orders.columns:
    value_col = "gross_sales" if "gross_sales" in cust_orders.columns else None
    cust_orders["_date"] = pd.to_datetime(cust_orders["date"], errors="coerce")
    if value_col:
        daily = (cust_orders.groupby(cust_orders["_date"].dt.date)
                 .agg(metric_val=(value_col,"sum"), orders=("order_id","nunique")))
        fig_hist = px.line(daily, y="metric_val", template="plotly_white")
        fig_hist.update_layout(yaxis_title="Doanh thu", xaxis_title="", title="", height=320)
    else:
        daily = (cust_orders.groupby(cust_orders["_date"].dt.date)
                 .agg(orders=("order_id","nunique")))
        fig_hist = px.line(daily, y="orders", template="plotly_white")
        fig_hist.update_layout(yaxis_title="Số đơn", xaxis_title="", title="", height=320)
    fig_hist.update_layout(margin=dict(l=10,r=10,t=10,b=4), showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("Không đủ cột (date) hoặc không có dữ liệu đơn hàng.")

# ================= Preferences Boxes =================
def icon_for_category(cat: str) -> str:
    CATEGORY_ICONS = {
        "Beverages":"🥤","Drink":"🥤","Food":"🍱","Snack":"🍪","Personal Care":"🧴",
        "Cosmetics":"💄","Beauty":"💄","Household":"🏠","Home":"🏠","Electronics":"🔌",
        "Device":"🔌","Fashion":"👕","Apparel":"👕","Health":"💊","Book":"📚",
        "Sports":"🏃","Baby":"🍼","Pet":"🐾"
    }
    if not isinstance(cat,str) or not cat.strip():
        return "📦"
    lower = cat.lower()
    for k, ic in CATEGORY_ICONS.items():
        if k.lower() in lower:
            return ic
    return "📦"

def extract_customer_preferences(cust_orders: pd.DataFrame, top_n: int = 5):
    if cust_orders is None or cust_orders.empty:
        return [], []
    prod_col = next((c for c in ["product_name","sku_name","product_title","product_id"] if c in cust_orders.columns), None)
    cat_col = next((c for c in ["category","category_name","department","cat_name"] if c in cust_orders.columns), None)
    top_products=[]; top_categories=[]
    if prod_col:
        prod_rank = cust_orders.groupby(prod_col).size().sort_values(ascending=False).head(top_n)
        top_products = list(map(str, prod_rank.index))
    if cat_col:
        cat_rank = cust_orders.groupby(cat_col).size().sort_values(ascending=False).head(top_n)
        top_categories = list(map(str, cat_rank.index))
    return top_products, top_categories

top_products, top_categories = extract_customer_preferences(cust_orders, top_n=6)
combo_lines=[]
if combo_recs:
    seen_pairs=set()
    for r_ in combo_recs:
        a=r_["antecedent"]; b=r_["consequent"]
        if (a,b) not in seen_pairs:
            seen_pairs.add((a,b))
            combo_lines.append(f"<li>{a} + {b}</li>")

st.markdown(
    "<div class='triple-wrapper'>"
    "<div class='pref-box'>"
    "<h5>Ngành hàng ưa thích</h5>"
    + ("<ul>" + "".join(f"<li>{icon_for_category(c)} {c}</li>" for c in top_categories) + "</ul>" if top_categories else "<p>(Không đủ dữ liệu)</p>")
    + "</div>"
    "<div class='pref-box'>"
    "<h5>Sản phẩm thường mua</h5>"
    + ("<ul>" + "".join(f"<li>{p}</li>" for p in top_products) + "</ul>" if top_products else "<p>(Không đủ dữ liệu)</p>")
    + "</div>"
    "<div class='pref-box'>"
    "<h5>Sản phẩm gợi ý (Combo)</h5>"
    + ("<ul class='combo-list'>" + "".join(combo_lines) + "</ul>" if combo_lines else "<p>(Chưa có gợi ý)</p>")
    + "</div>"
    "</div>",
    unsafe_allow_html=True
)

# ================= KPI Tooltips (VIETNAMESE UPDATED) =================
KPI_TOOLTIPS = {
    "CLV": "Customer Lifetime Value: Giá trị vòng đời ước tính (tổng lợi nhuận gộp kỳ vọng trừ chi phí phục vụ).",
    "AOV": "Average Order Value: Giá trị đơn hàng trung bình = Doanh thu / Số đơn.",
    "Frequency": "Tần suất mua bình quân (số đơn / khách trong kỳ).",
    "Retention": "Tỷ lệ khách vẫn phát sinh mua ở kỳ tiếp theo (hoặc trong cửa sổ N ngày).",
    "Second Purchase Rate": "Tỷ lệ khách mới có đơn thứ 2 trong ≤30 ngày kể từ đơn đầu.",
    "Upsell Rate": "Tỷ lệ đơn có giá trị > ngưỡng chuẩn (ví dụ > median AOV phân khúc).",
    "Cross-sell Rate": "Tỷ lệ đơn chứa ≥2 ngành hàng hoặc sản phẩm bổ trợ.",
    "Referral Rate": "Tỷ lệ khách tạo ít nhất 1 giới thiệu thành công (referral có đơn).",
    "Referral": "Tương đương Referral Rate: khách tạo ít nhất 1 referral chuyển đổi.",
    "Reactivation Rate": "Tỷ lệ khách ngủ đông (không mua X ngày) quay lại mua trong kỳ.",
    "Open Rate": "Tỷ lệ mở kênh (email / push) = Lượt mở / Lượt gửi hợp lệ.",
    "Return Purchase": "Tỷ lệ khách quay lại có thêm ≥1 đơn sau lần mua gần nhất trong khung quan sát.",
    "Repeat Rate": "Tỷ lệ khách có ≥2 đơn trong kỳ / tổng khách phát sinh đơn.",
    "Activation": "Tỷ lệ khách mới đạt điều kiện kích hoạt (ví dụ ≥2 đơn trong 30 ngày đầu).",
    "Repeat Cycle Time": "Thời gian trung bình (ngày) giữa 2 đơn liên tiếp (chu kỳ lặp lại).",
    "Onboarding Completion": "Tỷ lệ khách mới hoàn tất chuỗi onboarding (mở đủ bước hoặc thực hiện hành vi chính).",
    "Data Completeness": "Mức độ đầy đủ dữ liệu: % trường quan trọng có giá trị hợp lệ.",
    "Monetary": "Tổng chi tiêu (doanh thu) của khách trong kỳ phân tích."
}

CAT_TOOLTIPS = {
    "Reactivation":"Kích hoạt lại khách rời bỏ / ngủ đông",
    "Onboarding":"Đưa khách mới tới đơn thứ 2 nhanh",
    "Monetize":"Tăng giá trị đơn & CLV",
    "Growth":"Mở rộng tệp qua referral",
    "Retention":"Giữ chân, giảm churn",
    "General":"Khác / hỗ trợ"
}
RISK_TIP = "Các tín hiệu cần giám sát."
PRIORITY_TIP = "90+ rất cao; 80–89 cao; 60–79 trung bình; <60 hỗ trợ."

# ================= Strategy Renderer =================
def render_strategy_box(plan):
    tactic_html=[]
    for t in plan["tactics"]:
        cat=t["category"]; pr=t["priority"]
        cat_tip = CAT_TOOLTIPS.get(cat,"")
        tactic_html.append(
            f"<li style='margin-bottom:6px;'>{t['tactic']} "
            f"<span class='priority-badge pill' data-tip='{PRIORITY_TIP}'>{pr}</span>"
            f" <span class='pill cat-{cat}' data-tip='{cat_tip}'>{cat}</span></li>"
        )
    kpi_html=" ".join(
        f"<span class='pill' data-tip='{KPI_TOOLTIPS.get(k,k)}'>{k}</span>"
        for k in plan["kpis"]
    )
    risk_html=" ".join(f"<span class='pill risk-pill' data-tip='{RISK_TIP}'>{r}</span>" for r in plan["risk_signals"])
    notes_html=""
    if plan.get("notes"):
        notes_html="<p><i>Notes: "+ " | ".join(plan["notes"]) + "</i></p>"
    return f"""
    <div class='blue-box' style="margin-bottom:14px;">
      <h4>Chiến lược & Gợi ý Cá nhân</h4>
      <p><b>Goal:</b> {plan['goal']}</p>
      <p><b>KPIs:</b> {kpi_html if kpi_html else '—'}</p>
      <p><b>Nguy cơ theo dõi:</b> {risk_html if risk_html else '—'}</p>
      <p><b>Đường nâng cấp:</b> {plan['upgrade_path']}</p>
      <p style="margin-bottom:4px;"><b>Chiến thuật ưu tiên:</b></p>
      <ul style="margin-top:0; padding-left:20px;">{''.join(tactic_html)}</ul>
      {notes_html}
      <p class='note-hover'><i>Rê chuột vào pill để xem chú thích.</i></p>
    </div>
    """
st.markdown(render_strategy_box(personalized_plan), unsafe_allow_html=True)

# ================= So sánh cụm =================
st.markdown("### So sánh với Trung bình Cụm")
cl_profile_row = fetch_cluster_row(profile_df, cluster_id) if (cluster_id is not None and pd.notna(cluster_id)) else None
if cl_profile_row is not None:
    compare = pd.DataFrame({
        "Metric":["Recency","Frequency","Monetary"],
        "ClusterMean":[cl_profile_row.get("Recency_mean"),
                       cl_profile_row.get("Frequency_mean"),
                       cl_profile_row.get("Monetary_mean")],
        "Customer":[row["Recency"],row["Frequency"],row["Monetary"]]
    })
    long_cmp = compare.melt(id_vars="Metric", value_vars=["ClusterMean","Customer"],
                            var_name="Type", value_name="Value")
    name_map = {"Recency":"Recency (days ↓)","Frequency":"Frequency","Monetary":"Monetary"}
    long_cmp["MetricLabel"] = long_cmp["Metric"].map(name_map)
    fig_group = px.bar(long_cmp, x="MetricLabel", y="Value", color="Type",
                       barmode="group", text="Value", template="plotly_white",
                       title="Customer vs Cluster Mean")
    fig_group.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_group.update_layout(yaxis_title="Value", legend_title="")
    st.plotly_chart(fig_group, use_container_width=True)

    compare["DiffPctRaw"] = (compare["Customer"] - compare["ClusterMean"])/(compare["ClusterMean"]+1e-9)*100
    def adj(rw):
        return -rw["DiffPctRaw"] if rw["Metric"]=="Recency" else rw["DiffPctRaw"]
    compare["DiffPctAdj"] = compare.apply(adj, axis=1)
    compare["Direction"] = np.where(compare["DiffPctAdj"]>=0,"Better / Higher","Worse / Lower")
    fig_diff = px.bar(
        compare, x="DiffPctAdj", y="Metric", color="Direction", orientation="h",
        text=compare["DiffPctAdj"].map(lambda v: f"{v:+.1f}%"),
        template="plotly_white", title="Chênh lệch % so với Cụm (Recency đảo dấu)"
    )
    fig_diff.add_vline(x=0, line_color="#666", line_dash="dash")
    fig_diff.update_layout(xaxis_title="Adj % Difference (Positive = Better)", yaxis_title="")
    st.plotly_chart(fig_diff, use_container_width=True)

    with st.expander("Chi tiết so sánh"):
        st.dataframe(compare[["Metric","ClusterMean","Customer","DiffPctRaw","DiffPctAdj"]].round(3))
else:
    st.info("Không đủ thông tin cụm để so sánh.")

# ================= Order Detail =================
with st.expander("Chi tiết đơn hàng (top 50 gần nhất)"):
    if not cust_orders.empty and "date" in cust_orders.columns:
        cust_orders = cust_orders.sort_values("date", ascending=False)
    st.dataframe(cust_orders.head(50))

st.markdown("<div style='text-align:left; color:#666; font-size:13px; margin-top:30px;'>© 2025 Đồ án tốt nghiệp lớp DL07_K306 - RFM Segmentation - Nhóm J</div>", unsafe_allow_html=True)
