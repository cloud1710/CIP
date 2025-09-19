import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math
from datetime import timedelta

st.set_page_config(
    page_title="EDA Đơn hàng, Sản phẩm & Khách hàng",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CSS ==================
st.markdown("""
<style>
.analysis-box {
    padding: 0.9rem 1rem;
    border-left: 4px solid #4b8bec;
    background: #f0f7ff;
    border-radius: 4px;
    font-size: 0.9rem;
    line-height: 1.4rem;
}
.analysis-box h4 {
    margin-top: 0;
    margin-bottom: 0.4rem;
    font-size: 1rem;
}
.strategy-box {
    padding: 0.9rem 1rem;
    border-left: 4px solid #d27d2d;
    background: #fff5eb;
    border-radius: 4px;
    font-size: 0.9rem;
    line-height: 1.35rem;
}
.strategy-box h4 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}
.weekday-box {
    padding: 0.75rem 1rem;
    border-left: 4px solid #7b4ded;
    background:#f4f0ff;
    border-radius:6px;
    font-size:0.82rem;
    line-height:1.3rem;
}
.abc-box {
    padding: 0.75rem 1rem;
    background:#fff7e6;
    border:1px solid #f0d28a;
    border-radius:6px;
    font-size:0.85rem;
    line-height:1.3rem;
    margin-top:0.75rem;
    margin-bottom:0.75rem;
}
.info-box {
    padding:0.65rem 0.9rem;
    background:#f5f9ff;
    border:1px solid #c8dcf5;
    border-radius:6px;
    font-size:0.8rem;
    line-height:1.25rem;
    margin-top:0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ================== HÀM TIỆN ÍCH ==================
def style_big_number(v):
    if v >= 1e9: return f"{v/1e9:.2f}B"
    if v >= 1e6: return f"{v/1e6:.2f}M"
    if v >= 1e3: return f"{v/1e3:.1f}K"
    return f"{v:.0f}"

def rename_for_display(df: pd.DataFrame, mapping: dict):
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

@st.cache_data(show_spinner=True, ttl=3600)
def load_data(path: str):
    return pd.read_csv(path)

# ================== HÀM TÍNH CHỈ SỐ ==================
@st.cache_data(show_spinner=False)
def compute_daily_metrics(df: pd.DataFrame):
    return (df.groupby("order_date")
              .agg(orders=("order_id","nunique"),
                   revenue=("order_value","sum"),
                   customers=("customer_id","nunique"))
              .reset_index())

@st.cache_data(show_spinner=False)
def compute_category_metrics(df: pd.DataFrame, top_n=15):
    cat = (df.groupby("category")
           .agg(orders=("order_id","nunique"),
                revenue=("order_value","sum"),
                customers=("customer_id","nunique"),
                quantity=("quantity","sum"))
           .reset_index()
           .sort_values("revenue", ascending=False))
    cat["revenue_share_%"] = cat["revenue"] / cat["revenue"].sum() * 100
    if len(cat) > top_n:
        others = cat.iloc[top_n:].copy()
        top = cat.iloc[:top_n].copy()
        row_other = {
            "category": "OTHERS",
            "orders": others["orders"].sum(),
            "revenue": others["revenue"].sum(),
            "customers": others["customers"].sum(),
            "quantity": others["quantity"].sum(),
            "revenue_share_%": others["revenue"].sum() / cat["revenue"].sum() * 100
        }
        top = pd.concat([top, pd.DataFrame([row_other])], ignore_index=True)
        return top
    return cat

@st.cache_data(show_spinner=False)
def compute_product_metrics(df: pd.DataFrame, top_n=20):
    if "product_id" not in df.columns:
        return pd.DataFrame()
    prod = (df.groupby(["product_id","product_name"])
            .agg(orders=("order_id","nunique"),
                 revenue=("order_value","sum"),
                 quantity=("quantity","sum"),
                 customers=("customer_id","nunique"))
            .reset_index())
    prod["rev_per_order"] = prod["revenue"] / prod["orders"].replace(0, np.nan)
    return prod.sort_values("revenue", ascending=False).head(top_n)

@st.cache_data(show_spinner=False)
def compute_customer_metrics(df: pd.DataFrame):
    cust = (df.groupby("customer_id")
            .agg(orders=("order_id","nunique"),
                 revenue=("order_value","sum"),
                 first_date=("order_date","min"),
                 last_date=("order_date","max"))
            .reset_index())
    cust["lifetime_days"] = (cust["last_date"] - cust["first_date"]).dt.days.clip(lower=0)
    cust["avg_order_value_customer"] = cust["revenue"] / cust["orders"].replace(0, np.nan)
    return cust

@st.cache_data(show_spinner=False)
def compute_weekday_summary(df: pd.DataFrame):
    t = df.copy()
    t["weekday"] = t["order_date"].dt.weekday
    return (t.groupby("weekday")
              .agg(revenue=("order_value","sum"),
                   orders=("order_id","nunique"))
              .reset_index())

@st.cache_data(show_spinner=True)
def compute_product_advanced(df: pd.DataFrame):
    if "product_id" not in df.columns:
        return pd.DataFrame(), 0
    min_date = df["order_date"].min()
    max_date = df["order_date"].max()
    observed_days_total = (max_date - min_date).days + 1

    daily_prod = (df.groupby(["product_id","product_name","order_date"])
                    .agg(daily_units=("quantity","sum"),
                         daily_revenue=("order_value","sum"))
                    .reset_index())

    base = (df.groupby(["product_id","product_name"])
              .agg(first_sale=("order_date","min"),
                   last_sale=("order_date","max"),
                   total_units=("quantity","sum"),
                   total_revenue=("order_value","sum"),
                   customers_total=("customer_id","nunique"))
              .reset_index())

    active_days_df = (daily_prod.groupby("product_id")["order_date"]
                      .nunique()
                      .reset_index()
                      .rename(columns={"order_date":"active_days"}))
    base = base.merge(active_days_df, on="product_id", how="left")

    stats_list = []
    for pid, grp in daily_prod.groupby("product_id"):
        vals = grp["daily_units"].values.astype(float)
        sum_units = vals.sum()
        mean_all = sum_units / observed_days_total if observed_days_total > 0 else 0
        if observed_days_total > 1:
            sum_sq = np.sum(vals**2)
            variance = (sum_sq - (sum_units**2)/observed_days_total) / (observed_days_total - 1)
            variance = max(variance, 0)
            std_all = math.sqrt(variance)
        else:
            std_all = 0
        demand_cv = std_all / mean_all if mean_all > 0 else np.nan
        stats_list.append({
            "product_id": pid,
            "mean_daily_all": mean_all,
            "std_daily_all": std_all,
            "demand_cv": demand_cv
        })
    stats_df = pd.DataFrame(stats_list)
    base = base.merge(stats_df, on="product_id", how="left")

    rep_records = []
    pud = df.groupby(["product_id","customer_id"])["order_date"].apply(lambda s: sorted(set(s))).reset_index()
    for pid, sub in pud.groupby("product_id"):
        diffs_all = []
        repeat_customers = 0
        for _, row in sub.iterrows():
            dates = row["order_date"]
            if len(dates) >= 2:
                repeat_customers += 1
                diffs = np.diff(dates)
                diffs_days = [d.days for d in diffs if d.days >= 0]
                diffs_all.extend(diffs_days)
        if diffs_all:
            inter_median = float(np.median(diffs_all))
            inter_mean = float(np.mean(diffs_all))
        else:
            inter_median = np.nan
            inter_mean = np.nan
        rep_records.append({
            "product_id": pid,
            "customers_repeat": repeat_customers,
            "interpurchase_median_days": inter_median,
            "interpurchase_mean_days": inter_mean
        })
    rep_df = pd.DataFrame(rep_records)
    base = base.merge(rep_df, on="product_id", how="left")

    base["lifecycle_days"] = (base["last_sale"] - base["first_sale"]).dt.days + 1
    base["days_since_last"] = (max_date - base["last_sale"]).dt.days
    base["coverage"] = base["active_days"] / observed_days_total
    base["repeat_rate_product"] = base["customers_repeat"] / base["customers_total"].replace(0, np.nan)
    base["repeat_rate_product"] = base["repeat_rate_product"].fillna(0)

    def classify_variability(cv):
        if pd.isna(cv): return "Thiếu dữ liệu"
        if cv < 0.5: return "Ổn định"
        if cv < 1: return "Biến động vừa"
        return "Biến động cao"
    base["variability_class"] = base["demand_cv"].apply(classify_variability)

    base = base.sort_values("total_revenue", ascending=False)
    base["cum_revenue"] = base["total_revenue"].cumsum()
    total_rev_all = base["total_revenue"].sum()
    base["cum_share"] = base["cum_revenue"] / total_rev_all * 100

    def abc_label(cshare):
        if cshare <= 80: return "A"
        if cshare <= 95: return "B"
        return "C"
    base["abc_class"] = base["cum_share"].apply(abc_label)

    cols = [
        "product_id","product_name","first_sale","last_sale","lifecycle_days",
        "days_since_last","active_days","coverage","total_units","total_revenue",
        "customers_total","customers_repeat","repeat_rate_product",
        "mean_daily_all","std_daily_all","demand_cv","variability_class",
        "interpurchase_median_days","interpurchase_mean_days","abc_class","cum_share"
    ]
    for c in cols:
        if c not in base.columns: base[c] = np.nan
    return base[cols], observed_days_total

# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Bộ lọc")
data_path = st.sidebar.text_input("Đường dẫn file CSV", "data/orders_full.csv")
allow_upload = st.sidebar.checkbox("Upload file khác?", False)
up = st.sidebar.file_uploader("Chọn CSV", type=["csv"]) if allow_upload else None
if st.sidebar.button("↻ Xóa cache"):
    st.cache_data.clear()

# ================== LOAD & CHUẨN HÓA ==================
if up is not None:
    raw_df = load_data(up)
    source_label = "Uploaded"
else:
    try:
        raw_df = load_data(data_path)
        source_label = data_path
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        st.stop()

expected = ["member_number","order_id","date","product_id","product_name","category","items","price","gross_sales"]
missing = [c for c in expected if c not in raw_df.columns]
if missing:
    st.error(f"Thiếu cột bắt buộc: {missing}")
    st.stop()

df = raw_df.copy()
df["order_date"] = pd.to_datetime(df["date"], errors="coerce")
if df["order_date"].isna().all():
    st.error("Không parse được cột ngày (date).")
    st.stop()
raw_df["_date_parsed"] = pd.to_datetime(raw_df["date"], errors="coerce")

df["customer_id"] = df["member_number"]
df["quantity"] = df["items"]
df["unit_price"] = df["price"]
df["order_value"] = df["gross_sales"]

# Lọc thời gian
min_date = df["order_date"].min()
max_date = df["order_date"].max()
date_range = st.sidebar.date_input(
    "Khoảng ngày",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range
mask = (df["order_date"] >= pd.to_datetime(start_date)) & (df["order_date"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
df = df.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"Nguồn: {source_label}")
st.sidebar.markdown(f"Số dòng: **{len(df)}**")

if df.empty:
    st.warning("Không có dữ liệu sau khi lọc.")
    st.stop()

# ================== KPI ==================
st.title("📊 Phân tích đơn hàng, sản phẩm và khách hàng")
total_revenue = df["order_value"].sum()
total_orders = df["order_id"].nunique()
total_customers = df["customer_id"].nunique()
aov = total_revenue / total_orders if total_orders else 0
cust_order_counts = df.groupby("customer_id")["order_id"].nunique()
repeat_rate = cust_order_counts[cust_order_counts >= 2].count() / total_customers * 100 if total_customers else 0
rev_per_customer = total_revenue / total_customers if total_customers else 0

help_text = {
    "Doanh thu": "Tổng order_value (gross_sales) trong khoảng thời gian đã lọc.",
    "Số đơn": "Số order_id duy nhất trong khoảng thời gian đang xem.",
    "Số khách": "Số customer_id duy nhất có phát sinh ít nhất 1 đơn.",
    "Giá trị TB/Đơn": "Average Order Value = Doanh thu / Số đơn.",
    "Tỷ lệ mua lại": "% khách có từ 2 đơn trở lên (khách mua ≥2 / tổng khách * 100).",
    "Doanh thu TB/Khách": "Doanh thu trung bình trên mỗi khách = Doanh thu / Số khách."
}

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Doanh thu", style_big_number(total_revenue), help=help_text["Doanh thu"])
c2.metric("Số đơn", style_big_number(total_orders), help=help_text["Số đơn"])
c3.metric("Số khách", style_big_number(total_customers), help=help_text["Số khách"])
c4.metric("Giá trị TB/Đơn", f"{aov:,.0f}", help=help_text["Giá trị TB/Đơn"])
c5.metric("Tỷ lệ mua lại", f"{repeat_rate:.1f}%", help=help_text["Tỷ lệ mua lại"])
c6.metric("Doanh thu TB/Khách", f"{rev_per_customer:,.0f}", help=help_text["Doanh thu TB/Khách"])

st.markdown("---")

# ================== TABS (ĐÃ ĐỔI THỨ TỰ: Heatmap SAU Phân tích nâng cao) ==================
tab_overview, tab_category, tab_product, tab_customers, tab_adv_product, tab_heatmap, tab_download = st.tabs([
    "📈 Tổng quan",
    "🏷️ Ngành hàng",
    "📦 Sản phẩm",
    "👥 Khách hàng",
    "🔬 Phân tích nâng cao",
    "🗓️ Heatmap tuần",
    "⬇️ Xuất dữ liệu"
])

# -------- Tổng quan ----------
with tab_overview:
    st.subheader("Xu hướng theo thời gian")
    gran = st.selectbox("Độ chi tiết", ["Ngày","Tuần","Tháng"], key="gran_main")
    daily = compute_daily_metrics(df).sort_values("order_date")
    if gran == "Ngày":
        plot_df = daily
    elif gran == "Tuần":
        plot_df = (daily.assign(week=lambda d: d["order_date"].dt.to_period("W").apply(lambda r: r.start_time))
                   .groupby("week")
                   .agg(orders=("orders","sum"),
                        revenue=("revenue","sum"),
                        customers=("customers","sum"))
                   .reset_index()
                   .rename(columns={"week":"order_date"}))
    else:
        plot_df = (daily.assign(month=lambda d: d["order_date"].dt.to_period("M").dt.to_timestamp())
                   .groupby("month")
                   .agg(orders=("orders","sum"),
                        revenue=("revenue","sum"),
                        customers=("customers","sum"))
                   .reset_index()
                   .rename(columns={"month":"order_date"}))
    plot_df_vn = rename_for_display(plot_df, {
        "order_date":"Ngày","revenue":"Doanh thu","orders":"Số đơn","customers":"Số khách"
    })
    c_rev = (alt.Chart(plot_df_vn)
             .mark_line(point=True)
             .encode(
                 x=alt.X("Ngày:T", title="Ngày"),
                 y=alt.Y("Doanh thu:Q", title="Doanh thu"),
                 tooltip=["Ngày:T","Doanh thu:Q","Số đơn:Q","Số khách:Q"]
             ).properties(height=300, title="Doanh thu theo thời gian"))
    c_orders = (alt.Chart(plot_df_vn)
                .mark_bar()
                .encode(
                    x=alt.X("Ngày:T", title="Ngày"),
                    y=alt.Y("Số đơn:Q", title="Số đơn"),
                    tooltip=["Ngày:T","Số đơn:Q"]
                ).properties(height=150, title="Số đơn theo thời gian"))
    st.altair_chart(c_rev, use_container_width=True)
    st.altair_chart(c_orders, use_container_width=True)

    # So sánh tăng trưởng giữa 2 năm
    st.markdown("### So sánh tăng trưởng giữa 2 năm")
    years_available = sorted(df["order_date"].dt.year.unique())
    if len(years_available) >= 2:
        default_curr = years_available[-1]
        default_prev = years_available[-2]
        coly1, coly2, coly3 = st.columns([1,1,2])
        with coly1:
            year_curr = st.selectbox("Năm hiện tại", years_available, index=years_available.index(default_curr))
        with coly2:
            year_prev = st.selectbox("Năm so sánh", [y for y in years_available if y != year_curr],
                                     index=0 if default_prev != year_curr else 0)
        with coly3:
            metric_choice = st.selectbox("Chỉ tiêu", ["Doanh thu","Số đơn","Số khách"])
        metric_map = {"Doanh thu":"revenue","Số đơn":"orders","Số khách":"customers"}
        metric_col = metric_map[metric_choice]

        month_agg = (daily.assign(year=daily["order_date"].dt.year,
                                  month=daily["order_date"].dt.month)
                           .groupby(["year","month"])
                           .agg(orders=("orders","sum"),
                                revenue=("revenue","sum"),
                                customers=("customers","sum"))
                           .reset_index())
        subset = month_agg[month_agg["year"].isin([year_prev, year_curr])].copy()
        if subset.empty:
            st.info("Không đủ dữ liệu cho 2 năm đã chọn.")
        else:
            subset["Năm"] = subset["year"].astype(str)
            subset["Tháng"] = subset["month"].astype(int).astype(str)
            plot_metric_df = subset[["Năm","Tháng", metric_col]].rename(columns={metric_col:"Giá trị"})
            line_yoy = ( alt.Chart(plot_metric_df)
                         .mark_line(point=True)
                         .encode(
                             x=alt.X("Tháng:O", sort=None, title="Tháng"),
                             y=alt.Y("Giá trị:Q", title=metric_choice),
                             color=alt.Color("Năm:N"),
                             tooltip=["Năm","Tháng","Giá trị:Q"]
                         ).properties(height=300, title=f"{metric_choice} theo tháng - So sánh {year_prev} vs {year_curr}") )
            pivot = plot_metric_df.pivot(index="Tháng", columns="Năm", values="Giá trị").reset_index()
            if str(year_prev) in pivot.columns and str(year_curr) in pivot.columns:
                pivot["Tăng trưởng %"] = (pivot[str(year_curr)] - pivot[str(year_prev)]) / pivot[str(year_prev)].replace(0, np.nan) * 100
                yoy_bar = (alt.Chart(pivot)
                           .mark_bar()
                           .encode(
                               x=alt.X("Tháng:O", title="Tháng"),
                               y=alt.Y("Tăng trưởng %:Q", title="Tăng trưởng %"),
                               tooltip=["Tháng","Tăng trưởng %:Q"]
                           ).properties(height=220, title="Tăng trưởng % theo tháng (YoY)"))
                st.altair_chart(line_yoy, use_container_width=True)
                st.altair_chart(yoy_bar, use_container_width=True)

                total_prev = pivot[str(year_prev)].sum()
                total_curr = pivot[str(year_curr)].sum()
                growth_total = (total_curr - total_prev) / total_prev * 100 if total_prev else np.nan
                best_month = pivot.loc[pivot["Tăng trưởng %"].idxmax()] if pivot["Tăng trưởng %"].notna().any() else None
                worst_month = pivot.loc[pivot["Tăng trưởng %"].idxmin()] if pivot["Tăng trưởng %"].notna().any() else None
                lines_growth = [
                    f"- Tổng {metric_choice} {year_prev}: {total_prev:,.0f}",
                    f"- Tổng {metric_choice} {year_curr}: {total_curr:,.0f}",
                    f"- Tăng trưởng tổng: {growth_total:+.1f}%" if not np.isnan(growth_total) else "- Không tính được tăng trưởng tổng"
                ]
                if best_month is not None and not np.isnan(best_month["Tăng trưởng %"]):
                    lines_growth.append(f"- Tháng tăng mạnh nhất: {best_month['Tháng']} ({best_month['Tăng trưởng %']:+.1f}%)")
                if worst_month is not None and not np.isnan(worst_month["Tăng trưởng %"]):
                    lines_growth.append(f"- Tháng giảm mạnh nhất: {worst_month['Tháng']} ({worst_month['Tăng trưởng %']:+.1f}%)")

                if metric_choice == "Doanh thu":
                    curr_year_data = month_agg[month_agg["year"] == year_curr].copy()
                    if not curr_year_data.empty:
                        top2 = curr_year_data.sort_values("revenue", ascending=False).head(2)
                        if len(top2) == 1:
                            m1 = int(top2.iloc[0]["month"])
                            r1 = top2.iloc[0]["revenue"]
                            lines_growth.append(f"- Top tháng doanh thu {year_curr}: Tháng {m1} ({r1:,.0f})")
                        elif len(top2) == 2:
                            m1 = int(top2.iloc[0]["month"]); r1 = top2.iloc[0]["revenue"]
                            m2 = int(top2.iloc[1]["month"]); r2 = int(top2.iloc[1]["revenue"])
                            lines_growth.append(f"- Top 2 tháng doanh thu {year_curr}: Tháng {m1} ({r1:,.0f}), Tháng {m2} ({r2:,.0f})")

                st.markdown(f"""
                <div class="analysis-box">
                <h4>📌 Tóm tắt tăng trưởng {metric_choice}</h4>
                {'<br>'.join(lines_growth)}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.altair_chart(line_yoy, use_container_width=True)
                st.info("Thiếu dữ liệu để tính YoY %.")          
    else:
        st.info("Dữ liệu chưa đủ 2 năm để so sánh.")

    st.markdown("### Phân bố giá trị đơn hàng")
    ov = df.groupby("order_id")["order_value"].sum().reset_index().rename(columns={"order_value":"Giá trị đơn"})
    hist = (alt.Chart(ov)
            .mark_bar()
            .encode(
                alt.X("Giá trị đơn:Q", bin=alt.Bin(maxbins=40), title="Giá trị đơn"),
                y=alt.Y("count():Q", title="Số đơn"),
                tooltip=[alt.Tooltip("count():Q", title="Số đơn")]
            ).properties(height=250, title="Phân bố giá trị đơn hàng"))
    st.altair_chart(hist, use_container_width=True)

    # -------- Phân tích nhanh & Gợi ý chiến lược song song ----------
    try:
        num_days = (df["order_date"].max() - df["order_date"].min()).days + 1
        daily = daily  # đã có
        avg_rev_day = daily["revenue"].mean()
        peak_row = daily.loc[daily["revenue"].idxmax()]
        peak_date, peak_rev = peak_row["order_date"], peak_row["revenue"]
        top7_share = (daily.sort_values("revenue", ascending=False).head(7)["revenue"].sum() /
                      daily["revenue"].sum() * 100 if len(daily) >= 7 else np.nan)
        median_order_value = ov["Giá trị đơn"].median()
        p90_order_value = ov["Giá trị đơn"].quantile(0.90)
        start_current = df["order_date"].min()
        end_current = df["order_date"].max()
        period_len = (end_current - start_current).days + 1
        prev_start2 = start_current - timedelta(days=period_len)
        prev_end2 = start_current - timedelta(days=1)
        raw_prev_mask = (raw_df["_date_parsed"] >= prev_start2) & (raw_df["_date_parsed"] <= prev_end2)
        prev_df = raw_df.loc[raw_prev_mask].copy()
        prev_df["order_value_prev"] = prev_df["gross_sales"]
        prev_revenue = prev_df["order_value_prev"].sum() if not prev_df.empty else np.nan
        growth_pct = (total_revenue - prev_revenue) / prev_revenue * 100 if (not np.isnan(prev_revenue) and prev_revenue > 0) else np.nan
        mean_ov = ov["Giá trị đơn"].mean()
        std_ov = ov["Giá trị đơn"].std()
        cv_ov = std_ov / mean_ov if mean_ov > 0 else np.nan
        p90_median_ratio = (p90_order_value / median_order_value) if median_order_value > 0 else np.nan

        # --- Nội dung phân tích (fact) ---
        lines = [
            f"- Khoảng ngày: {start_current.date()} → {end_current.date()} ({num_days} ngày).",
            f"- Doanh thu bình quân/ngày: {avg_rev_day:,.0f}.",
            f"- Ngày cao nhất: {peak_date.date()} (doanh thu {peak_rev:,.0f}).",
            f"- Giá trị đơn trung vị: {median_order_value:,.0f}; P90: {p90_order_value:,.0f}."
        ]
        if not np.isnan(top7_share):
            lines.append(f"- Top 7 ngày chiếm {top7_share:.1f}% doanh thu.")
        if not np.isnan(cv_ov):
            lines.append(f"- CV giá trị đơn: {cv_ov:.2f}.")
        if not np.isnan(growth_pct):
            lines.append(f"- Tăng trưởng so kỳ trước: {growth_pct:+.1f}% (Trước: {prev_revenue:,.0f}).")
        else:
            lines.append("- Thiếu dữ liệu để so sánh kỳ trước.")
        if not np.isnan(p90_median_ratio):
            lines.append(f"- P90/Median = {p90_median_ratio:.2f} (độ phân tán giá trị đơn).")

        # --- Sinh gợi ý chiến lược (insight/action) ---
        strategy_lines = []
        # Phát hiện không có spike
        if not np.isnan(top7_share) and top7_share < 25:
            strategy_lines.append(f"Doanh thu phân bổ phẳng (Top 7 ngày chỉ {top7_share:.1f}%): gần như không có spike → tạo 'hero campaign' theo chủ đề để kích cầu (Flash Sale 1 ngày / Payday).")
        elif not np.isnan(top7_share) and top7_share < 35:
            strategy_lines.append(f"Phân bổ khá đều (Top 7 ngày {top7_share:.1f}%). Có thể chủ động tạo 1–2 đỉnh để tối ưu hiệu ứng FOMO.")
        elif not np.isnan(top7_share) and top7_share > 55:
            strategy_lines.append("Doanh thu quá tập trung vào ít ngày → phân tán rủi ro bằng mini-campaign vào các ngày yếu.")

        # Tăng trưởng
        if not np.isnan(growth_pct):
            if growth_pct < 0:
                strategy_lines.append("Giảm tăng trưởng: điều tra funnel (traffic nguồn, tỷ lệ chuyển đổi, repeat) + tái kích hoạt qua email/Zalo workflow.")
            elif growth_pct < 10:
                strategy_lines.append("Tăng trưởng nhẹ: tập trung tối ưu chuyển đổi & nâng AOV trước khi mở rộng chi phí quảng cáo.")
            else:
                strategy_lines.append("Tăng trưởng tốt: chuẩn hoá playbook kênh hiệu quả và scale ngân sách theo CPA mục tiêu.")
        else:
            strategy_lines.append("Chưa có baseline tăng trưởng → giữ log tuần & dựng phân tích hồi cứu để phân tách khách hàng quay lại so với khách hàng mới.")

        # Repeat rate
        if repeat_rate < 20:
            strategy_lines.append("Repeat thấp (<20%): xây chuỗi onboarding + nhắc tái mua canh theo chu kỳ median, thêm incentive lần 2 (voucher giá trị nhỏ).")
        elif repeat_rate < 40:
            strategy_lines.append("Repeat trung bình: triển khai điểm thưởng / bundle ưu đãi để đẩy lần mua thứ 3.")
        else:
            strategy_lines.append("Repeat cao: chuyển trọng tâm sang tăng AOV (bundle cao cấp, gợi ý phụ kiện, referral).")

        # Giá trị đơn thấp + khoảng P90
        if median_order_value < 30 and p90_order_value < 80:
            strategy_lines.append(f"Giá trị đơn thấp (Median {median_order_value:,.0f}, P90 {p90_order_value:,.0f}): thiết kế combo > {int(median_order_value*1.8)} để nâng cấp giỏ; highlight free-ship / ngưỡng quà.")
        elif p90_median_ratio > 2.2:
            strategy_lines.append(f"Khoảng cách P90/Median cao ({p90_median_ratio:.2f}): phân khúc khách high-value tạo ưu đãi riêng (VIP tier / early access).")
        elif p90_median_ratio < 1.5:
            strategy_lines.append("P90/Median thấp: bổ sung sản phẩm premium để mở trần chi tiêu.")

        # Biến động AOV
        if not np.isnan(cv_ov):
            if cv_ov > 1.0:
                strategy_lines.append("CV AOV cao: cá nhân hoá upsell theo segment (giá trị đơn lần gần nhất).")
            elif cv_ov < 0.4:
                strategy_lines.append("CV AOV thấp: dễ dự báo → thử tăng ngưỡng freeship hoặc bundle để kéo AOV lên lớp cao hơn.")

        # AOV vs P90
        if aov and p90_order_value and aov < p90_order_value * 0.45:
            strategy_lines.append("AOV còn xa nhóm khách chi tiêu cao → thêm gợi ý mua kèm ở trang giỏ / sau thanh toán.")

        # Tổng hợp nền tảng
        strategy_lines.append("Phân tích cohort theo tháng nhằm nhận diện tỷ trọng đến từ khách hàng quay lại so với khách hàng mới.")
        strategy_lines.append("Theo dõi tuần các chỉ số: New Customers, Repeat %, AOV, CAC (ngoại bảng).")

        col_analysis, col_strategy = st.columns([4,6])
        with col_analysis:
            st.markdown(f"""
            <div class="analysis-box">
            <h4>📌 Phân tích nhanh</h4>
            {'<br>'.join(lines)}
            </div>
            """, unsafe_allow_html=True)
        with col_strategy:
            st.markdown(f"""
            <div class="strategy-box">
            <h4>🎯 Gợi ý chiến lược tổng thể</h4>
            {'<br>'.join('• ' + s for s in strategy_lines)}
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Không tạo được phân tích tự động (chi tiết: {e})")

# -------- Ngành hàng ----------
with tab_category:
    st.subheader("Ngành hàng")
    top_n_cat = st.slider("Top N (theo doanh thu)", 5, 11, 11, key="top_cat")
    cat_df = compute_category_metrics(df, top_n=top_n_cat)
    cat_vn = rename_for_display(cat_df, {
        "category":"Ngành hàng","revenue":"Doanh thu","orders":"Số đơn",
        "customers":"Số khách","quantity":"Số lượng","revenue_share_%":"Tỷ trọng (%)"
    })
    col1, col2 = st.columns([2,1])
    with col1:
        bar_cat = (alt.Chart(cat_vn)
                   .mark_bar()
                   .encode(
                       x=alt.X("Doanh thu:Q", title="Doanh thu"),
                       y=alt.Y("Ngành hàng:N", sort="-x"),
                       tooltip=["Ngành hàng","Doanh thu","Số đơn","Số khách","Tỷ trọng (%)"]
                   ).properties(height=400, title="Top ngành hàng"))
        st.altair_chart(bar_cat, use_container_width=True)
    with col2:
        pie = (alt.Chart(cat_vn)
               .mark_arc()
               .encode(
                   theta=alt.Theta("Doanh thu:Q"),
                   color=alt.Color("Ngành hàng:N"),
                   tooltip=["Ngành hàng","Doanh thu","Tỷ trọng (%)"]
               ).properties(title="Tỷ trọng doanh thu"))
        st.altair_chart(pie, use_container_width=True)
    st.dataframe(cat_vn.style.format({
        "Doanh thu":"{:,.0f}","Tỷ trọng (%)":"{:,.2f}","Số lượng":"{:,.0f}"
    }), use_container_width=True)

# -------- Sản phẩm ----------
with tab_product:
    st.subheader("Sản phẩm")
    if "product_id" not in df.columns:
        st.info("Thiếu product_id.")
    else:
        top_n_prod = st.slider("Top N sản phẩm", 5, 100, 38, key="top_prod")
        prod_df = compute_product_metrics(df, top_n=top_n_prod)
        if not prod_df.empty:
            prod_vn = rename_for_display(prod_df, {
                "product_id":"Mã SP","product_name":"Tên SP","revenue":"Doanh thu",
                "orders":"Số đơn","quantity":"Số lượng","customers":"Số khách",
                "rev_per_order":"Doanh thu/Đơn"
            })
            bar_prod = (alt.Chart(prod_vn)
                        .mark_bar()
                        .encode(
                            x=alt.X("Doanh thu:Q", title="Doanh thu"),
                            y=alt.Y("Tên SP:N", sort="-x"),
                            tooltip=["Mã SP","Tên SP","Doanh thu","Số đơn","Số lượng","Doanh thu/Đơn"]
                        ).properties(height=500, title="Top sản phẩm"))
            st.altair_chart(bar_prod, use_container_width=True)
            st.dataframe(prod_vn.style.format({
                "Doanh thu":"{:,.0f}","Doanh thu/Đơn":"{:,.0f}","Số lượng":"{:,.0f}"
            }), use_container_width=True)
        else:
            st.warning("Không có dữ liệu sản phẩm.")

# -------- Khách hàng ----------
with tab_customers:
    st.subheader("Khách hàng")
    cust_full = compute_customer_metrics(df)
    if cust_full.empty:
        st.warning("Không có dữ liệu khách hàng.")
    else:
        top_n_cust = st.slider("Top N khách hàng (theo Số đơn)", 5, 300, 10, key="top_cust_v3")
        top_cust = cust_full.sort_values("orders", ascending=False).head(top_n_cust).copy()

        mapping = {
            "customer_id": "Mã KH",
            "orders": "Số đơn",
            "revenue": "Doanh thu",
            "avg_order_value_customer": "Giá trị TB/Đơn (KH)",
            "first_date": "Ngày đầu",
            "last_date": "Ngày cuối",
            "lifetime_days": "Số ngày vòng đời"
        }
        top_vn = rename_for_display(top_cust, mapping)

        st.markdown("#### Top khách hàng mua sắm nhiều nhất (theo Số đơn)")
        chart_top = (alt.Chart(top_vn)
                     .mark_bar()
                     .encode(
                         x=alt.X("Số đơn:Q", title="Số đơn", axis=alt.Axis(tickMinStep=1, format='d')),
                         y=alt.Y("Mã KH:N", sort="-x"),
                         tooltip=["Mã KH","Số đơn","Doanh thu","Giá trị TB/Đơn (KH)"]
                     ).properties(height=400))
        st.altair_chart(chart_top, use_container_width=True)

        st.dataframe(top_vn.style.format({
            "Số đơn":"{:,.0f}",
            "Doanh thu":"{:,.0f}",
            "Giá trị TB/Đơn (KH)":"{:,.0f}"
        }), use_container_width=True)

        st.markdown("#### Phân bố Số khách hàng / Số đơn (Histogram nằm ngang)")
        freq_dist = (cust_full.groupby("orders")
                                .size()
                                .reset_index(name="num_customers")
                                .sort_values("orders"))
        if not freq_dist.empty:
            min_o = int(freq_dist["orders"].min())
            max_o = int(freq_dist["orders"].max())
            full_range = pd.DataFrame({"orders": list(range(min_o, max_o + 1))})
            freq_dist = full_range.merge(freq_dist, on="orders", how="left").fillna({"num_customers": 0})
        freq_dist["orders"] = freq_dist["orders"].astype(int)
        total_customers_all = freq_dist["num_customers"].sum()
        freq_dist["share_%"] = np.where(total_customers_all > 0,
                                        freq_dist["num_customers"] / total_customers_all * 100, 0.0)
        freq_dist["cum_share_%"] = freq_dist["share_%"].cumsum()

        freq_vn = freq_dist.rename(columns={
            "orders":"Số đơn",
            "num_customers":"Số KH",
            "share_%":"Tỷ trọng (%)",
            "cum_share_%":"Tỷ trọng lũy kế (%)"
        })

        chart_dist = (alt.Chart(freq_vn)
                      .mark_bar()
                      .encode(
                          y=alt.Y("Số đơn:O", title="Số đơn", sort=None),
                          x=alt.X("Số KH:Q", title="Số khách"),
                          tooltip=["Số đơn","Số KH","Tỷ trọng (%)","Tỷ trọng lũy kế (%)"]
                      ).properties(height=400, title="Histogram ngang: Số KH theo Số đơn"))
        st.altair_chart(chart_dist, use_container_width=True)

        st.dataframe(
            freq_vn.style.format({
                "Số KH":"{:,.0f}",
                "Tỷ trọng (%)":"{:,.2f}",
                "Tỷ trọng lũy kế (%)":"{:,.2f}"
            }),
            use_container_width=True
        )

        st.markdown("""
        <div class="info-box">
        <b>Diễn giải</b><br>
        - Trục dọc: Số đơn (1,2,3...). Trục ngang: số lượng khách tương ứng.<br>
        - Bin trống hiển thị = 0 giúp thấy khoảng hành vi chưa lấp đầy.<br>
        - Có thể dùng Tỷ trọng lũy kế để chọn ngưỡng khách trung thành (ví dụ ≥3 đơn).<br>
        </div>
        """, unsafe_allow_html=True)

# -------- Phân tích nâng cao ----------
with tab_adv_product:
    st.subheader("Phân tích nâng cao")
    adv_df, observed_days_total = compute_product_advanced(df)
    if adv_df.empty:
        st.info("Thiếu dữ liệu sản phẩm để phân tích nâng cao.")
    else:
        view_vn = rename_for_display(adv_df, {
            "product_id":"Mã SP","product_name":"Tên SP",
            "first_sale":"Ngày bán đầu","last_sale":"Ngày bán cuối",
            "lifecycle_days":"Số ngày vòng đời","days_since_last":"Số ngày từ lần cuối",
            "active_days":"Số ngày có bán","coverage":"Tỷ lệ ngày có bán",
            "total_units":"Tổng SL","total_revenue":"Tổng doanh thu",
            "customers_total":"Số KH","customers_repeat":"KH mua lặp",
            "repeat_rate_product":"Tỷ lệ mua lại","mean_daily_all":"SL TB/ngày",
            "std_daily_all":"Độ lệch chuẩn/ngày","demand_cv":"CV nhu cầu",
            "variability_class":"Phân loại biến động",
            "interpurchase_median_days":"Trung vị chu kỳ mua",
            "interpurchase_mean_days":"TB chu kỳ mua",
            "abc_class":"ABC","cum_share":"Tỷ trọng lũy kế (%)"
        })
        st.write(f"Tổng ngày quan sát: {observed_days_total}")
        st.dataframe(view_vn.style.format({
            "Tổng doanh thu":"{:,.0f}",
            "Tỷ lệ ngày có bán":"{:,.2f}",
            "Tỷ lệ mua lại":"{:,.2f}",
            "SL TB/ngày":"{:,.2f}",
            "Độ lệch chuẩn/ngày":"{:,.2f}",
            "CV nhu cầu":"{:,.2f}",
            "Trung vị chu kỳ mua":"{:,.1f}",
            "TB chu kỳ mua":"{:,.1f}",
            "Tỷ trọng lũy kế (%)":"{:,.2f}"
        }), use_container_width=True, height=520)

        st.markdown("##### Phân bố vòng đời (Số ngày vòng đời)")
        lifecycle_hist = (alt.Chart(view_vn)
                          .mark_bar()
                          .encode(
                              alt.X("Số ngày vòng đời:Q", bin=alt.Bin(maxbins=30), title="Số ngày vòng đời"),
                              y=alt.Y("count():Q", title="Số SP"),
                              tooltip=[alt.Tooltip("count():Q", title="Số SP")]
                          ).properties(height=250))
        st.altair_chart(lifecycle_hist, use_container_width=True)

        st.markdown("""
        <div class="abc-box">
        <b>Định nghĩa nhóm ABC</b><br>
        - <b>Nhóm A</b>: Cộng dồn đến 80% doanh thu.<br>
        - <b>Nhóm B</b>: 80% → 95%.<br>
        - <b>Nhóm C</b>: 5% còn lại.<br>
        Khuyến nghị: Ưu tiên A, tối ưu B, tinh gọn C.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Scatter: CV nhu cầu vs Tỷ lệ mua lại")
        scatter = (alt.Chart(view_vn)
                   .mark_circle(size=60, opacity=0.7)
                   .encode(
                       x=alt.X("CV nhu cầu:Q", title="CV nhu cầu"),
                       y=alt.Y("Tỷ lệ mua lại:Q", title="Tỷ lệ mua lại"),
                       color=alt.Color(
                           "ABC:N",
                           title="ABC",
                           scale=alt.Scale(domain=["A","B","C"], range=["#2ca02c","#ffbf00","#d62728"])
                       ),
                       tooltip=["Mã SP","Tên SP","Tổng doanh thu","Tỷ lệ mua lại","CV nhu cầu","Phân loại biến động"]
                   ).properties(height=350))
        st.altair_chart(scatter, use_container_width=True)

# -------- Heatmap tuần ----------
with tab_heatmap:
    st.subheader("Heatmap theo ngày trong tuần")
    weekday_df = compute_weekday_summary(df)
    if not weekday_df.empty:
        mapping_weekday = {0:"Thứ 2",1:"Thứ 3",2:"Thứ 4",3:"Thứ 5",4:"Thứ 6",5:"Thứ 7",6:"CN"}
        weekday_df["Thứ"] = weekday_df["weekday"].map(mapping_weekday)
        wd_vn = weekday_df.rename(columns={"revenue":"Doanh thu","orders":"Số đơn"})
        # Biểu đồ: dùng bar + màu theo cường độ để giống "heat"
        heat_chart = (alt.Chart(wd_vn)
                      .mark_bar()
                      .encode(
                          x=alt.X("Thứ:N", sort=["Thứ 2","Thứ 3","Thứ 4","Thứ 5","Thứ 6","Thứ 7","CN"]),
                          y=alt.Y("Doanh thu:Q", title="Doanh thu"),
                          color=alt.Color("Doanh thu:Q", scale=alt.Scale(scheme="blues"), title=""),
                          tooltip=["Thứ","Doanh thu:Q","Số đơn:Q"]
                      ).properties(height=320, title=""))
        st.altair_chart(heat_chart, use_container_width=True)

        # Gợi ý tối ưu theo ngày tuần
        worst_row = wd_vn.loc[wd_vn["Doanh thu"].idxmin()]
        best_row = wd_vn.loc[wd_vn["Doanh thu"].idxmax()]
        worst_day = worst_row["Thứ"]
        best_day = best_row["Thứ"]
        ratio = (best_row["Doanh thu"] / worst_row["Doanh thu"]) if worst_row["Doanh thu"] else np.nan
        weekday_suggestions = []
        weekday_suggestions.append(f"Ngày doanh thu thấp nhất: {worst_day} (≈ {worst_row['Doanh thu']:,.0f}). Cao nhất: {best_day} (≈ {best_row['Doanh thu']:,.0f}).")
        if not np.isnan(ratio) and ratio >= 1.6:
            weekday_suggestions.append(f"Chênh lệch cao (gấp {ratio:.1f} lần): ưu tiên triển khai chiến dịch cứu {worst_day}.")
        weekday_suggestions.append(f"Đề xuất: đẩy flash sale / combo xả tồn / voucher ràng buộc tối thiểu vào {worst_day}.")
        weekday_suggestions.append("A/B test thông điệp giá trị khác nhau cho ngày yếu (ví dụ: freeship ngưỡng thấp hơn).")
        weekday_suggestions.append("Tận dụng retarget khách đã xem sản phẩm nhưng chưa mua trong 3 ngày gần nhất vào ngày yếu.")

        st.markdown(f"""
        <div class="weekday-box">
        <b>Gợi ý tối ưu theo ngày trong tuần</b><br>
        {'<br>'.join(weekday_suggestions)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Không đủ dữ liệu để xây dựng heatmap tuần.")

# -------- Xuất dữ liệu ----------
with tab_download:
    st.subheader("Xuất dữ liệu")
    def convert_df(dff):
        return dff.to_csv(index=False).encode("utf-8-sig")
    daily_metrics = compute_daily_metrics(df)
    cat_df = compute_category_metrics(df)
    prod_df = compute_product_metrics(df)
    cust_df_all = compute_customer_metrics(df)
    adv_df_export, _ = compute_product_advanced(df)
    st.download_button("⬇️ Daily metrics", data=convert_df(daily_metrics), file_name="daily_metrics.csv")
    st.download_button("⬇️ Category metrics", data=convert_df(cat_df), file_name="category_metrics.csv")
    if not prod_df.empty:
        st.download_button("⬇️ Product metrics", data=convert_df(prod_df), file_name="product_metrics.csv")
    if not cust_df_all.empty:
        st.download_button("⬇️ Customer metrics", data=convert_df(cust_df_all), file_name="customer_metrics.csv")
    if not adv_df_export.empty:
        st.download_button("⬇️ Advanced product (full)", data=convert_df(adv_df_export), file_name="product_advanced.csv")

st.markdown("---")
st.markdown(
    "<div style='text-align:left; color:#666; font-size:13px; margin-top:30px;'>© 2025 Đồ án tốt nghiệp lớp DL07_K306 - RFM Segmentation - Nhóm J</div>",
    unsafe_allow_html=True
)
