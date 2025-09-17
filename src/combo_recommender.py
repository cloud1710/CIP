import pandas as pd
import itertools
from collections import defaultdict

def prepare_line_items(df, product_col="product_name"):
    required = {"order_id", product_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Thiếu cột: {required - set(df.columns)}")
    # Giữ mỗi sản phẩm 1 lần/đơn (nếu quantity >1 vẫn tính là đồng xuất hiện)
    line = df[["order_id", product_col]].dropna().drop_duplicates()
    return line

def build_cooccurrence(line_df, product_col="product_name", min_support_orders=5):
    # Đếm số đơn mỗi sản phẩm
    prod_orders = (line_df.groupby(product_col)["order_id"]
                   .nunique()
                   .rename("prod_order_count"))
    total_orders = line_df["order_id"].nunique()

    # Nhóm theo order_id để lấy danh sách sản phẩm
    grouped = (line_df.groupby("order_id")[product_col]
               .apply(list))

    pair_counts = defaultdict(int)
    for prods in grouped:
        # Loại bỏ trùng trong cùng đơn
        unique = sorted(set(prods))
        for a, b in itertools.combinations(unique, 2):
            pair_counts[(a, b)] += 1
            pair_counts[(b, a)] += 1  # cho hướng ngược

    rows = []
    for (a, b), c in pair_counts.items():
        if c < min_support_orders:  # lọc noise
            continue
        count_a = prod_orders.get(a, 1)
        count_b = prod_orders.get(b, 1)
        support_pair = c / total_orders
        conf = c / count_a
        lift = (c * total_orders) / (count_a * count_b)
        rows.append({
            "antecedent": a,
            "consequent": b,
            "pair_order_count": c,
            "support_pair": support_pair,
            "confidence": conf,
            "lift": lift,
            "count_a": count_a,
            "count_b": count_b
        })
    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules, prod_orders, total_orders
    # Chuẩn hoá điểm đơn giản
    rules["score_base"] = (0.5 * rules["lift"]) + (0.5 * rules["confidence"])
    return rules.sort_values("score_base", ascending=False), prod_orders, total_orders

def build_customer_profile(orders, cust_id, product_col="product_name"):
    subset = orders[orders["customer_id"] == cust_id]
    if subset.empty:
        return {"recent_products": [], "all_products": []}
    # Sản phẩm mua gần đây nhất (top k)
    subset = subset.sort_values("date", ascending=False)
    recent = subset[product_col].head(5).dropna().tolist()
    allp = subset[product_col].dropna().unique().tolist()
    return {"recent_products": recent, "all_products": allp}

def recommend_combos_for_customer(cust_id, orders, rules_df,
                                  product_col="product_name",
                                  top_k=5, min_lift=1.05):
    profile = build_customer_profile(orders, cust_id, product_col)
    owned = set(profile["all_products"])
    if not owned or rules_df is None or rules_df.empty:
        return []
    candidate_rows = rules_df[rules_df["antecedent"].isin(owned)].copy()
    candidate_rows = candidate_rows[candidate_rows["lift"] >= min_lift]
    # Loại sản phẩm khách đã mua nhiều lần nếu muốn gợi ý mới
    candidate_rows = candidate_rows[~candidate_rows["consequent"].isin(owned)]
    if candidate_rows.empty:
        return []
    # Scoring có thể thêm margin, affinity...
    candidate_rows["final_score"] = candidate_rows["score_base"]
    out = (candidate_rows
           .sort_values("final_score", ascending=False)
           .head(top_k)
           .to_dict("records"))
    return out