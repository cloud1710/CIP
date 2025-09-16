import streamlit as st
import os, base64

# ================== CẤU HÌNH CƠ BẢN ==================
APP_NAME = "ECOM - Customers Segmentation"
APP_VERSION = "v1.3.0"

# ================== ẢNH LOCAL ==================
AVATAR_PHUONG = "assets/phuong.png"
AVATAR_TIEN = "assets/tien.png"
AVATAR_VINH = "assets/vinh.png"

# (Nếu chưa có BASE64_PLACEHOLDER thì cần khai báo biến này, ví dụ:
# BASE64_PLACEHOLDER = "iVBORw0KGgoAAA..."  # chuỗi base64 của 1 ảnh nhỏ)

def load_local_or_placeholder(path: str):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return f.read(), False
        except Exception:
            pass
    # Chú ý: cần đảm bảo BASE64_PLACEHOLDER tồn tại
    return base64.b64decode(BASE64_PLACEHOLDER), True

# ================== UI: TIÊU ĐỀ ==================
st.title("📘 Giới thiệu dự án")
st.caption(f"{APP_NAME} • Phiên bản {APP_VERSION}")

# ================== GIẢNG VIÊN HƯỚNG DẪN ==================
st.markdown("### 👩‍🏫 Giảng viên hướng dẫn")
col_gv = st.columns(1)[0]
with col_gv:
    gv_img_bytes, gv_is_fallback = load_local_or_placeholder(AVATAR_PHUONG)
    st.image(gv_img_bytes, caption="ThS. Khuất Thuỳ Phương", width=150)
    st.write("Giảng viên hướng dẫn")
    if gv_is_fallback:
        st.warning(f"Không đọc được ảnh: {AVATAR_PHUONG} (đang dùng placeholder).")

# ================== TEAM ==================
st.markdown("### 👥 Thực hiện - Nhóm J")
members = [
    {
        "name": "Phạm Đông Đức Tiến",
        "roles": "Data / Modeling",
        "email": "✉ tdongpham21@gmail.com",
        "avatar": AVATAR_TIEN
    },
    {
        "name": "Nguyễn Hoàng Vinh",
        "roles": "Data Engineering / GUI & Visualization",
        "email": "✉ jr.vinh@gmail.com",
        "avatar": AVATAR_VINH
    }
]

cols = st.columns(len(members))
for col, m in zip(cols, members):
    with col:
        img_bytes, is_fallback = load_local_or_placeholder(m["avatar"])
        st.image(img_bytes, caption=m["name"], width=150)
        st.write(m["roles"])
        st.caption(m["email"])
        if is_fallback:
            st.warning(f"Không đọc được ảnh: {m['avatar']} (đang dùng placeholder).")

# ================== MỤC TIÊU ==================
st.markdown("### 🎯 Mục tiêu")
st.markdown(
    "- Chuẩn hoá phân khúc khách hàng dựa trên Recency, Frequency, Monetary (RFM)\n"
    "- Kết hợp phân khúc luật RFM + mô hình Gaussian Mixture (GMM)\n"
    "- Hỗ trợ gợi ý chiến lược Marketing / CSKH"
)

# ================== ĐỊNH NGHĨA RFM ==================
with st.expander("📗 Định nghĩa R / F / M"):
    st.markdown(
        """
        - Recency: Số ngày từ lần mua gần nhất  
        - Frequency: Số đơn hàng trong giai đoạn quan sát  
        - Monetary: Tổng giá trị chi tiêu  
        - RFM Score: Ghép điểm R-F-M (ví dụ 5-3-4)  
        - RFM_Level: Phân loại theo luật đặt trước  
        """
    )

# ================== PIPELINE ==================
st.markdown("### 🧪 Pipeline tóm tắt")
st.markdown(
    "1. Tiền xử lý dữ liệu đơn hàng\n"
    "2. Tính R / F / M\n"
    "3. Chấm điểm & gán RFM_Level (rule-based)\n"
    "4. Huấn luyện GMM trên (R, F, M) đã chuẩn hoá\n"
    "5. Dashboard hiển thị & khai thác"
)

# ================== TECH STACK ==================
st.markdown("### 🛠 Tech Stack")
st.markdown(
    "- Python, Pandas, NumPy\n"
    "- Scikit-learn (GaussianMixture)\n"
    "- Streamlit Dashboard\n"
    "- Rule-based RFM + GMM Clustering"
)

# ================== GHI CHÚ ==================
st.markdown("### ⚖️ Lưu ý")
st.markdown(
    "- Dữ liệu ví dụ không phản ánh toàn bộ dữ liệu thật.\n"
    "- Gợi ý chiến lược cần kiểm thử trước khi áp dụng rộng.\n"
    "- Ảnh avatar đọc từ thư mục local: assets/."
)

# ================== LỜI CẢM ƠN ==================
st.markdown("### 🙏 Lời cảm ơn")
st.markdown(
    "Nhóm xin chân thành cảm ơn ThS. **Khuất Thuỳ Phương** đã tận tình hướng dẫn, "
    "định hướng phương pháp và đóng góp nhiều ý kiến chuyên môn quan trọng trong suốt quá trình thực hiện đề tài.  \n"
    "Xin cảm ơn thầy Trí Lê và nhà trường đã tạo điều kiện để học tập.  \n"
    "Cảm ơn các bạn cùng lớp đã hỗ trợ trao đổi dữ liệu & kinh nghiệm.  \n"
    "Mọi thiếu sót nhóm rất mong nhận được góp ý để hoàn thiện hơn."
)

# ================== FOOTER ==================
st.markdown(
    "<div style='text-align:left; color:#666; font-size:13px; margin-top:30px;'>© 2025 Đồ án tốt nghiệp lớp DL07_K306 - RFM Segmentation - Nhóm J</div>",
    unsafe_allow_html=True
)
