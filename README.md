# CIP
Customer Intelligence Platform
# Customer Intelligence Platform (E-commerce)

Nền tảng phân tích & khai phá khách hàng cho E-commerce: RFM, phân cụm, tập luật kết hợp giao diện trực quan (Streamlit).  
Mục tiêu: hỗ trợ đội Marketing / CRM / Data vận hành chiến lược retention, tăng CLV và tối ưu ngân sách.


## 🔥 Tính năng nổi bật

- RFM Segmentation: Phân nhóm khách hàng theo Recency / Frequency / Monetary.
- Association Rules: Khai phá sản phẩm thường được mua cùng (gợi ý cross-sell / bundle).
- Clustering (GMM / unsupervised): Khám phá phân khúc hành vi tiềm ẩn.
- Ảnh nền động + gradient tối bên trái → mờ dần giúp nội dung rõ ràng.
- Cấu trúc code dễ mở rộng (thêm mô-đun pipeline phân tích khác).
- Tối ưu tải ảnh (resize + WEBP nếu vượt quá ngưỡng).
- Dễ tuỳ biến theme (gradient, overlay, CTA, bật/tắt debug).

## 🗂 Cấu trúc (đề xuất)

```
.
├── app.py / cip.py          # File chính (UI Streamlit) (có thể đặt tên ecom.py trước đó)
├── assets/
│   ├── frontdoor.png         # Ảnh hero gốc
│   ├── hero_optimized.webp   # Ảnh hero sau tối ưu
│   ├── rfm.png               # Ảnh RFM
│   ├── rule.png              # Ảnh tập luật
│   ├── cluster.png           # Ảnh phân cụm
│   ├── phuong.png            # Ảnh GVHD (Khuất Thuỳ Phương)
│   ├── tien.png              # Ảnh học viên 1 (Phạm Đông Đức Tiến)
│   └── vinh.png              # Ảnh học viên 2 (Nguyễn Hoàng Vinh)
├── data/
│   ├── orders_full.csv       # Data sau xử lý
│   └── rfm_segments.csv      # Data sau chia RFM
├── model/
│   └── gmm/
│        └── gmm_rfm_v1/
│             ├── cluster_assignments.csv
│             ├── cluster_profile.parquet
│             ├── labels.parquet
│             ├── mapping_components.json
│             ├── meta.json
│             ├── metadata.json
│             ├── model.pkl
│             ├── model_gmm.pkl
│             ├── profile.csv
│             ├── scaler.pkl
│             └── segmentation_master.csv
├── pages/
│   ├── 01_Về chúng tôi.py         # About
│   ├── 02_Phân tích dữ liệu.py    # EDA
│   ├── 03_Phân cụm khách hàng.py  # Segmentation
│   └── 04_Phân tích chuyên sâu.py # Clustering chuyên sâu
├── scripts/
│   └── train_gmm.py          # Train mô hình GMM hoặc các pipeline
├── src/
│   ├── cluster_profile.py
│   ├── __init__.py
│   ├── model_io.py
│   ├── plot_rfm.py
│   ├── rfm_gmm.py
│   ├── rfm_kmeans.py
│   ├── rfm_labeling.py
│   ├── rfm_rule_scoring.py
│   └── rfm_utils.py
├── requirements.txt
└── README.md
```

## 👥 Nhóm thực hiện

| Thành viên | Vai trò | Email | Avatar |
|------------|---------|-------|--------|
| Phạm Đông Đức Tiến | Data / Modeling | tdongpham21@gmail.com | assets/tien.png |
| Nguyễn Hoàng Vinh | Data Engineering / GUI & Visualization | jr.vinh@gmail.com | assets/vinh.png |

## 🎓 Giảng viên hướng dẫn

- Thạc sĩ Khuất Thuỳ Phương

## 📦 Yêu cầu hệ thống

- Python >= 3.9
- Streamlit >= 1.33
- Pillow
- numpy, pandas, scikit-learn
- mlxtend (Apriori / FP-Growth)
- (Tùy chọn) plotly / seaborn

## ⚙ Cài đặt

```bash
git clone https://github.com/your-org/customer-intel.git
cd customer-intel

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run cip.py  # hoặc app.py / ecom.py tùy tên bạn dùng
```

## 🧪 Dữ liệu đầu vào (tối thiểu)

File orders (orders.csv):
- order_id
- customer_id
- order_date (YYYY-MM-DD hoặc timestamp)
- amount (giá trị đơn hàng)
- product_id (nếu dùng luật / giỏ hàng)
- quantity (tùy chọn)

Nếu một đơn có nhiều dòng: chuẩn hóa thành transaction lines.

## 🧮 RFM (gợi ý logic)

- Recency: ngày hiện tại - ngày mua cuối
- Frequency: số đơn hoàn tất
- Monetary: tổng chi tiêu
- Chấm điểm 1–5 theo quintile / rule tùy biến
- Segment: ghép mã (ví dụ 555 = "Champions")

## 🔗 Association Rules

- Gom transactions: order_id → list(product_id)
- Chạy Apriori / FP-Growth
- Lọc luật theo min_support, min_confidence
- Giữ lift > 1.05 hoặc theo ngưỡng nghiệp vụ

## 🤖 Clustering

- Chuẩn hóa (StandardScaler)
- Có thể PCA giảm chiều (tùy)
- Mô hình: KMeans, GMM
- Đánh giá: Silhouette (KMeans), BIC / AIC (GMM)
- Lưu phân cụm: cluster_assignments + profile

## 🎨 Tuỳ biến giao diện

Trong file UI có các biến gradient:

```css
:root {
  --grad-strong: rgba(0,0,0,0.70);
  --grad-mid:    rgba(0,0,0,0.42);
  --grad-soft:   rgba(0,0,0,0.18);
}
```

Giảm tối: hạ alpha (0.70 → 0.58).  
Tăng tốc fade: đổi 34% / 62% thành 28% / 52%.

### Bật / tắt gradient nhanh (dev)

```python
use_grad = st.sidebar.toggle("Gradient overlay", value=True)
grad_css = "opacity:1;" if use_grad else "opacity:0;"
st.markdown(f"<style>.hero-banner:before,.feature-card:before{{{grad_css}}}</style>", unsafe_allow_html=True)
```

## 🛠 Cải thiện & tối ưu

- @st.cache_data: load CSV nhanh
- @st.cache_resource: giữ model (KMeans / GMM)
- Ảnh lớn: resize + WEBP (>600KB)
- Lazy compute: chỉ chạy khi người dùng bấm nút
- Có thể tách CSS: assets/styles.css

## 🚀 Triển khai

### Streamlit Community Cloud
1. Push repo lên GitHub
2. Add app (chọn cip.py / app.py)

Build & run:
```bash
docker build -t customer-intel .
docker run -p 8501:8501 customer-intel
```

## 🔐 Biến môi trường 

- DATA_PATH (đường dẫn thư mục dữ liệu)
- MODEL_CACHE_PATH

```python
import os
DATA_PATH = os.getenv("DATA_PATH", "data")
```

## 🧩 Lộ trình mở rộng

| Hạng mục | Mô tả | Ưu tiên |
|----------|-------|---------|
| Churn Prediction | Dự báo rời bỏ | Cao |
| Next Purchase Date | Ước lượng ngày mua tiếp theo | Trung bình |
| Uplift Modeling | Đo tác động chiến dịch | Thấp |
| Real-time Events | Streaming Kafka / PubSub | Trung bình |
| Recommendation | Gợi ý sản phẩm (Hybrid) | Cao |
| Cohort Analysis | Giữ chân theo nhóm tháng | Cao |

## 🙏 Acknowledgements / Ghi nhận

Xin ghi nhận và cảm ơn các nguồn, công cụ và đóng góp sau:
- Giảng viên hướng dẫn: ThS. Khuất Thuỳ Phương – định hướng phương pháp và phản biện học thuật.
- Open Source Ecosystem:
  - Streamlit (UI nhanh)
  - pandas & numpy (xử lý dữ liệu)
  - scikit-learn (chuẩn hóa & clustering)
  - mlxtend (Apriori / FP-Growth)
  - plotly / seaborn (trực quan hoá)
- Các bộ dữ liệu mẫu nội bộ / giả lập được tạo nhằm mục đích học thuật (không chứa dữ liệu cá nhân thực).
- Ý tưởng phân nhóm RFM & mapping segment tham khảo từ các guideline phổ biến trên cộng đồng Marketing Analytics (điều chỉnh lại cho phù hợp).
- Đóng góp nội bộ nhóm:
  - Phạm Đông Đức Tiến: Thiết kế pipeline RFM + GMM, logic gán nhãn, chuẩn hoá dữ liệu.
  - Nguyễn Hoàng Vinh: Phản biện, xây dựng UI, tối ưu hiển thị, tổ chức cấu trúc dự án & visualization.
- Các tài liệu học thuật / blog về Gaussian Mixture & Customer Segmentation được sử dụng làm tham chiếu (không tái sử dụng nguyên văn).

(Nếu sử dụng tài nguyên ngoài như icon, font, cần bổ sung nguồn cụ thể tại đây.)

## 🙌 Lời cảm ơn

Cảm ơn Giảng viên: ThS. Khuất Thuỳ Phương đã hướng dẫn và góp ý chuyên môn.  
Đồ án phục vụ mục đích học thuật và thử nghiệm kỹ thuật phân tích khách hàng.

---
Liên hệ:  
- Giảng viên hướng dẫn: Khuất Thuỳ Phương - tubirona@gmail.com
- Học viên 1: Phạm Đông Đức Tiến - tdongpham21@gmail.com  
- Học viên 2: Nguyễn Hoàng Vinh - jr.vinh@gmail.com  
