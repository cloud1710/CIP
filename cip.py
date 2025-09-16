import streamlit as st
from pathlib import Path
import base64, mimetypes
from PIL import Image

st.set_page_config(page_title="Ecommerce", page_icon="🛍️", layout="wide")

# ============ ẢNH HERO ============
RAW_HERO = Path("assets/frontdoor.png")
OPTIMIZED_HERO = Path("assets/hero_optimized.webp")
MAX_SIZE_KB = 600
TARGET_MAX_WIDTH = 1720
QUALITY = 82

def ensure_optimized(src: Path, dst: Path):
    if not src.exists():
        return None
    if dst.exists():
        return dst
    size_kb = src.stat().st_size / 1024
    if size_kb <= MAX_SIZE_KB:
        return src
    try:
        img = Image.open(src)
        if img.width > TARGET_MAX_WIDTH:
            ratio = TARGET_MAX_WIDTH / img.width
            img = img.resize((TARGET_MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode not in ("RGB","L"):
            img = img.convert("RGB")
        img.save(dst, format="WEBP", quality=QUALITY, method=6)
        return dst
    except Exception:
        return src

HERO_FILE = ensure_optimized(RAW_HERO, OPTIMIZED_HERO)
HERO_EXISTS = HERO_FILE is not None and HERO_FILE.exists()

def to_b64(p: Path):
    if not p or not p.exists():
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode()
    except Exception:
        return None

HERO_B64 = to_b64(HERO_FILE) if HERO_EXISTS else None
HERO_MIME = mimetypes.guess_type(str(HERO_FILE))[0] if HERO_EXISTS else "image/webp"

# ============ ẢNH FEATURES ============
IMG_RFM = Path("assets/rfm.png")
IMG_RULE = Path("assets/rule.png")
IMG_CLUSTER = Path("assets/cluster.png")

def to_b64_local(p: Path):
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()

features = [
    {
        "title": "Phân tích R-F-M",
        "img": IMG_RFM,
        "text": "Phân nhóm dựa trên Recency, Frequency, Monetary.\nƯu tiên chăm sóc nhóm giá trị & ngăn khách hàng rời bỏ sớm",
        "tag": "RFM"
    },
    {
        "title": "Tập luật",
        "img": IMG_RULE,
        "text": "Khai phá các nhóm khách hàng.\nXây dựng chiến lược kích cầu, khuyến mãi phù hợp với từng người",
        "tag": "TẬP LUẬT"
    },
    {
        "title": "Phân cụm theo Gaussian Mixture Model",
        "img": IMG_CLUSTER,
        "text": "Nhóm khách tương đồng hành vi mua sắm.\nKhám phá phân khúc & thiết kế chiến dịch chính xác",
        "tag": "GMM"
    },
]

# ============ CSS HERO ============
def hero_background_css():
    if HERO_B64:
        return f"""
        .hero-banner {{
          background:
            url("data:{HERO_MIME};base64,{HERO_B64}") center/cover no-repeat;
          background-color:#0f1115;
        }}
        """
    else:
        return """
        .hero-banner {
          background-color:#0f1115;
        }
        """

BASE_CSS = f"""
<style>
  :root {{
    /* Dễ chỉnh cường độ gradient chung */
    --grad-strong: rgba(0,0,0,0.70);
    --grad-mid:    rgba(0,0,0,0.42);
    --grad-soft:   rgba(0,0,0,0.18);
  }}
  body, .block-container {{
    font-family:-apple-system,BlinkMacSystemFont,Inter,Segoe UI,Roboto,Ubuntu,sans-serif;
  }}

  /* HERO */
  .hero-banner {{
    position:relative;
    border-radius:34px;
    padding:60px 60px 72px;
    color:#ffffff;
    overflow:hidden;
    border:1px solid rgba(255,255,255,0.10);
    box-shadow:0 14px 40px -18px rgba(0,0,0,0.55), 0 6px 18px -4px rgba(0,0,0,0.35);
    isolation:isolate;
  }}
  {hero_background_css()}
  .hero-banner:before {{
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(90deg,
      var(--grad-strong) 0%,
      var(--grad-mid) 70%,
      var(--grad-soft) 80%,
      rgba(0,0,0,0) 100%);
    z-index:0;
    pointer-events:none;
  }}
  .hero-content {{ position:relative; z-index:1; max-width:760px; }}
  .hero-title {{
    font-size:48px;
    line-height:1.05;
    font-weight:800;
    letter-spacing:-1.4px;
    margin:0 0 22px;
    color:#ffffff;
    text-shadow:0 2px 8px rgba(0,0,0,0.45);
  }}
  .hero-sub {{
    font-size:18.5px;
    line-height:1.55;
    font-weight:450;
    color:#ffffff;
    margin:0 0 26px;
    white-space:pre-line;
    text-shadow:0 2px 6px rgba(0,0,0,0.55);
  }}
  .hero-chips {{
    display:flex;
    flex-wrap:wrap;
    gap:10px;
  }}
  .hero-chips span {{
    color:#fff;
    font-size:13.5px;
    padding:7px 16px 6px;
    border-radius:24px;
    font-weight:600;
    letter-spacing:.35px;
    background:rgba(255,255,255,0.16);
    backdrop-filter:blur(6px);
    border:1px solid rgba(255,255,255,0.34);
    text-shadow:0 1px 4px rgba(0,0,0,0.55);
  }}

  /* FEATURE GRID */
  .section-title {{
    font-size:28px;
    font-weight:700;
    letter-spacing:-0.6px;
    margin:4px 0 6px;
    background:linear-gradient(90deg,#111,#4338CA 70%);
    -webkit-background-clip:text;
    color:transparent;
  }}
  .feature-grid {{
    display:grid;
    gap:30px;
    grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
    margin-top:8px;
  }}
  .feature-card {{
    position:relative;
    border:1px solid #222831;
    border-radius:28px;
    padding:28px 26px 30px;
    overflow:hidden;
    min-height:340px;
    background:#12161b;
    color:#fff;
    display:flex;
    flex-direction:column;
    justify-content:flex-end;
    box-shadow:0 10px 34px -12px rgba(0,0,0,0.55), 0 4px 14px -4px rgba(0,0,0,0.45);
    transition:.45s;
    isolation:isolate;
  }}
  .feature-card:before {{
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(90deg,
      var(--grad-strong) 0%,
      var(--grad-mid) 34%,
      var(--grad-soft) 62%,
      rgba(0,0,0,0) 100%);
    z-index:0;
    pointer-events:none;
    transition:.5s;
  }}
  .feature-card:hover {{
    transform:translateY(-8px);
    box-shadow:0 22px 50px -18px rgba(0,0,0,0.70), 0 6px 22px -6px rgba(0,0,0,0.55);
    border-color:#2c343c;
  }}
  .feature-card:hover:before {{
    background:linear-gradient(90deg,
      rgba(0,0,0,0.58) 0%,
      rgba(0,0,0,0.34) 34%,
      rgba(0,0,0,0.10) 62%,
      rgba(0,0,0,0) 100%);
  }}
  .feature-title {{
    position:relative;
    z-index:1;
    font-size:24px;
    font-weight:700;
    margin:0 0 12px;
    line-height:1.12;
    letter-spacing:-0.5px;
    color:#ffffff;
    text-shadow:0 2px 8px rgba(0,0,0,0.65);
  }}
  .feature-text {{
    position:relative;
    z-index:1;
    font-size:15.5px;
    line-height:1.55;
    white-space:pre-line;
    font-weight:460;
    color:#f0f4f6;
    text-shadow:0 1px 5px rgba(0,0,0,0.65);
    margin:0 0 6px;
  }}
  .meta-chip {{
    position:absolute;
    top:12px;
    right:12px;
    background:rgba(0,0,0,0.55);
    color:#fff;
    font-size:12px;
    padding:6px 14px 5px;
    border-radius:40px;
    letter-spacing:.5px;
    font-weight:600;
    backdrop-filter:blur(4px);
    z-index:2;
    border:1px solid rgba(255,255,255,0.25);
    text-shadow:0 1px 3px rgba(0,0,0,0.55);
  }}

  @media (max-width:880px) {{
    .hero-title {{ font-size:40px; }}
    .hero-banner {{ padding:50px 40px 64px; }}
    .feature-grid {{ grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }}
  }}
  @media (max-width:600px) {{
    .hero-title {{ font-size:34px; }}
    .hero-sub {{ font-size:16.5px; }}
    .hero-banner {{ padding:42px 28px 56px; border-radius:26px; }}
    .feature-card {{ min-height:300px; padding:24px 22px 26px; }}
    .feature-title {{ font-size:22px; }}
  }}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ============ CSS BACKGROUND CHO TỪNG FEATURE ============
dynamic_css_parts = []
for idx, f in enumerate(features):
    b64 = to_b64_local(f["img"])
    if b64:
        dynamic_css_parts.append(
            f""".feature-card.bg-{idx} {{
                background:
                  url("data:image/png;base64,{b64}") center/cover no-repeat;
              }}"""
        )
    else:
        dynamic_css_parts.append(
            f""".feature-card.bg-{idx} {{
                background:#1d2329;
              }}"""
        )
if dynamic_css_parts:
    st.markdown("<style>" + "\n".join(dynamic_css_parts) + "</style>", unsafe_allow_html=True)

# ============ HERO ============
st.markdown(
    """
    <div class="hero-banner">
      <div class="hero-content">
        <h1 class="hero-title">Nền tảng phân tích khách hàng dành cho các cửa hàng bán lẻ</h1>
        <div class="hero-sub">Khai phá hành vi & giá trị khách hàng với RFM, phân cụm, phân tích luật
Làm nền tảng cho chiến lược giữ chân, tăng giá trị vòng đời khách hàng và tối ưu ngân sách</div>
        <div class="hero-chips">
          <span>RFM</span>
          <span>Gaussian Mixture</span>
          <span>Insight Dashboard</span>
          <span>Automation Ready</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ============ FEATURES ============
st.markdown('<div class="section-title">Giải pháp cốt lõi</div>', unsafe_allow_html=True)

st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
for idx, f in enumerate(features):
    st.markdown(
        f"""
        <div class="feature-card bg-{idx}">
          <div class="meta-chip">{f["tag"]}</div>
          <div class="feature-title">{f["title"]}</div>
          <div class="feature-text">{f["text"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:left; color:#666; font-size:13px; margin-top:30px;'>© 2025 Đồ án tốt nghiệp lớp DL07_K306 - RFM Segmentation - Nhóm J</div>",
    unsafe_allow_html=True
)
