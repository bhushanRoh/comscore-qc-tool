import streamlit as st
import cv2
import numpy as np
import pandas as pd
from utils.image_processing import process_image, create_tiles
from utils.ocr_engine import OCREngine
from utils.blacklist_manager import load_blacklist, add_term, remove_term, check_compliance, find_term_positions

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COMSCORE QC Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    .brand-accent {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }

    /* Metric card */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .metric-card.fail .metric-value { color: #ef4444; }
    .metric-card.pass .metric-value { color: #22c55e; }
    .metric-card.info .metric-value { color: #6366f1; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* Status badges */
    .badge-fail {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-pass {
        background: linear-gradient(90deg, #16a34a, #22c55e);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Initialize OCR Engine (Cached) ────────────────────────────────────────────
@st.cache_resource
def get_ocr_engine():
    return OCREngine()

ocr_engine = get_ocr_engine()


# ─── Sidebar: Admin Panel ──────────────────────────────────────────────────────
st.sidebar.markdown("## 🔐 Brand Database")
st.sidebar.caption("Manage forbidden brand names. Changes are auto-saved.")
st.sidebar.markdown("---")

# ── Add brands (supports multiple, comma-separated) ──
st.sidebar.markdown("### ➕ Add Brands")
new_terms_input = st.sidebar.text_area(
    "Enter brand names (comma-separated)",
    placeholder="e.g. Repatha, Evolocumab, Pfizer",
    height=80,
    key="new_terms_input"
)
if st.sidebar.button("💾 Save to Database", use_container_width=True):
    if new_terms_input:
        # Parse comma-separated and newline-separated terms
        raw_terms = new_terms_input.replace("\n", ",").split(",")
        terms_to_add = [t.strip() for t in raw_terms if t.strip()]

        added = []
        duplicates = []
        for term in terms_to_add:
            if add_term(term):
                added.append(term)
            else:
                duplicates.append(term)

        if added:
            st.sidebar.success(f"✅ Added {len(added)}: {', '.join(added)}")
        if duplicates:
            st.sidebar.warning(f"⚠️ Already exist: {', '.join(duplicates)}")
        if added:
            st.rerun()

# ── Current database ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Current Database")
blacklist_terms = load_blacklist()

if blacklist_terms:
    st.sidebar.markdown(f"**{len(blacklist_terms)}** brands tracked")
    df_blacklist = pd.DataFrame(blacklist_terms, columns=["Brand Name"])
    st.sidebar.dataframe(df_blacklist, use_container_width=True, hide_index=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗑️ Remove Brand")
    term_to_remove = st.sidebar.selectbox(
        "Select brand to remove",
        ["—"] + blacklist_terms,
        label_visibility="collapsed"
    )
    if st.sidebar.button("Delete Selected", use_container_width=True, type="secondary"):
        if term_to_remove and term_to_remove != "—":
            if remove_term(term_to_remove):
                st.sidebar.success(f"Deleted: {term_to_remove}")
                st.rerun()
else:
    st.sidebar.info("No brands in database yet. Add some above.")


# ─── Main Dashboard ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 <span class="brand-accent">COMSCORE</span> QC Tool</h1>
    <p>Upload promotional boards to scan for forbidden brand names and ensure compliance.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image Board (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Supports single images and multi-panel boards with 4-5 panels."
)

if uploaded_file is not None:
    # 1. Process Image
    image_variants, original_img = process_image(uploaded_file.read())

    # 2. Create tiles for board images
    tiles = create_tiles(original_img, grid=(2, 2), overlap=0.15)

    with st.spinner("🔍 Scanning all regions for compliance violations..."):
        # 3. Run OCR across full-image variants + tiles
        results = ocr_engine.extract_text_multi(image_variants, tiles=tiles)

        # 4. Check Compliance
        violations_found = []
        image_with_boxes = original_img.copy()
        current_blacklist = load_blacklist()

        for (bbox, text, prob) in results:
            term_positions = find_term_positions(text, current_blacklist)

            if not term_positions:
                continue

            text_stripped = text.strip()
            total_chars = len(text_stripped)

            if total_chars == 0:
                continue

            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            bbox_width = x2 - x1
            char_width = bbox_width / total_chars

            for (term, start_idx, end_idx) in term_positions:
                violations_found.append((text, term))

                sub_x1 = int(x1 + start_idx * char_width)
                sub_x2 = int(x1 + end_idx * char_width)
                sub_y1 = int(y1)
                sub_y2 = int(y2)

                pad = 3
                sub_x1 = max(0, sub_x1 - pad)
                sub_y1 = max(0, sub_y1 - pad)
                sub_x2 = sub_x2 + pad
                sub_y2 = sub_y2 + pad

                cv2.rectangle(image_with_boxes, (sub_x1, sub_y1), (sub_x2, sub_y2), (0, 0, 255), 3)

    # 5. Results Display
    unique_violations = set([v[1] for v in violations_found])
    total_hits = len(violations_found)
    total_scanned = len(results)

    if violations_found:
        # Metrics row
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card fail">
                <div class="metric-value">{total_hits}</div>
                <div class="metric-label">Violations Found</div>
            </div>
            <div class="metric-card info">
                <div class="metric-value">{len(unique_violations)}</div>
                <div class="metric-label">Unique Brands</div>
            </div>
            <div class="metric-card info">
                <div class="metric-value">{total_scanned}</div>
                <div class="metric-label">Text Regions Scanned</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="badge-fail">❌ FAILED QC</div>', unsafe_allow_html=True)
        st.markdown(f"**Detected brands:** {', '.join(f'`{v}`' for v in unique_violations)}")

        st.image(
            cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB),
            caption="Violations Highlighted",
            use_container_width=True
        )

    else:
        # Metrics row
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card pass">
                <div class="metric-value">0</div>
                <div class="metric-label">Violations Found</div>
            </div>
            <div class="metric-card info">
                <div class="metric-value">{total_scanned}</div>
                <div class="metric-label">Text Regions Scanned</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="badge-pass">✅ GOOD TO GO</div>', unsafe_allow_html=True)
        st.image(
            cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
            caption="Clean — No violations detected",
            use_container_width=True
        )
