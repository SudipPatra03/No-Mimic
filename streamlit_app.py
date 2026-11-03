"""
Streamlit Web Application — No-Mimic
==========================================
Premium UI for audio upload, prediction, and visualization.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import sys
import json
import tempfile
import numpy as np
from PIL import Image, ImageDraw

import torch
import torchaudio
import torch.nn as nn

# Import modular components
from config import config
from src.preprocessor import AudioPreprocessor
from src.feature_extractor import FeatureExtractor
from src.model import VoiceAuthenticityModel
from src.predict import VoicePredictor


# ── Generate Favicon to avoid Emoji in Page Config ────────────
favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.png")
try:
    img = Image.new('RGBA', (32, 32), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 28, 28], fill=(20, 25, 35, 255), outline=(122, 171, 255, 255), width=2)
    draw.ellipse([11, 11, 21, 21], fill=(122, 171, 255, 255))
    img.save(favicon_path)
except Exception:
    pass

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=favicon_path if os.path.exists(favicon_path) else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG Definitions ───────────────────────────────────────────
SVG_MIC = """<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>"""

SVG_SEARCH = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>"""

SVG_CPU = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>"""

SVG_CHART = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>"""

SVG_UPLOAD = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""

SVG_CHECK_SHIELD = """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 11 2 2 4-4"/></svg>"""

SVG_ALERT_SHIELD = """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

SVG_MUSIC = """<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>"""

SVG_LAYERS = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>"""

SVG_AWARD = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>"""

# ── Reason Card SVG Icons (16x16, inline) ─────────────────────
SVG_REASON_SPECTRUM = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>"""

SVG_REASON_ENERGY = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>"""

SVG_REASON_MFCC = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10s3-3 5-3 5 5 7 5 5-3 5-3"/><path d="M2 16s3-3 5-3 5 5 7 5 5-3 5-3"/></svg>"""

SVG_REASON_FREQUENCY = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>"""

SVG_REASON_TEMPORAL = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>"""

SVG_REASON_MODEL_OK = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 11 2 2 4-4"/></svg>"""

SVG_REASON_MODEL_WARNING = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

SVG_REASON_SEARCH = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>"""

SVG_REASON_RULER = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7aabff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/></svg>"""

# Map reason icon identifiers to SVGs
REASON_ICON_MAP = {
    "spectrum": SVG_REASON_SPECTRUM,
    "energy": SVG_REASON_ENERGY,
    "mfcc": SVG_REASON_MFCC,
    "frequency": SVG_REASON_FREQUENCY,
    "temporal": SVG_REASON_TEMPORAL,
    "model_ok": SVG_REASON_MODEL_OK,
    "model_warning": SVG_REASON_MODEL_WARNING,
}

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0f1318;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #141920;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio label span,
section[data-testid="stSidebar"] .stRadio label p,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #e0e4ea !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover,
section[data-testid="stSidebar"] .stRadio label:hover span,
section[data-testid="stSidebar"] .stRadio label:hover p {
    color: #fff !important;
}

/* Hide default material icons inside Streamlit headers/sidebars to prevent text fallback rendering bugs */
[data-testid="stIconMaterial"] {
    display: none !important;
}

/* Render clean right-chevron arrow for collapsed sidebar toggle */
[data-testid="collapsedControl"] button[data-testid="stBaseButton-headerNoPadding"]::after {
    content: "▶" !important;
    font-size: 14px !important;
    color: #e0e4ea !important;
    font-weight: bold !important;
}

/* Render clean left-chevron arrow for expanded sidebar collapse button */
[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]::after {
    content: "◀" !important;
    font-size: 14px !important;
    color: #e0e4ea !important;
    font-weight: bold !important;
}

/* ── Page Header Elements ── */
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f0f2f5;
    margin: 0;
}
.page-desc {
    color: #8b95a5;
    font-size: 0.95rem;
    margin-bottom: 28px;
}

/* ── Section Label ── */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #8b95a5;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
}

/* ── Cards ── */
.card {
    background: #1a1f2a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 18px;
}

/* ── Stat Cards ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 14px 0;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 110px;
    background: #1a1f2a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: #7aabff;
}
.stat-lbl {
    font-size: 0.73rem;
    color: #8b95a5;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Result Box ── */
.result-box {
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    margin: 8px 0;
}
.result-ok {
    background: #13201a;
    border: 1px solid rgba(52,211,153,0.3);
}
.result-bad {
    background: #201316;
    border: 1px solid rgba(239,68,68,0.3);
}
.result-title { font-size: 1.3rem; font-weight: 700; }
.result-pct { font-size: 2.6rem; font-weight: 800; margin: 6px 0; }
.c-green { color: #34d399; }
.c-red { color: #ef4444; }
.result-sub { font-size: 0.82rem; color: #8b95a5; margin-top: 6px; }

/* ── Confidence Bar ── */
.bar-track {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    height: 5px;
    overflow: hidden;
    margin-top: 14px;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 50px 20px;
}
.empty-state .icon { opacity: 0.35; margin-bottom: 12px; }
.empty-state .msg {
    margin-top: 12px;
    font-size: 0.9rem;
    color: #6b7585;
}

/* ── Architecture Box ── */
.arch-box {
    background: #12161e;
    border: 1px solid rgba(122,171,255,0.1);
    border-radius: 10px;
    padding: 20px;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #b0bdd0;
    line-height: 1.65;
    white-space: pre;
    overflow-x: auto;
}

/* ── Metric Row ── */
.m-row {
    background: #1a1f2a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.m-label { color: #8b95a5; font-size: 0.84rem; }
.m-value { color: #7aabff; font-weight: 600; font-size: 0.88rem; }

/* ── Upload Area ── */
[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(122,171,255,0.2) !important;
    border-radius: 10px !important;
    background: rgba(122,171,255,0.03) !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] section > button {
    background: #1e2536 !important;
    color: #c8d0dc !important;
    border: 1px solid rgba(122,171,255,0.2) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] section > button:hover {
    background: #283044 !important;
}
/* Hide broken icon text in uploader */
[data-testid="stFileUploader"] [data-testid="stIconMaterial"] {
    display: none !important;
}
[data-testid="stFileUploader"] small {
    color: #6b7585 !important;
}

/* ── Primary Button ── */
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: #4f8ef7 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #3b78e0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #8b95a5;
    padding: 8px 14px;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(122,171,255,0.1);
    color: #7aabff;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f1318; }
::-webkit-scrollbar-thumb { background: #252d3a; border-radius: 3px; }

/* ── Hide branding, keeping collapsed control visible ── */
[data-testid="stHeaderDecoration"] {
    display: none !important;
}
[data-testid="stHeaderActionElements"] {
    display: none !important;
}
header {
    background: transparent !important;
}
footer { visibility: hidden; }

/* ── Divider ── */
.sep { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 14px 0; }

/* ── Analysis Reasons ── */
.reasons-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e0e4ea;
    margin: 20px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.reason-card {
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.reason-card-ok {
    background: rgba(52, 211, 153, 0.06);
    border: 1px solid rgba(52, 211, 153, 0.15);
}
.reason-card-warning {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.15);
}
.reason-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.reason-content {
    flex: 1;
}
.reason-title {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.reason-title-ok { color: #34d399; }
.reason-title-warning { color: #ef4444; }
.reason-detail {
    font-size: 0.82rem;
    color: #8b95a5;
    line-height: 1.5;
}
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 8px;
}
.metric-item {
    background: #1a1f2a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.metric-val {
    font-size: 1rem;
    font-weight: 700;
    color: #7aabff;
}
.metric-name {
    font-size: 0.68rem;
    color: #6b7585;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper to Base64 Encode and Render Images Dark ────────────
def get_base64_image_tag(image_path, alt_text="Image"):
    """Reads a local image file, encodes it as base64, and returns a styled HTML tag for dark mode."""
    import base64
    if not os.path.exists(image_path):
        return ""
    try:
        with open(image_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{b64_data}" alt="{alt_text}" style="filter: invert(0.9) hue-rotate(180deg) brightness(1.15); max-width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);" />'
    except Exception as e:
        return f'<div style="color: #ef4444; font-size: 0.85rem;">Failed to load image: {str(e)}</div>'

# ── Lazy-load predictor ─────────────────────────────────────
@st.cache_resource
def load_predictor():
    """Load the model predictor (cached across reruns)."""
    # Inlined: from src.predict import VoicePredictor
    return VoicePredictor()

# ── Sidebar Navigation ──────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 18px 0 8px;">
        <div style="display: flex; justify-content: center; align-items: center; height: 40px;">
            {SVG_MIC}
        </div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #7aabff; margin-top: 6px;">
            No-Mimic</div>
        <div style="font-size: 0.72rem; color: #6b7585; margin-top: 2px;">
            AI Deepfake Voice Detection</div>
    </div>
    <hr class="sep">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Analyze Audio", "Model Info", "Training History"],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # Model status
    model_exists = os.path.exists(config.BEST_MODEL_PATH)
    if model_exists:
        size_mb = os.path.getsize(config.BEST_MODEL_PATH) / (1024 * 1024)
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #34d399;"></div>
                <div style="color: #34d399; font-size: 0.84rem; font-weight: 600;">Model Loaded</div>
            </div>
            <div style="color: #6b7585; font-size: 0.74rem; margin-top: 5px; padding-left: 16px;">Size: {size_mb:.1f} MB</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #ef4444;"></div>
                <div style="color: #ef4444; font-size: 0.84rem; font-weight: 600;">No Model Found</div>
            </div>
            <div style="color: #6b7585; font-size: 0.74rem; margin-top: 5px; padding-left: 16px;">Train a model first</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Page: Analyze Audio
# ══════════════════════════════════════════════════════════════
def page_analyze():
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
        {SVG_SEARCH}
        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #f0f2f5;">No-Mimic</h1>
    </div>
    <div class="page-desc">Upload or record an audio file to detect AI-generated deepfake voices</div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-label">Input Source</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Select Input Source",
            ["Upload File", "Record Audio"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode"
        )

        uploaded = None
        if input_mode == "Upload File":
            st.markdown(f'<div class="section-label" style="margin-top: 15px;">{SVG_UPLOAD} Upload Audio</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop your audio file here",
                type=list(config.ALLOWED_EXTENSIONS),
                label_visibility="collapsed",
                key="file_uploader"
            )
            if uploaded_file:
                uploaded = uploaded_file
        else:
            st.markdown(f'<div class="section-label" style="margin-top: 15px;">{SVG_MIC} Record Audio</div>', unsafe_allow_html=True)
            recorded_file = st.audio_input(
                "Record audio directly from your microphone",
                label_visibility="collapsed",
                key="audio_recorder"
            )
            if recorded_file:
                uploaded = recorded_file

        if uploaded:
            # Playback is only necessary for uploads; the audio_recorder widget already provides playback controls
            if input_mode == "Upload File":
                ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else "wav"
                st.audio(uploaded, format=f"audio/{ext}")

            file_size = len(uploaded.getvalue()) / 1024
            display_name = "Voice Recording" if input_mode == "Record Audio" else uploaded.name
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-val" style="font-size:0.92rem; word-break: break-all;">{display_name}</div>
                    <div class="stat-lbl">Source / Filename</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val" style="font-size:0.92rem;">{file_size:.1f} KB</div>
                    <div class="stat-lbl">File Size</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.button("Analyze Voice", type="primary", key="analyze_btn",
                       on_click=lambda: st.session_state.update(run_analysis=True))

    with col_result:
        st.markdown(f'<div class="section-label">{SVG_CHART} Analysis Result</div>', unsafe_allow_html=True)
        if uploaded and st.session_state.get("run_analysis"):
            _run_prediction(uploaded)
            st.session_state.run_analysis = False
        elif not uploaded:
            st.markdown(f"""
            <div class="empty-state">
                <div class="icon">{SVG_MUSIC}</div>
                <div class="msg">Upload or record an audio file to get started</div>
            </div>
            """, unsafe_allow_html=True)


def _run_prediction(uploaded_file):
    """Run prediction on uploaded or recorded audio file."""
    with st.spinner("Analyzing audio..."):
        try:
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else "wav"
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            predictor = load_predictor()
            result = predictor.predict(tmp_path)
            os.unlink(tmp_path)
            _display_results(result)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def _display_results(result):
    """Display prediction results with analysis reasons."""
    is_authentic = result["prediction"] == "authentic"
    box_cls = "result-ok" if is_authentic else "result-bad"
    c_cls = "c-green" if is_authentic else "c-red"
    svg_icon = SVG_CHECK_SHIELD if is_authentic else SVG_ALERT_SHIELD
    label = result["label"]
    confidence = result["confidence"]
    bar_color = "#34d399" if is_authentic else "#ef4444"

    st.markdown(f"""
    <div class="result-box {box_cls}">
        <div class="result-icon">{svg_icon}</div>
        <div class="result-title {c_cls}">{label}</div>
        <div class="result-pct {c_cls}">{confidence}%</div>
        <div class="result-sub">Confidence Score &nbsp;|&nbsp; Raw: {result['raw_score']}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width:{confidence}%; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display analysis reasons
    if "reasons" in result:
        _display_reasons(result)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    _display_visualizations(result)


def _display_reasons(result):
    """Display human-readable reasons for the classification verdict."""
    reasons = result.get("reasons", [])
    metrics = result.get("metrics", {})
    is_authentic = result["prediction"] == "authentic"

    header_text = "Why Authentic?" if is_authentic else "Why AI-Generated?"

    st.markdown(f"""
    <div class="reasons-header">
        {SVG_REASON_SEARCH}
        <span>{header_text}</span>
    </div>
    """, unsafe_allow_html=True)

    for reason in reasons:
        r_type = reason.get("type", "ok")
        card_cls = f"reason-card-{r_type}"
        title_cls = f"reason-title-{r_type}"
        icon_key = reason.get("icon", "")
        icon_svg = REASON_ICON_MAP.get(icon_key, "")
        title = reason.get("title", "")
        detail = reason.get("detail", "")

        # Color the SVG icon based on reason type
        icon_color = "#34d399" if r_type == "ok" else "#ef4444"
        colored_icon = icon_svg.replace('stroke="currentColor"', f'stroke="{icon_color}"')

        st.markdown(f"""
        <div class="reason-card {card_cls}">
            <div class="reason-icon">{colored_icon}</div>
            <div class="reason-content">
                <div class="reason-title {title_cls}">{title}</div>
                <div class="reason-detail">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Expandable raw metrics
    if metrics:
        with st.expander("Raw Acoustic Metrics"):
            metric_labels = {
                "spectral_flatness": "Spectral Flatness",
                "energy_cv": "Energy CV",
                "mfcc_temporal_var": "MFCC Variance",
                "mfcc_delta": "MFCC Delta",
                "hf_ratio": "HF Ratio",
                "frame_smoothness": "Frame Smoothness",
            }
            items = list(metrics.items())
            # Build metrics grid
            html = '<div class="metrics-grid">'
            for key, val in items:
                label = metric_labels.get(key, key)
                html += f"""
                <div class="metric-item">
                    <div class="metric-val">{val}</div>
                    <div class="metric-name">{label}</div>
                </div>"""
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


def _display_visualizations(result):
    """Display mel spectrogram and MFCC visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG = "#0f1318"
    tab_mel, tab_mfcc = st.tabs(["Mel Spectrogram", "MFCC"])

    with tab_mel:
        if "mel_spectrogram" in result:
            mel = np.array(result["mel_spectrogram"])
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma",
                           interpolation="bilinear")
            ax.set_xlabel("Time Frame", color="white", fontsize=9)
            ax.set_ylabel("Mel Bin", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cbar.ax.tick_params(colors="white", labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab_mfcc:
        if "mfcc" in result:
            mfcc = np.array(result["mfcc"])
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis",
                           interpolation="bilinear")
            ax.set_xlabel("Time Frame", color="white", fontsize=9)
            ax.set_ylabel("MFCC Coefficient", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cbar.ax.tick_params(colors="white", labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════
# Page: Model Info
# ══════════════════════════════════════════════════════════════
def page_model_info():
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
        {SVG_CPU}
        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #f0f2f5;">Model Architecture</h1>
    </div>
    <div class="page-desc">CNN-BiLSTM with Self-Attention — Hybrid deepfake detector</div>
    """, unsafe_allow_html=True)

    import torch
    # Inlined: from src.model import VoiceAuthenticityModel

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            {SVG_LAYERS}
            <span style="font-size: 1.25rem; font-weight: 600; color: #f0f2f5;">Architecture Overview</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="arch-box">Input Audio (3s @ 16kHz)
    │
    ▼  Preprocessing
    │  Resample → Mono → Normalize → Trim → Pad
    │
    ▼  Feature Extraction
    │  Mel Spectrogram ({config.N_MELS} bins)
    │  MFCC ({config.N_MFCC} coefficients)
    │  → Dual-channel tensor (2, {config.N_MELS}, T)
    │
    ▼  CNN Blocks
    │  Conv2d({config.CNN_CHANNELS[0]}→{config.CNN_CHANNELS[1]}) → BN → ReLU → MaxPool
    │  Conv2d({config.CNN_CHANNELS[1]}→{config.CNN_CHANNELS[2]}) → BN → ReLU → MaxPool
    │  Conv2d({config.CNN_CHANNELS[2]}→{config.CNN_CHANNELS[3]}) → BN → ReLU → MaxPool
    │
    ▼  BiLSTM ({config.LSTM_NUM_LAYERS} layers, {config.LSTM_HIDDEN_SIZE} hidden)
    │
    ▼  Self-Attention Pooling
    │
    ▼  Classifier
    │  FC({config.LSTM_HIDDEN_SIZE * 2}→{config.FC_HIDDEN_SIZE}) → ReLU → Dropout
    │  FC({config.FC_HIDDEN_SIZE}→1) → Sigmoid
    │
    ▼  Output: Authentic / AI-Generated</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            {SVG_CHART}
            <span style="font-size: 1.25rem; font-weight: 600; color: #f0f2f5;">Model Statistics</span>
        </div>
        """, unsafe_allow_html=True)

        model = VoiceAuthenticityModel()
        param_count = model.count_parameters()
        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

        metrics = [
            ("Total Parameters", f"{param_count:,}"),
            ("Device", device),
            ("Input Shape", f"(B, 2, {config.N_MELS}, {config.NUM_TIME_FRAMES})"),
            ("Output", "Binary (Sigmoid)"),
            ("Loss Function", "BCEWithLogitsLoss"),
            ("Optimizer", f"AdamW (lr={config.LEARNING_RATE})"),
            ("Scheduler", "CosineAnnealingLR"),
            ("Batch Size", str(config.BATCH_SIZE)),
            ("Max Epochs", str(config.NUM_EPOCHS)),
            ("Early Stopping", f"Patience={config.EARLY_STOPPING_PATIENCE}"),
        ]

        for label, value in metrics:
            st.markdown(f"""
            <div class="m-row">
                <span class="m-label">{label}</span>
                <span class="m-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        # Checkpoint info
        if os.path.exists(config.BEST_MODEL_PATH):
            ckpt = torch.load(config.BEST_MODEL_PATH, weights_only=False, map_location="cpu")
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-top: 20px; margin-bottom: 8px;">
                {SVG_AWARD}
                <span style="font-size: 1.15rem; font-weight: 600; color: #f0f2f5;">Best Checkpoint</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
                <span class="m-label">Epoch:</span> <span class="m-value">{ckpt.get('epoch', 'N/A')}</span>
                &nbsp;|&nbsp;
                <span class="m-label">Val Loss:</span> <span class="m-value">{ckpt.get('val_loss', 0):.4f}</span>
                &nbsp;|&nbsp;
                <span class="m-label">Val Acc:</span> <span class="m-value">{ckpt.get('val_accuracy', 0):.2%}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Page: Training History
# ══════════════════════════════════════════════════════════════
def page_training_history():
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
        {SVG_CHART}
        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #f0f2f5;">Training History</h1>
    </div>
    <div class="page-desc">Loss & accuracy curves from the most recent training run</div>
    """, unsafe_allow_html=True)

    history_path = os.path.join(config.MODELS_DIR, "training_history.json")
    curves_path = os.path.join(config.MODELS_DIR, "training_curves.png")
    eval_path = os.path.join(config.MODELS_DIR, "evaluation_results.png")

    history = None

    # Try to load history from the best model checkpoint first
    if os.path.exists(config.BEST_MODEL_PATH):
        try:
            import torch
            checkpoint = torch.load(config.BEST_MODEL_PATH, weights_only=False, map_location="cpu")
            if "history" in checkpoint:
                history = checkpoint["history"]
        except Exception:
            pass

    # Fallback to training_history.json
    if history is None and os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            pass

    if history is None:
        st.warning("No training history found. Train a model first using `voice_authenticity.ipynb`.")
        return


    # Interactive plotly charts
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"),
                            horizontal_spacing=0.12)

        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                                 line=dict(color="#7aabff", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss",
                                 line=dict(color="#ef4444", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train Acc",
                                 line=dict(color="#7aabff", width=2.5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val Acc",
                                 line=dict(color="#ef4444", width=2.5)), row=1, col=2)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,19,24,0.9)",
            height=400,
            font=dict(family="Inter", color="#c8d0dc"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

        st.plotly_chart(fig, width="stretch")

    except ImportError:
        if os.path.exists(curves_path):
            img_tag = get_base64_image_tag(curves_path, "Training Curves")
            st.markdown(img_tag, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    final_epoch = len(history["train_loss"])
    best_val_loss = min(history["val_loss"])
    best_val_acc = max(history["val_acc"])
    final_train_acc = history["train_acc"][-1]

    for col, label, value in [
        (col1, "Epochs Trained", str(final_epoch)),
        (col2, "Best Val Loss", f"{best_val_loss:.4f}"),
        (col3, "Best Val Accuracy", f"{best_val_acc:.2%}"),
        (col4, "Final Train Accuracy", f"{final_train_acc:.2%}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card" style="margin:0;">
                <div class="stat-val">{value}</div>
                <div class="stat-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Evaluation results
    if os.path.exists(eval_path):
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 24px; margin-bottom: 12px;">
            {SVG_CHART}
            <span style="font-size: 1.25rem; font-weight: 600; color: #f0f2f5;">Evaluation Results</span>
        </div>
        """, unsafe_allow_html=True)
        img_tag = get_base64_image_tag(eval_path, "Confusion Matrix & ROC Curve")
        st.markdown(img_tag, unsafe_allow_html=True)


# ── Router ───────────────────────────────────────────────────
if page == "Analyze Audio":
    page_analyze()
elif page == "Model Info":
    page_model_info()
elif page == "Training History":
    page_training_history()
