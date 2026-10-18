"""
NeuroVision AI - Deep Learning Brain MRI Diagnostic & Explainability Studio.
Streamlit Application for brain tumor classification and Grad-CAM visualization.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import streamlit as st

# Add src/ to path so neurovision_ai package can be loaded seamlessly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurovision_ai.model_utils import (
    CLASSES,
    CLASS_INFO,
    find_serialized_models,
    load_mri_model,
    preprocess_image,
    run_inference,
    generate_gradcam,
    get_available_sample_images,
    create_and_save_baseline_model,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroVision AI | Brain MRI Tumor Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern Medical UI Styling
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Overall page glow background */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.05) 0%, transparent 40%);
}

/* Glassmorphic cards */
.neuro-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.12);
}

.neuro-card-highlight {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-normal {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-danger {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-purple {
    background: rgba(139, 92, 246, 0.15);
    color: #a78bfa;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

/* Metric displays */
.metric-container {
    display: flex;
    align-items: baseline;
    gap: 10px;
}

.metric-val {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

/* Custom progress bar */
.prob-bar-container {
    margin-bottom: 14px;
}

.prob-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 0.9rem;
    font-weight: 600;
}

.prob-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

/* App Header */
.app-header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.app-header-subtitle {
    color: #94a3b8;
    font-size: 1.0rem;
    margin-bottom: 24px;
}

/* Radio buttons & pills tweak */
div[role="radiogroup"] > label {
    padding: 6px 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Caching for Models
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str):
    """
    Cached model loader to avoid reloading weights on every user interaction.
    """
    return load_mri_model(model_path)


# ---------------------------------------------------------
# Sidebar: Model Management & Parameters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 NeuroVision Model Hub")
    st.caption("Manages serialized deep neural networks in `models/`")

    available_models = find_serialized_models("models")

    selected_model_path: Optional[str] = None
    model_instance = None

    if available_models:
        model_options = {m["filename"]: m["path"] for m in available_models}
        chosen_name = st.selectbox(
            "Active Model Architecture",
            options=list(model_options.keys()),
            help="Select serialized model file found in the 'models' directory.",
        )
        selected_model_path = model_options[chosen_name]

        # Selected model metadata
        chosen_info = next(m for m in available_models if m["filename"] == chosen_name)
        st.markdown(
            f"""
            <div style="font-size:0.8rem; background:rgba(255,255,255,0.03); border-radius:8px; padding:10px; margin-top:8px; border:1px solid rgba(255,255,255,0.06);">
                <div><b>File:</b> <code>{chosen_info['filename']}</code></div>
                <div><b>Size:</b> {chosen_info['size_mb']} MB</div>
                <div><b>Format:</b> {chosen_info['type']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            with st.spinner("Loading neural network weights..."):
                model_instance = get_cached_model(selected_model_path)
            st.success("Model ready for inference", icon="⚡")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("No serialized model file found in `models/`.", icon="📁")
        st.info(
            "When you finish training in `notebooks/model_training.ipynb`, save your model to `models/best_model.keras`."
        )

        if st.button("🚀 Initialize Baseline Model", type="primary", use_container_width=True):
            with st.spinner("Synthesizing model architecture into `models/baseline_model.keras`..."):
                created_path = create_and_save_baseline_model("models/baseline_model.keras")
                st.cache_resource.clear()
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Diagnostic Settings")

    confidence_threshold = st.slider(
        "Confidence Alert Threshold (%)",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
        help="Diagnoses with confidence below this threshold will flag a clinical uncertainty notice.",
    )

    colormap_choice = st.selectbox(
        "Grad-CAM Color Scheme",
        options=["Jet (Standard)", "Turbo (Enhanced)", "Inferno (Thermal)", "Viridis (Perceptual)"],
        index=0,
    )

    colormap_dict = {
        "Jet (Standard)": cv2.COLORMAP_JET,
        "Turbo (Enhanced)": cv2.COLORMAP_TURBO,
        "Inferno (Thermal)": cv2.COLORMAP_INFERNO,
        "Viridis (Perceptual)": cv2.COLORMAP_VIRIDIS,
    }
    selected_colormap = colormap_dict[colormap_choice]

    heatmap_alpha = st.slider(
        "Heatmap Overlay Blend (α)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust transparency between original MRI and Grad-CAM activation.",
    )

    st.markdown("---")
    st.markdown("### 📋 Classes Monitored")
    for cls in CLASSES:
        info = CLASS_INFO[cls]
        st.markdown(f"• **{info['display_name']}** (`{cls}`)")

    st.markdown("---")
    st.caption("NeuroVision AI v0.1.0 • Deep Learning Diagnostic Assistant")


# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
col_header_left, col_header_right = st.columns([3, 1])

with col_header_left:
    st.markdown('<div class="app-header-title">NeuroVision AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-header-subtitle">Intelligent Brain MRI Classification & Grad-CAM Visual Explainability Platform</div>',
        unsafe_allow_html=True,
    )

with col_header_right:
    status_color = "#10b981" if model_instance is not None else "#f59e0b"
    status_text = "System Online" if model_instance is not None else "Awaiting Model"
    st.markdown(
        f"""
        <div style="text-align:right; padding-top:12px;">
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{status_color}; margin-right:6px;"></span>
            <span style="color:#94a3b8; font-size:0.88rem; font-weight:600;">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Image Selection: File Uploader or Dataset Samples
# ---------------------------------------------------------
st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
st.markdown("#### 📥 Select or Upload Brain MRI Scan")

source_tab1, source_tab2 = st.tabs(["📤 Upload Custom MRI Scan", "🖼️ Quick Sample Gallery (Dataset)"])

selected_image_input = None
scan_source_name = "Uploaded MRI"

with source_tab1:
    uploaded_file = st.file_uploader(
        "Upload brain MRI image (axial, sagittal, or coronal)",
        type=["png", "jpg", "jpeg"],
        help="Upload standard DICOM-exported MRI scans in PNG or JPG format.",
    )
    if uploaded_file is not None:
        selected_image_input = uploaded_file
        scan_source_name = f"File: {uploaded_file.name}"

with source_tab2:
    samples_by_class = get_available_sample_images(["data/processed", "data/raw"])
    has_any_samples = any(len(v) > 0 for v in samples_by_class.values())

    if has_any_samples:
        col_cat, col_img = st.columns([1, 2])
        with col_cat:
            sample_class = st.selectbox(
                "Filter Sample by Condition",
                options=CLASSES,
                format_func=lambda x: CLASS_INFO[x]["display_name"],
            )
        with col_img:
            avail_images = samples_by_class.get(sample_class, [])
            if avail_images:
                chosen_sample_path = st.selectbox(
                    "Choose Sample MRI Scan",
                    options=avail_images,
                    format_func=lambda p: Path(p).name,
                )
                if st.button("🔍 Load Selected Sample", type="secondary", use_container_width=True):
                    st.session_state["active_sample_path"] = chosen_sample_path

        if "active_sample_path" in st.session_state and uploaded_file is None:
            active_p = Path(st.session_state["active_sample_path"])
            if active_p.exists():
                selected_image_input = Image.open(active_p)
                scan_source_name = f"Sample: {active_p.parent.name}/{active_p.name}"
    else:
        st.info("No sample images located in `data/processed` or `data/raw`.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Processing & Display Section
# ---------------------------------------------------------
if selected_image_input is not None:
    try:
        raw_rgb, batch_tensor = preprocess_image(selected_image_input)
    except Exception as e:
        st.error(f"Failed to process image: {e}")
        st.stop()

    # Image enhancement tools
    with st.expander("🛠️ MRI Image Radiologist Controls (Contrast, Brightness, Inversion)", expanded=False):
        c_enh1, c_enh2, c_enh3 = st.columns(3)
        with c_enh1:
            brightness_val = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
        with c_enh2:
            contrast_val = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
        with c_enh3:
            invert_colors = st.checkbox("Invert Colors (Dark/Light)", value=False)
            show_edges = st.checkbox("Contour / Edge Filter (Sobel)", value=False)

    # Apply adjustments for display
    display_pil = Image.fromarray(raw_rgb)
    if brightness_val != 1.0:
        display_pil = ImageEnhance.Brightness(display_pil).enhance(brightness_val)
    if contrast_val != 1.0:
        display_pil = ImageEnhance.Contrast(display_pil).enhance(contrast_val)
    if invert_colors:
        display_pil = ImageOps.invert(display_pil)

    display_rgb = np.array(display_pil)

    if show_edges:
        gray = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(255 * (magnitude / (np.max(magnitude) + 1e-8)))
        display_rgb = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)

    # ---------------------------------------------------------
    # Inference execution
    # ---------------------------------------------------------
    if model_instance is None:
        st.warning(
            "Please select or initialize a model in the sidebar to run diagnostic prediction and Grad-CAM.",
            icon="⚠️",
        )
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(display_rgb, caption=f"Active Scan: {scan_source_name}", use_container_width=True)
        with col_info:
            st.info("Model weights not yet loaded. Awaiting model selection.")
        st.stop()

    with st.spinner("Analyzing MRI parenchyma and calculating spatial gradients..."):
        results = run_inference(model_instance, batch_tensor)
        pred_class = results["predicted_class"]
        pred_conf = results["confidence"]
        probabilities = results["probabilities"]
        class_idx = results["class_index"]
        pred_info = CLASS_INFO[pred_class]

    # Two column layout: Left = MRI & Grad-CAM, Right = Diagnostic Findings
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
        st.markdown(f"#### 🔍 Brain MRI Scan: `{scan_source_name}`")

        # Tabs for original vs Explainability
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Original MRI", "🔥 Grad-CAM Overlay", "Side-by-Side Comparison"])

        with viz_tab1:
            st.image(display_rgb, caption="Preprocessed Scan (Input resolution: 224x224)", use_container_width=True)

        # Generate Grad-CAM
        try:
            heatmap_rgb, gradcam_overlay = generate_gradcam(
                model=model_instance,
                batch_tensor=batch_tensor,
                target_class_idx=class_idx,
                original_rgb=raw_rgb,
                colormap=selected_colormap,
                alpha=heatmap_alpha,
            )
            gradcam_success = True
        except Exception as e:
            gradcam_success = False
            gradcam_error = str(e)

        with viz_tab2:
            if gradcam_success:
                st.image(
                    gradcam_overlay,
                    caption=f"Grad-CAM Heatmap overlay for predicted: {pred_info['display_name']} (α={heatmap_alpha})",
                    use_container_width=True,
                )
                st.caption(
                    "🔴 Warmer regions (Red/Yellow) highlight focal anatomical features driving the neural network's decision."
                )
            else:
                st.error(f"Grad-CAM generation error: {gradcam_error}")

        with viz_tab3:
            if gradcam_success:
                c_orig, c_heat, c_over = st.columns(3)
                with c_orig:
                    st.caption("1. Input MRI")
                    st.image(raw_rgb, use_container_width=True)
                with c_heat:
                    st.caption("2. Raw Heatmap")
                    st.image(heatmap_rgb, use_container_width=True)
                with c_over:
                    st.caption("3. Superimposed")
                    st.image(gradcam_overlay, use_container_width=True)
            else:
                st.info("Grad-CAM comparison not available.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Diagnostic Assessment Card
        card_class = "neuro-card-highlight" if pred_class != "notumor" else "neuro-card"
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

        # Severity Badge
        badge_type = (
            "badge-normal" if pred_class == "notumor"
            else "badge-danger" if pred_class == "glioma"
            else "badge-warning" if pred_class == "meningioma"
            else "badge-purple"
        )

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span class="badge {badge_type}">{pred_info['icon']} {pred_info['category']}</span>
                <span style="font-size:0.85rem; color:#94a3b8;">Severity: <b>{pred_info['severity']}</b></span>
            </div>
            <div class="metric-container">
                <div class="metric-val" style="color:{pred_info['badge_color']};">{pred_info['display_name']}</div>
            </div>
            <div style="margin-top:4px; font-size:1.05rem; color:#e2e8f0;">
                Diagnostic Confidence: <b>{pred_conf * 100:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Threshold notice
        if (pred_conf * 100) < confidence_threshold:
            st.warning(
                f"Confidence ({pred_conf * 100:.1f}%) is below your alert threshold ({confidence_threshold}%). "
                "Second radiologist opinion and follow-up high-resolution imaging strongly advised.",
                icon="⚠️",
            )

        st.markdown("---")
        st.markdown("##### 📊 Differential Probability Breakdown")

        # Visual progress bars for all classes
        for cls in CLASSES:
            prob = probabilities[cls]
            pct = prob * 100
            info = CLASS_INFO[cls]
            is_winner = cls == pred_class
            weight_text = "bold" if is_winner else "normal"
            bar_color = info["badge_color"]

            st.markdown(
                f"""
                <div class="prob-bar-container">
                    <div class="prob-bar-header">
                        <span style="font-weight:{weight_text}; color:{'#f8fafc' if is_winner else '#94a3b8'};">
                            {info['display_name']} {(' (Primary)' if is_winner else '')}
                        </span>
                        <span style="font-family:'JetBrains Mono', monospace; font-weight:{weight_text};">
                            {pct:.2f}%
                        </span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{max(pct, 1.5):.2f}%; background:{bar_color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Comprehensive Clinical Monograph & Next Steps
    # ---------------------------------------------------------
    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    st.markdown(f"### 🩺 Clinical Overview: {pred_info['display_name']}")

    col_desc, col_symp = st.columns([1.2, 1], gap="medium")
    with col_desc:
        st.markdown("##### 🔬 Pathophysiology & Description")
        st.write(pred_info["description"])

        st.markdown("##### 🩻 Hallmark MRI Findings")
        st.write(pred_info["mri_features"])

    with col_symp:
        st.markdown("##### 🚨 Typical Clinical Manifestations")
        for symp in pred_info["typical_symptoms"]:
            st.markdown(f"• {symp}")

        st.markdown("##### 📋 Recommended Clinical Follow-up")
        st.info(pred_info["clinical_recommendation"], icon="ℹ️")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Diagnostic Report Export
    # ---------------------------------------------------------
    st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Export Diagnostic Findings")

    report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    report_markdown = f"""# NeuroVision AI Diagnostic Assessment Report
Generated: {report_timestamp}
Scan Source: {scan_source_name}
Model: {selected_model_path if selected_model_path else 'Baseline Model'}

## Diagnostic Summary
- Primary Finding: {pred_info['display_name']} ({pred_class})
- Classification Category: {pred_info['category']}
- Confidence Score: {pred_conf * 100:.2f}%
- Clinical Severity: {pred_info['severity']}

## Differential Probabilities
"""
    for cls in CLASSES:
        report_markdown += f"- {CLASS_INFO[cls]['display_name']}: {probabilities[cls] * 100:.2f}%\n"

    report_markdown += f"""
## Clinical Monograph
{pred_info['description']}

### Hallmark MRI Findings
{pred_info['mri_features']}

### Recommended Next Steps
{pred_info['clinical_recommendation']}

---
*DISCLAIMER: This report is generated automatically by an artificial intelligence research prototype for investigational and educational use only. It does not constitute a certified medical diagnosis and should never replace qualified clinical judgement.*
"""

    report_json = json.dumps(
        {
            "timestamp": report_timestamp,
            "scan_source": scan_source_name,
            "model_path": selected_model_path,
            "primary_diagnosis": {
                "class": pred_class,
                "display_name": pred_info["display_name"],
                "confidence": float(pred_conf),
                "severity": pred_info["severity"],
            },
            "differential_probabilities": {k: float(v) for k, v in probabilities.items()},
            "clinical_recommendation": pred_info["clinical_recommendation"],
        },
        indent=2,
    )

    c_rep1, c_rep2 = st.columns(2)
    with c_rep1:
        st.download_button(
            label="📥 Download Clinical Report (Markdown)",
            data=report_markdown,
            file_name=f"neurovision_report_{datetime.date.today()}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c_rep2:
        st.download_button(
            label="📥 Download Structured Data (JSON)",
            data=report_json,
            file_name=f"neurovision_data_{datetime.date.today()}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty State: Friendly Guide
    st.markdown('<div class="neuro-card" style="text-align:center; padding:48px 24px;">', unsafe_allow_html=True)
    st.markdown("### 🧠 Welcome to NeuroVision AI")
    st.markdown(
        "Upload a brain MRI scan above or choose a sample image from the dataset gallery to begin deep learning classification and explainability analysis."
    )
    st.markdown(
        """
        <div style="display:flex; justify-content:center; gap:16px; margin-top:20px; flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; width:220px; text-align:left;">
                <div style="font-size:1.4rem;">🔬</div>
                <div style="font-weight:700; margin-top:6px;">Multi-Class CNN</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Detects Glioma, Meningioma, Pituitary tumor, or Normal MRI scans.</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; width:220px; text-align:left;">
                <div style="font-size:1.4rem;">🔥</div>
                <div style="font-weight:700; margin-top:6px;">Grad-CAM Explainability</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Visualizes exact neural activation heatmaps on intracranial parenchyma.</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; width:220px; text-align:left;">
                <div style="font-size:1.4rem;">🩺</div>
                <div style="font-weight:700; margin-top:6px;">Clinical Insights</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Comprehensive symptomology, radiological features, and next clinical steps.</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; width:220px; text-align:left;">
                <div style="font-size:1.4rem;">📁</div>
                <div style="font-weight:700; margin-top:6px;">Model Serialization</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Automatically detects and hot-swaps trained models in the <code>models/</code> folder.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Footer & Clinical Disclaimer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="font-size:0.78rem; color:#64748b; line-height:1.6; text-align:center; padding-bottom:24px;">
        <b>⚠️ MEDICAL DISCLAIMER:</b> NeuroVision AI is an artificial intelligence research and clinical decision support demonstration tool.
        It is intended solely for investigational, educational, and research purposes and is NOT cleared or approved by any medical device regulatory agency
        for primary diagnosis. All predictions, confidence scores, and Grad-CAM heatmaps must be verified by a board-certified radiologist or medical professional.
    </div>
    """,
    unsafe_allow_html=True,
)
