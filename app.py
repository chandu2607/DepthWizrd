import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2
import torch
import streamlit as st
import streamlit.components.v1 as components
import rasterio
import rasterio.transform
import pyvista as pv

# System path setup
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from depthwizard.config import DepthConfig, TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.data.raster_loader import load_raster_input, RasterInput
from depthwizard.calibration import CalibrationEngine, CalibrationMode, CalibrationResult
from depthwizard.analysis.slope import compute_slope
from depthwizard.analysis.height import analyze_building_massing, probe_point_elevation
from depthwizard.metrics.validation import run_validation
from depthwizard.viz.interactive_viewer import generate_interactive_webgl_html, generate_footprint_debug
from scripts.run_phase29_peak_recovery import PeakRecoveryMLP

pv.OFF_SCREEN = True
DATA_DIR = Path("data/dfc2023_multicity")

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DepthWizard — Single-View 3D Elevation & Flythrough",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium Dark-Mode CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

body, .stApp { background: #0D1117; color: #C9D1D9; }

/* ─── Hero ─────────────────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 60%, #1c1010 100%);
    border: 1px solid #30363D;
    border-radius: 16px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.8rem;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #FF4B4B 0%, #FF8C8C 50%, #FFB347 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-tagline {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #C9D1D9;
    margin: 0.5rem 0 0.4rem;
}
.hero-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #8B949E;
    margin-top: 0.3rem;
    max-width: 780px;
}
.hero-pills { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.pill {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(63,185,80,0.12);
    color: #3FB950;
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    letter-spacing: 0.3px;
}
.pill-red {
    background: rgba(255,75,75,0.12);
    color: #FF6B6B;
    border-color: rgba(255,75,75,0.3);
}
.pill-blue {
    background: rgba(88,166,255,0.12);
    color: #58A6FF;
    border-color: rgba(88,166,255,0.3);
}

/* ─── Section headers ───────────────────────────────────────────── */
.section-header {
    font-family: 'Outfit', sans-serif;
    color: #E6EDF3;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 0.7rem;
    border-left: 3px solid #FF4B4B;
    padding-left: 0.75rem;
}

/* ─── Metric cards ──────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #161B22 0%, #1C2128 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #FF4B4B66; }
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #3FB950;
    line-height: 1.1;
}
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #8B949E;
    margin-top: 0.25rem;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ─── Status banners ────────────────────────────────────────────── */
.mode-abs {
    background: rgba(63,185,80,0.15);
    border: 1px solid #3FB950;
    color: #3FB950;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}
.mode-rel {
    background: rgba(210,153,34,0.15);
    border: 1px solid #D29922;
    color: #E3B341;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}
.input-meta {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #C9D1D9;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Global Engine Initialization
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_depth_backbone():
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    return DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)

@st.cache_resource
def get_calibration_engine():
    return CalibrationEngine(runs_dir=Path("runs"))

try:
    depth_model = get_depth_backbone()
    calib_engine = get_calibration_engine()
    models_ready = True
except Exception as e:
    models_ready = False
    model_err = str(e)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚡ DepthWizard")
st.sidebar.caption("SIH Problem Statement 26175")
st.sidebar.markdown("---")

demo_scene = st.sidebar.button("🏙️ Load Demo Scene (NYC)", use_container_width=True)
st.sidebar.markdown("**Or upload an optical image:**")
uploaded_file = st.sidebar.file_uploader(
    "Upload RGB Satellite / Drone image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Calibration Settings")
calib_choice = st.sidebar.selectbox(
    "Calibration Mode",
    [
        CalibrationMode.AUTO.value,
        CalibrationMode.STRUCTURAL_PRIOR.value,
        CalibrationMode.DEM_ANCHORED.value,
        CalibrationMode.GROUND_REFERENCED.value,
        CalibrationMode.GCP_ANCHORED.value,
        CalibrationMode.MONOCULAR_RELATIVE.value
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 3D Render Controls")
exaggeration = st.sidebar.select_slider("Vertical Exaggeration", options=[1.0, 1.5, 2.0, 3.0], value=1.0)
camera_angle = st.sidebar.selectbox("Camera Preset", ["City Overview", "Urban Oblique", "Inspection", "Top-Down", "Pedestrian"])
render_mode = st.sidebar.selectbox("Render Mode", ["RGB City", "Elevation Colormap", "Building Height", "Terrain Slope"])

# ──────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
  <div class='hero-title'>🌐 DepthWizard</div>
  <div class='hero-tagline'>Single-View Height Estimation & 3D Flythrough Platform</div>
  <div class='hero-desc'>
    Convert single optical satellite or drone imagery into verified Digital Surface Models (DSM)
    and an interactive 3D WebGL flythrough environment with structural height, slope, and validation analytics.
  </div>
  <div class='hero-pills'>
    <span class='pill'>✓ Depth Anything V2</span>
    <span class='pill'>✓ Phase 29 PeakRecovery MLP</span>
    <span class='pill'>✓ Phase 30 DTM / nDSM</span>
    <span class='pill pill-blue'>✓ Three.js WebGL 60FPS</span>
    <span class='pill pill-red'>SIH 2025 Submission</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error(f"⚠️ Model resources failed to initialize: {model_err}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Input Handling & Routing
# ──────────────────────────────────────────────────────────────────────────────
if demo_scene:
    fname = "SV_NewYork_40.7401_-73.9915.tif"
    rgb_path = DATA_DIR / "rgb" / fname
    if not rgb_path.exists():
        rgb_path = Path("demo/demo_rgb.tif")
    if rgb_path.exists():
        raster_in = load_raster_input(rgb_path, filename=fname)
        st.session_state["raster_input"] = raster_in
        for k in ["calib_result", "slope_result", "massing_df", "val_report", "render_cache"]:
            st.session_state.pop(k, None)
    else:
        st.error("Demo file not found. Please verify data/ or demo/ folder.")

elif uploaded_file is not None:
    if st.session_state.get("active_filename") != uploaded_file.name:
        raw_bytes = uploaded_file.read()
        raster_in = load_raster_input(raw_bytes, filename=uploaded_file.name)
        st.session_state["raster_input"] = raster_in
        st.session_state["active_filename"] = uploaded_file.name
        for k in ["calib_result", "slope_result", "massing_df", "val_report", "render_cache"]:
            st.session_state.pop(k, None)

raster_input: Optional[RasterInput] = st.session_state.get("raster_input")

if raster_input is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🛰️ **Single-View Optical**\n\nNo stereo pairs, LiDAR hardware, or multi-view passes required.")
    with c2:
        st.info("📐 **Metric & Relative DSM**\n\nAbsolute elevation in meters or relative structural scale (rDSM).")
    with c3:
        st.info("🏙️ **Interactive 3D Flythrough**\n\nFull 60fps orbit, pan, zoom, and WASD first-person navigation.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Click 'Load Demo Scene (NYC)' in the sidebar** or upload your own satellite raster to begin.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Section 1: Input Specification
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>1. Ingested Input Specifications</div>", unsafe_allow_html=True)
sc1, sc2 = st.columns([1, 1])
with sc1:
    st.image(raster_input.rgb, caption=raster_input.filename, use_container_width=True)
with sc2:
    summ = raster_input.get_summary()
    if raster_input.is_georeferenced:
        st.markdown(f"""
        <div class='mode-abs'>✅ ABSOLUTE DSM MODE — Georeferenced</div>
        <div class='input-meta'>
        <b>File:</b> {summ['filename']}<br>
        <b>Dimensions:</b> {summ['dimensions']}<br>
        <b>CRS:</b> {summ['crs']}<br>
        <b>GSD:</b> {summ['gsd_m']} per pixel<br>
        <b>Bounds:</b> {summ['bounds']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='mode-rel'>⚠️ RELATIVE ELEVATION MODE (rDSM) — Non-Georeferenced</div>
        <div class='input-meta'>
        <b>File:</b> {summ['filename']}<br>
        <b>Dimensions:</b> {summ['dimensions']}<br>
        <b>Spatial Reference:</b> Not present (PNG/JPG input)<br>
        <b>Output Target:</b> Relative Digital Surface Model (rDSM, normalized 0–10 scale)
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 2: Reconstruct Elevation Surface
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>2. Reconstruct Elevation Surface</div>", unsafe_allow_html=True)
run_btn = st.button("🚀 EXECUTE DEPTHWIZARD PIPELINE", type="primary")

# Re-run pipeline if calibration mode changed in sidebar
calib_changed = st.session_state.get("active_calib_choice") != calib_choice

if run_btn or calib_changed or "calib_result" not in st.session_state:
    with st.spinner(f"⏳ Running Depth Backbone & Calibration Engine ({calib_choice})…"):
        t0 = time.perf_counter()
        st.session_state["active_calib_choice"] = calib_choice
        
        # 1. Monocular Depth Inference
        h, w = raster_input.shape
        depth_raw = depth_model.infer(raster_input.rgb, raster_input.filename, target_hw=(h, w))
        
        # 2. Reference Elevation Lookup (for demo/georeferenced evaluation)
        ref_elevation = None
        dsm_truth_path = DATA_DIR / "dsm" / raster_input.filename
        if not dsm_truth_path.exists() and raster_input.filename == "SV_NewYork_40.7401_-73.9915.tif":
            dsm_truth_path = Path("demo/demo_dsm.tif")
        if dsm_truth_path.exists():
            ref_elevation = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # 3. Calibration Execution
        mode_enum = next((m for m in CalibrationMode if m.value == calib_choice), CalibrationMode.AUTO)
        calib_res = calib_engine.calibrate(
            depth_raw=depth_raw,
            rgb=raster_input.rgb,
            is_georeferenced=raster_input.is_georeferenced,
            mode=mode_enum,
            reference_elevation=ref_elevation,
            filename=raster_input.filename
        )
        
        # 4. Slope & Structural Analysis
        slope_res = compute_slope(calib_res.dsm, gsd_x=raster_input.gsd[0], gsd_y=raster_input.gsd[1], mask_bldg=calib_res.mask_bldg)
        massing_df = analyze_building_massing(calib_res.dsm, calib_res.dtm, calib_res.mask_bldg, gsd=raster_input.gsd)
        
        # 5. Validation Execution (if GT available)
        val_rep = None
        if ref_elevation is not None and calib_res.is_metric:
            val_rep = run_validation(calib_res.dsm, ref_elevation)
            
        t_elapsed = round(time.perf_counter() - t0, 2)
        
        st.session_state["calib_result"] = calib_res
        st.session_state["depth_raw"] = depth_raw
        st.session_state["slope_result"] = slope_res
        st.session_state["massing_df"] = massing_df
        st.session_state["val_report"] = val_rep
        st.session_state["pipeline_time_s"] = t_elapsed

calib_res: CalibrationResult = st.session_state["calib_result"]
depth_raw = st.session_state["depth_raw"]
slope_res = st.session_state["slope_result"]
massing_df = st.session_state["massing_df"]
val_rep = st.session_state["val_report"]

st.success(f"⚡ Processing completed in **{st.session_state.get('pipeline_time_s', 0.5)}s** using **{calib_res.provenance.get('calibration_mode', 'Auto')}**")

# ──────────────────────────────────────────────────────────────────────────────
# Section 3: Interactive 3D WebGL Flythrough (HERO COMPONENT)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>3. Interactive 3D WebGL Flythrough & Orbit Viewer</div>", unsafe_allow_html=True)
st.caption("🎮 **Full 60FPS WebGL Interaction**: Click and drag to orbit · Right-click to pan · Scroll to zoom · Use **WASD/Arrow keys** for ground navigation · Click **Cinematic Flythrough**.")

preset_map = {
    "City Overview": "overview",
    "Urban Oblique": "urban",
    "Inspection": "inspection",
    "Top-Down": "top",
    "Pedestrian": "street"
}
mode_map = {
    "RGB City": "rgb",
    "Elevation Colormap": "elev",
    "Building Height": "height",
    "Terrain Slope": "slope"
}

webgl_html = generate_interactive_webgl_html(
    rgb_img=raster_input.rgb,
    dsm=calib_res.dsm,
    dtm=calib_res.dtm,
    mask_bldg=calib_res.mask_bldg,
    gsd=raster_input.gsd,
    exaggeration=exaggeration,
    stride=4,
    default_preset=preset_map.get(camera_angle, "overview"),
    default_mode=mode_map.get(render_mode, "rgb")
)
components.html(webgl_html, height=720)

with st.expander("🔍 Inspection & Geometry Debug: Extracted Building Footprints", expanded=False):
    fp_debug_img = generate_footprint_debug(raster_input.rgb, calib_res.mask_bldg, gsd=raster_input.gsd)
    st.image(fp_debug_img, caption="Building Footprints (Green = Valid Building Object, Red = Rejected Mega-Component)", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 4: 2D Surface, Depth, and Slope Maps
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>4. Multi-Layer 2D Raster Suite</div>", unsafe_allow_html=True)
c2a, c2b, c2c = st.columns(3)
with c2a:
    d_vis = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-6)
    st.image(d_vis, caption="Monocular Relative Depth (Depth Anything V2)", use_container_width=True, clamp=True)
with c2b:
    dsm_vis = (calib_res.dsm - calib_res.dsm.min()) / (calib_res.dsm.max() - calib_res.dsm.min() + 1e-6)
    st.image(dsm_vis, caption=f"Reconstructed Surface ({calib_res.units})", use_container_width=True, clamp=True)
with c2c:
    slope_vis = (slope_res.slope_deg / 60.0).clip(0, 1)
    st.image(slope_vis, caption=f"Terrain Slope Map (Mean: {slope_res.stats['mean_terrain_slope_deg']}°)", use_container_width=True, clamp=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 5: Height & Structural Statistics Dashboard
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>5. Elevation & Structural Statistics</div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)
unit_s = "m" if calib_res.is_metric else "rel"
stat_list = [
    (m1, f"{calib_res.stats.get('min', 0.0):.1f} {unit_s}", "Min Surface Z"),
    (m2, f"{calib_res.stats.get('max', 0.0):.1f} {unit_s}", "Max Surface Z"),
    (m3, f"{calib_res.stats.get('mean', 0.0):.1f} {unit_s}", "Mean Surface Z"),
    (m4, f"{calib_res.stats.get('p95', 0.0):.1f} {unit_s}", "P95 Elevation"),
    (m5, f"{calib_res.ndsm.max():.1f} {unit_s}", "Max Structure (nDSM)"),
    (m6, f"{len(massing_df)}", "Extruded Buildings"),
]
for col, val, lbl in stat_list:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{val}</div>
          <div class='metric-label'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 6: Structural Height Measurement & Massing Analysis
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>6. Building Massing & Interactive Height Measurement</div>", unsafe_allow_html=True)
tab_bldg, tab_probe = st.tabs(["🏢 Building Massing Table", "📍 Point Elevation Probe"])

with tab_bldg:
    if not massing_df.empty:
        st.dataframe(massing_df, use_container_width=True, height=260)
        st.caption(f"Showing {len(massing_df)} segmented urban structures. Height $H = Z_{{\\text{{roof}}}} - Z_{{\\text{{ground}}}}$.")
    else:
        st.info("No discrete buildings detected in this raster.")

with tab_probe:
    col_px, col_py, col_probe_out = st.columns([1, 1, 2])
    with col_px:
        x_probe = st.number_input("Pixel X Coordinate", min_value=0, max_value=raster_input.width-1, value=raster_input.width//2)
    with col_py:
        y_probe = st.number_input("Pixel Y Coordinate", min_value=0, max_value=raster_input.height-1, value=raster_input.height//2)
    with col_probe_out:
        probe_res = probe_point_elevation(calib_res.dsm, calib_res.dtm, calib_res.mask_bldg, x_probe, y_probe, is_metric=calib_res.is_metric)
        st.markdown(f"""
        <div class='input-meta'>
        <b>Target:</b> ({probe_res['x']}, {probe_res['y']}) &nbsp;|&nbsp; <b>Class:</b> {'Building' if probe_res['is_building'] else 'Ground Terrain'}<br>
        <b>Surface Elevation:</b> {probe_res['elevation']}<br>
        <b>Ground Elevation (DTM):</b> {probe_res['ground_elevation']}<br>
        <b>Structural Height:</b> <b style='color:#3FB950'>{probe_res['structural_height']}</b>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 7: Terrain & Facade Slope Analysis
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>7. Terrain & Facade Slope Analysis</div>", unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{slope_res.stats['mean_terrain_slope_deg']}°</div><div class='metric-label'>Mean Terrain Slope</div></div>", unsafe_allow_html=True)
with s2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{slope_res.stats['p95_terrain_slope_deg']}°</div><div class='metric-label'>P95 Terrain Slope</div></div>", unsafe_allow_html=True)
with s3:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{slope_res.stats['steep_slope_pct_area']}%</div><div class='metric-label'>Steep Slope Area (>25°)</div></div>", unsafe_allow_html=True)
with s4:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{slope_res.stats['max_slope_deg']}°</div><div class='metric-label'>Max Facade Gradient</div></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 8: Quantitative Validation Dashboard
# ──────────────────────────────────────────────────────────────────────────────
if val_rep is not None:
    st.markdown("<div class='section-header'>8. Quantitative Validation vs Reference Ground Truth</div>", unsafe_allow_html=True)
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{val_rep.summary_metrics['MAE (m)']} m</div><div class='metric-label'>MAE</div></div>", unsafe_allow_html=True)
    with v2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{val_rep.summary_metrics['RMSE (m)']} m</div><div class='metric-label'>RMSE</div></div>", unsafe_allow_html=True)
    with v3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{val_rep.summary_metrics['Pearson R']}</div><div class='metric-label'>Pearson R</div></div>", unsafe_allow_html=True)
    with v4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{val_rep.summary_metrics['P90 AE (m)']} m</div><div class='metric-label'>P90 Abs Error</div></div>", unsafe_allow_html=True)
    with v5:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{val_rep.summary_metrics['Bias (m)']} m</div><div class='metric-label'>Mean Bias</div></div>", unsafe_allow_html=True)
    
    st.markdown("#### Binned Height Accuracy Breakdown")
    st.dataframe(val_rep.binned_table, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Section 9: Export Assets
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>9. Export Assets & Scientific Deliverables</div>", unsafe_allow_html=True)

def write_geotiff(arr, is_geo, transform, crs):
    h, w = arr.shape
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": -999.0,
        "width": w, "height": h, "count": 1,
        "crs": crs if is_geo else None,
        "transform": transform if is_geo else rasterio.transform.from_origin(0, h, 1, 1),
    }
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
        p = f.name
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    with open(p, "rb") as f:
        data = f.read()
    try: os.remove(p)
    except: pass
    return data

ex1, ex2, ex3, ex4 = st.columns(4)
with ex1:
    st.download_button("⬇️ Download DSM GeoTIFF",
                       data=write_geotiff(calib_res.dsm, raster_input.is_georeferenced, raster_input.transform, raster_input.crs),
                       file_name=f"DSM_{raster_input.filename}",
                       mime="image/tiff", use_container_width=True)
with ex2:
    st.download_button("⬇️ Download nDSM GeoTIFF",
                       data=write_geotiff(calib_res.ndsm, raster_input.is_georeferenced, raster_input.transform, raster_input.crs),
                       file_name=f"nDSM_{raster_input.filename}",
                       mime="image/tiff", use_container_width=True)
with ex3:
    st.download_button("⬇️ Download Building Massing (CSV)",
                       data=massing_df.to_csv(index=False),
                       file_name=f"Buildings_{Path(raster_input.filename).stem}.csv",
                       mime="text/csv", use_container_width=True)
with ex4:
    if val_rep is not None:
        st.download_button("⬇️ Download Validation Metrics (JSON)",
                           data=json.dumps(val_rep.summary_metrics, indent=2),
                           file_name=f"Validation_{Path(raster_input.filename).stem}.json",
                           mime="application/json", use_container_width=True)
    else:
        st.download_button("⬇️ Download Metadata Summary (JSON)",
                           data=json.dumps(raster_input.get_summary(), indent=2),
                           file_name=f"Meta_{Path(raster_input.filename).stem}.json",
                           mime="application/json", use_container_width=True)
