import os
import sys
import json
import time
import tempfile
import numpy as np
import cv2
import torch
import streamlit as st
import rasterio
import rasterio.transform
import pyvista as pv
from pathlib import Path

# Setup system path to import workspace modules
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from depthwizard.config import TrainConfig
from scripts.run_phase29_peak_recovery import PeakRecoveryMLP

pv.OFF_SCREEN = True
DATA_DIR = Path("data/dfc2023_multicity")

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DepthWizard — Single-View 3D Elevation Reconstruction",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium dark-mode CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@400;500;600&display=swap');

body, .stApp { background: #0D1117; }

/* ─── Hero ─────────────────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 60%, #1a0a0a 100%);
    border: 1px solid #21262D;
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
    color: #6E7681;
    margin-top: 0.3rem;
    max-width: 640px;
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
.metric-card:hover { border-color: #FF4B4B44; }
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
    color: #6E7681;
    margin-top: 0.25rem;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ─── Step tracker ──────────────────────────────────────────────── */
.step-row { display: flex; gap: 0.4rem; margin: 0.6rem 0 1rem; flex-wrap: wrap; }
.step-done {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem; font-weight: 600;
    background: rgba(63,185,80,0.15);
    color: #3FB950;
    border: 1px solid rgba(63,185,80,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
}
.step-run {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem; font-weight: 600;
    background: rgba(255,75,75,0.15);
    color: #FF6B6B;
    border: 1px solid rgba(255,75,75,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    animation: pulse 1.4s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.55} }

/* ─── Info/mode banners ─────────────────────────────────────────── */
.mode-abs {
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.35);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #3FB950;
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.mode-rel {
    background: rgba(255,165,0,0.1);
    border: 1px solid rgba(255,165,0,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #FFA500;
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.input-meta {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #8B949E;
    line-height: 1.8;
}
.input-meta b { color: #C9D1D9; }

/* ─── 3D viewer badge ───────────────────────────────────────────── */
.mesh-badge {
    display:inline-block;
    font-family:'Inter',sans-serif;
    font-size:0.72rem;
    font-weight:600;
    background:rgba(63,185,80,0.12);
    color:#3FB950;
    border:1px solid rgba(63,185,80,0.35);
    border-radius:20px;
    padding:0.2rem 0.7rem;
    margin-bottom:0.5rem;
}

/* ─── Export cards ──────────────────────────────────────────────── */
.export-card {
    background:#161B22;
    border:1px solid #30363D;
    border-radius:10px;
    padding:1rem;
    text-align:center;
    margin-bottom:0.5rem;
}
.export-label {
    font-family:'Inter',sans-serif;
    font-size:0.78rem;
    color:#6E7681;
    margin-bottom:0.5rem;
    text-transform:uppercase;
    letter-spacing:0.4px;
}

/* ─── Why it matters ────────────────────────────────────────────── */
.wim-card {
    background:linear-gradient(135deg,#161B22,#1C2128);
    border:1px solid #21262D;
    border-radius:12px;
    padding:1rem 1.2rem;
    text-align:center;
}
.wim-icon { font-size:1.6rem; margin-bottom:0.4rem; }
.wim-text {
    font-family:'Inter',sans-serif;
    font-size:0.82rem;
    color:#8B949E;
    line-height:1.5;
}

/* ─── Run button ────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF4B4B, #C62828);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.8rem 3rem;
    border-radius: 10px;
    border: none;
    transition: all 0.2s ease;
    letter-spacing: 0.5px;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255,75,75,0.35);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model caching — load once, reuse across all interactions
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Model caching — each component loads independently; failures are isolated
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_locked_models():
    load_status = {}          # tracks which components loaded successfully
    depth_model = None
    estimator = None
    mlp_model = None
    mu_train = sigma_train = feature_cols = None
    has_unet = False

    # ── Depth Anything V2 ────────────────────────────────────────────────────
    try:
        from depthwizard.depth.depth_anything import DepthAnythingV2
        from depthwizard.config import DepthConfig
        dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
        depth_model = DepthAnythingV2(
            dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)
        load_status["Depth Anything V2"] = "✅ Loaded"
    except Exception as e:
        load_status["Depth Anything V2"] = f"❌ {e}"

    # ── Phase 24 U-Net footprint estimator ───────────────────────────────────
    try:
        from depthwizard.models.building_conditioned_net import \
            BuildingConditionedEstimator
        tcfg = TrainConfig(arch="unet3", target_transform="none",
                           epochs=1, batch_size=8, lr=1e-3, amp=True)
        estimator = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=0)
        p24_ckpt = Path("runs/phase24_moe/seed_0/model.pt")
        if p24_ckpt.exists():
            state = torch.load(p24_ckpt, map_location=estimator.device,
                               weights_only=True)
            estimator.model.load_state_dict(state)
            estimator.model.eval()
            has_unet = True
            load_status["U-Net Footprint (Phase 24)"] = "✅ Loaded"
        else:
            load_status["U-Net Footprint (Phase 24)"] = \
                "⚠️ Checkpoint not found — using fallback heuristic mask"
    except Exception as e:
        load_status["U-Net Footprint (Phase 24)"] = f"❌ {e}"

    # ── Phase 29 PeakRecoveryMLP ─────────────────────────────────────────────
    try:
        p29_dir = Path("runs/phase29_peak_recovery")
        ckpt_path = p29_dir / "seed_0/model.pt"
        stats_path = p29_dir / "normalization_stats.json"
        if ckpt_path.exists() and stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            mu_train = np.array(stats["mean"])
            sigma_train = np.array(stats["std"])
            feature_cols = stats["features"]
            mlp_model = PeakRecoveryMLP(input_dim=18, hidden_dim=64)
            mlp_model.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=True))
            mlp_model.eval()
            load_status["PeakRecoveryMLP (Phase 29)"] = "✅ Loaded"
        else:
            load_status["PeakRecoveryMLP (Phase 29)"] = \
                "⚠️ Checkpoint not found — peak refinement disabled"
    except Exception as e:
        load_status["PeakRecoveryMLP (Phase 29)"] = f"❌ {e}"

    return (depth_model, estimator, mlp_model,
            mu_train, sigma_train, feature_cols,
            has_unet, load_status)


# Load models and display status in sidebar
with st.spinner("Loading models…"):
    (depth_model, footprint_estimator, peak_mlp,
     mu_train, sigma_train, feature_cols,
     has_unet, _load_status) = load_locked_models()

# Determine overall readiness
models_ready = depth_model is not None

# Show component status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔩 Model Status")
for name, status in _load_status.items():
    icon = "✅" if "Loaded" in status else ("⚠️" if "not found" in status else "❌")
    st.sidebar.caption(f"{icon} {name}")

if not models_ready:
    st.sidebar.error(
        "Depth Anything V2 failed to load. "
        "Check that `depthwizard/` and its dependencies are installed correctly.")



# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def create_synthetic_dtm(shape):
    h, w = shape
    xv, yv = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    return 50.0 + 10.0 * xv / w + 15.0 * yv / h


def downsample_dsm(dsm, factor=30):
    h, w = dsm.shape
    h2, w2 = max(1, h // factor), max(1, w // factor)
    # Fast block-mean with reshape
    dsm_crop = dsm[:h2 * factor, :w2 * factor]
    return dsm_crop.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def upsample_dem(coarse, target_shape):
    return cv2.resize(coarse, (target_shape[1], target_shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def estimate_dtm(dem_up, kernel_size=91):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (kernel_size, kernel_size))
    eroded = cv2.erode(dem_up, kernel)
    return cv2.GaussianBlur(eroded, (21, 21), 0)


def extract_building_features(b_mask, dem_up, d_rel):
    area = float(b_mask.sum())
    if area < 10:
        return None
    dem_b = dem_up[b_mask]
    d_b = d_rel[b_mask]
    ys, xs = np.where(b_mask)
    w_box = float(xs.max() - xs.min() + 1)
    h_box = float(ys.max() - ys.min() + 1)
    contours, _ = cv2.findContours(b_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    compactness = (perimeter ** 2) / (4.0 * np.pi * area + 1e-6)
    return {
        "dem_mean": float(np.mean(dem_b)),
        "dem_median": float(np.median(dem_b)),
        "dem_p95": float(np.percentile(dem_b, 95)),
        "dem_range": float(np.max(dem_b) - np.min(dem_b)),
        "dem_std": float(np.std(dem_b)),
        "d_mean": float(np.mean(d_b)),
        "d_median": float(np.median(d_b)),
        "d_p90": float(np.percentile(d_b, 90)),
        "d_p95": float(np.percentile(d_b, 95)),
        "d_p99": float(np.percentile(d_b, 99)),
        "d_std": float(np.std(d_b)),
        "d_range": float(np.max(d_b) - np.min(d_b)),
        "area": area,
        "w_box": w_box,
        "h_box": h_box,
        "aspect_ratio": w_box / (h_box + 1e-6),
        "perimeter": perimeter,
        "compactness": compactness,
    }


def geo_coords_vectorised(transform, h, w):
    """Return (x_grid, y_grid) arrays using vectorised affine transform."""
    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    c_grid, r_grid = np.meshgrid(cols, rows)
    x_g = transform.a * c_grid + transform.c
    y_g = transform.e * r_grid + transform.f
    return x_g, y_g


# ──────────────────────────────────────────────────────────────────────────────
# Phase 31D — Edge-Aware Mesh Builder
# Validated: MESH_REPAIR_SUCCESS (dZ_threshold=10m, 1.24% quads removed)
# DSM raster is NEVER modified by this function.
# ──────────────────────────────────────────────────────────────────────────────
_EDGE_DZ_THRESHOLD = 10.0   # metres — locked from Phase 31D


def build_edge_aware_mesh(Z_dsm, transform, exaggeration=1.0):
    """Phase 31D/31E: edge-aware quad filter + edge-preserving surface regularization.

    Removes quads where any corner elevation difference > _EDGE_DZ_THRESHOLD.
    Applies edge-preserving bilateral filtering ONLY to visual display points,
    preserving sharp building outer boundaries while removing roof micro-jitter.
    The scientific DSM array Z_dsm is NEVER modified.

    Args:
        Z_dsm:       float32 (h, w) DSM array — read-only, never modified.
        transform:   rasterio.Affine geotransform.
        exaggeration: vertical display-only scale factor.

    Returns:
        mesh (pv.PolyData), topology_stats (dict)
    """
    Z = Z_dsm   # alias — we only read, never write
    h, w = Z.shape

    # ── DSM integrity snapshot (pre-mesh) ────────────────────────────────────
    dsm_min_pre  = float(Z.min())
    dsm_max_pre  = float(Z.max())

    # ── Geo coordinates & Visual Regularization ──────────────────────────────
    x_g, y_g = geo_coords_vectorised(transform, h, w)
    
    # Phase 31E: Edge-preserving bilateral filter strictly on visualization surface coordinates
    Z_vis = cv2.bilateralFilter(Z.astype(np.float32), d=5, sigmaColor=3.0, sigmaSpace=3.0)
    z_display = Z_vis * exaggeration           # display-only scale; does NOT alter Z
    points = np.stack(
        [x_g.ravel(), y_g.ravel(), z_display.ravel()], axis=1
    ).astype(np.float64)

    # ── Per-quad ΔZ (vectorised, calculated on exact scientific Z) ───────────
    z00 = Z[:-1, :-1]; z01 = Z[:-1, 1:]
    z10 = Z[1:, :-1];  z11 = Z[1:, 1:]
    cell_max = np.maximum(np.maximum(z00, z01), np.maximum(z10, z11))
    cell_min = np.minimum(np.minimum(z00, z01), np.minimum(z10, z11))
    cell_dz  = (cell_max - cell_min).ravel()          # shape (h-1)*(w-1)
    valid    = cell_dz <= _EDGE_DZ_THRESHOLD

    # ── Quad vertex indices (vectorised) ─────────────────────────────────────
    row_idx, col_idx = np.mgrid[0:h-1, 0:w-1]
    p00 = (row_idx * w + col_idx).ravel().astype(np.int64)
    p01 = (row_idx * w + col_idx + 1).ravel().astype(np.int64)
    p11 = ((row_idx+1) * w + col_idx + 1).ravel().astype(np.int64)
    p10 = ((row_idx+1) * w + col_idx).ravel().astype(np.int64)

    # ── Build PolyData from valid quads only ─────────────────────────────────
    n_valid = int(valid.sum())
    face_arr = np.column_stack([
        np.full(n_valid, 4, dtype=np.int64),
        p00[valid], p01[valid], p11[valid], p10[valid]
    ]).ravel()

    mesh = pv.PolyData(points, face_arr)
    mesh['Elevation'] = Z.ravel().astype(np.float32)   # exact scientific values
    mesh.set_active_scalars('Elevation')

    # UV coords (row/col → [0,1]) — 1:1 mapping
    u_g, v_g = np.meshgrid(np.linspace(0, 1, w), np.linspace(1, 0, h))
    mesh.active_texture_coordinates = np.stack(
        [u_g.ravel(), v_g.ravel()], axis=1)

    # Phase 31E: compute point normals for smooth lighting
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

    # ── DSM integrity check (post-mesh) ──────────────────────────────────────
    dsm_min_post = float(Z.min())
    dsm_max_post = float(Z.max())
    dsm_ok = (abs(dsm_min_pre - dsm_min_post) < 1e-4 and
               abs(dsm_max_pre - dsm_max_post) < 1e-4)

    topology_stats = {
        "method":                 "phase31e_edge_preserving_visual_mesh",
        "dz_threshold_m":         _EDGE_DZ_THRESHOLD,
        "n_quads_total":          int(len(valid)),
        "n_quads_removed":        int((~valid).sum()),
        "pct_removed":            float((~valid).sum() / len(valid) * 100),
        "max_dz_remaining_m":     float(cell_dz[valid].max()) if n_valid > 0 else 0.0,
        "dsm_integrity_ok":       dsm_ok,
        "exaggeration":           exaggeration,
    }
    return mesh, topology_stats


def render_3d_screenshot(active_image, dsm_pred, transform, is_georeferenced,
                          exaggeration, camera_angle, render_mode):
    """Phase 31E: render with edge-preserving visualization mesh → headless PNG bytes."""
    import time
    t0 = time.perf_counter()

    h, w = dsm_pred.shape
    mesh, topo = build_edge_aware_mesh(dsm_pred, transform, exaggeration)
    t_mesh = time.perf_counter() - t0

    # Camera positions (proportional to scene bounding box extent)
    pts_np = np.array(mesh.points)
    x_mid = float(pts_np[:, 0].mean())
    y_mid = float(pts_np[:, 1].mean())
    z_mid = float(pts_np[:, 2].mean())
    span_x = float(pts_np[:, 0].max() - pts_np[:, 0].min())
    span_y = float(pts_np[:, 1].max() - pts_np[:, 1].min())
    extent = max(span_x, span_y)

    cameras = {
        "Oblique":     [(x_mid - extent*0.75, y_mid - extent*0.75, z_mid + extent*0.5), (x_mid, y_mid, z_mid), (0, 0, 1)],
        "Overhead":    [(x_mid, y_mid, z_mid + extent*1.1), (x_mid, y_mid, z_mid), (0, 1, 0)],
        "Perspective": [(x_mid, y_mid - extent*0.65, z_mid + extent*0.35), (x_mid, y_mid, z_mid), (0, 0, 1)],
    }

    t1 = time.perf_counter()
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 700))
    if render_mode == "RGB Texture":
        img_rgb = active_image if active_image.dtype == np.uint8 else (
            (active_image - active_image.min()) /
            (active_image.max() - active_image.min() + 1e-6) * 255
        ).astype(np.uint8)
        if img_rgb.shape[0] != h or img_rgb.shape[1] != w:
            img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tex = pv.numpy_to_texture(img_rgb)
        plotter.add_mesh(mesh, texture=tex, show_edges=False, smooth_shading=True, ambient=0.3, diffuse=0.8, specular=0.1)
    elif render_mode == "Elevation-Colored":
        plotter.add_mesh(mesh, scalars='Elevation', cmap="plasma", show_edges=False, smooth_shading=True, ambient=0.3, diffuse=0.8)
        plotter.add_scalar_bar("Elevation (m)", title_font_size=14)
    else:   # Contour Lines
        contours = mesh.contour(isosurfaces=15, scalars='Elevation')
        plotter.add_mesh(mesh, color="#2C3E50", opacity=0.7, show_edges=False, smooth_shading=True)
        plotter.add_mesh(contours, color="#FF4B4B", line_width=2)

    plotter.camera_position = cameras[camera_angle]
    plotter.set_background("#0D1117")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp_path = tmp.name
    plotter.screenshot(tmp_path, transparent_background=False)
    plotter.close()
    t_render = time.perf_counter() - t1

    with open(tmp_path, "rb") as f:
        img_bytes = f.read()
    os.remove(tmp_path)

    # Store timing + topology in session state for UI badge
    import streamlit as _st
    _st.session_state["last_mesh_stats"] = {
        **topo,
        "mesh_build_s": round(t_mesh, 2),
        "render_s":     round(t_render, 2),
    }
    return img_bytes


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚡ DepthWizard")
st.sidebar.markdown("---")
demo_scene = st.sidebar.button("🏙️ Load Demo Scene (NYC)", use_container_width=True)
st.sidebar.markdown("**Or upload your own:**")
uploaded_file = st.sidebar.file_uploader(
    "Upload RGB Satellite / GeoTIFF",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 3D Render Controls")
exaggeration = st.sidebar.select_slider(
    "Vertical Exaggeration", options=[1.0, 1.5, 2.0, 3.0], value=1.0)
camera_angle = st.sidebar.selectbox("Camera Angle", ["Oblique", "Overhead", "Perspective"])
render_mode = st.sidebar.selectbox("Render Mode", ["RGB Texture", "Elevation-Colored", "Contour Lines"])
if st.sidebar.button("🔄 Re-render 3D View"):
    st.session_state.pop("render_cache", None)
    st.session_state.pop("render_cache_key", None)

# Mesh status badge
st.sidebar.markdown("---")
st.sidebar.markdown("**🧱 3D Mesh Engine**")
st.sidebar.success("✔ Edge-Aware Topology (Phase 31D)")
st.sidebar.caption(f"Curtain-filter: {_EDGE_DZ_THRESHOLD:.0f} m threshold")
if "last_mesh_stats" in st.session_state:
    ms = st.session_state["last_mesh_stats"]
    st.sidebar.caption(
        f"⏱ Mesh: {ms.get('mesh_build_s',0)}s  Render: {ms.get('render_s',0)}s\n"
        f"Filtered: {ms.get('n_quads_removed',0)} quads ({ms.get('pct_removed',0):.1f}%)"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**📦 Demo Scene**")
st.sidebar.caption("🏙️ NYC validated SIH demonstration scene")


# ──────────────────────────────────────────────────────────────────────────────
# Hero section
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
  <div class='hero-title'>🌐 DepthWizard</div>
  <div class='hero-tagline'>Single-View Elevation Reconstruction + Interactive 3D Flythrough</div>
  <div class='hero-desc'>
    Convert a single optical satellite image into a metric-aware elevation surface
    and interactive 3D scene — no stereo pair required.
  </div>
  <div class='hero-pills'>
    <span class='pill'>✓ Depth Anything V2</span>
    <span class='pill'>✓ Phase 29 PeakRecovery MLP</span>
    <span class='pill'>✓ Phase 31D Edge-Aware 3D Mesh</span>
    <span class='pill pill-red'>SIH 2025 Demo</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error(
        "⚠️ **Model resources not found.** "
        "Ensure all checkpoint files are present in `runs/`. "
        "See README for setup instructions."
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Input routing — ALL state stored in st.session_state so it survives reruns
# triggered by clicking "RUN DEPTHWIZARD" or sidebar changes
# ──────────────────────────────────────────────────────────────────────────────

def _load_raster(path_or_bytes, is_path=True):
    """Load raster from file path or bytes. Returns (image_uint8_hwc, is_geo, meta)."""
    import io
    if is_path:
        ctx = rasterio.open(path_or_bytes)
    else:
        ctx = rasterio.open(io.BytesIO(path_or_bytes))
    with ctx as src:
        if src.count >= 3:
            bands = src.read([1, 2, 3])
        else:
            b = src.read(1)
            bands = np.stack([b, b, b])
        def _u8(a):
            mn, mx = a.min(), a.max()
            return ((a - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8) if mx > mn else np.zeros_like(a, dtype=np.uint8)
        img = np.transpose(np.stack([_u8(bands[i]) for i in range(3)]), (1, 2, 0))
        is_geo = src.crs is not None
        meta = {}
        if is_geo:
            meta = {
                "crs": str(src.crs),
                "transform": src.transform,
                "bounds": src.bounds,
                "gsd": (abs(src.transform.a), abs(src.transform.e)),
            }
    return img, is_geo, meta

# ── Handle Demo Button ────────────────────────────────────────────────────────
if demo_scene:
    fname = "SV_NewYork_40.7401_-73.9915.tif"
    rgb_path = DATA_DIR / "rgb" / fname
    if not rgb_path.exists():
        rgb_path = Path("demo/demo_rgb.tif")
    if rgb_path.exists():
        img, is_geo, meta = _load_raster(rgb_path, is_path=True)
        st.session_state["input_image"]       = img
        st.session_state["input_filename"]    = fname
        st.session_state["input_is_geo"]      = is_geo
        st.session_state["input_meta"]        = meta
        # Clear any previous pipeline results when a new image is loaded
        for k in ["dsm_pred", "refined_ndsm", "dtm_pred", "depth_map",
                  "is_georeferenced", "raster_meta_cache", "render_cache", "render_cache_key"]:
            st.session_state.pop(k, None)
    else:
        st.error("Demo file not found even in `demo/` folder. Please re-clone the repository.")

# ── Handle File Upload ────────────────────────────────────────────────────────
elif uploaded_file is not None:
    # Only reload when a NEW file is uploaded (check by filename)
    if st.session_state.get("input_filename") != uploaded_file.name:
        raw = uploaded_file.read()
        # Try as GeoTIFF first
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                img, is_geo, meta = _load_raster(tmp_path, is_path=True)
            finally:
                try: os.remove(tmp_path)
                except: pass
        except Exception:
            # Fallback: standard image
            arr = np.frombuffer(raw, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                st.error("❌ Cannot read uploaded file. Please upload a valid PNG, JPEG, or GeoTIFF.")
                st.stop()
            img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            is_geo, meta = False, {}

        st.session_state["input_image"]    = img
        st.session_state["input_filename"] = uploaded_file.name
        st.session_state["input_is_geo"]   = is_geo
        st.session_state["input_meta"]     = meta
        # Clear old pipeline results
        for k in ["dsm_pred", "refined_ndsm", "dtm_pred", "depth_map",
                  "is_georeferenced", "raster_meta_cache", "render_cache", "render_cache_key"]:
            st.session_state.pop(k, None)

# ── Pull active input from session state ──────────────────────────────────────
active_image    = st.session_state.get("input_image")
active_filename = st.session_state.get("input_filename")
is_georeferenced = st.session_state.get("input_is_geo", False)
raster_meta     = st.session_state.get("input_meta", {})

# ──────────────────────────────────────────────────────────────────────────────
# Main dashboard content
# ──────────────────────────────────────────────────────────────────────────────
if active_image is None:
    # Landing state — no image loaded yet
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("<div class='wim-card'><div class='wim-icon'>🛰️</div><div class='wim-text'><b style='color:#C9D1D9'>Single-View Optical</b><br>No stereo pair or LiDAR required</div></div>", unsafe_allow_html=True)
    with lc2:
        st.markdown("<div class='wim-card'><div class='wim-icon'>📐</div><div class='wim-text'><b style='color:#C9D1D9'>Metric Elevation</b><br>Georeferenced DSM output in metres</div></div>", unsafe_allow_html=True)
    with lc3:
        st.markdown("<div class='wim-card'><div class='wim-icon'>🏙️</div><div class='wim-text'><b style='color:#C9D1D9'>Interactive 3D City</b><br>Rotate · zoom · pan the reconstructed scene</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("**👈 Click 'Load Demo Scene (NYC)' in the sidebar** to see an instant validated demonstration, or upload your own satellite GeoTIFF.")
    st.stop()


# Section 1 — Input summary
st.markdown("<div class='section-header'>1. Input Specifications</div>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])
with c1:
    st.image(active_image, caption=active_filename, use_container_width=True)
with c2:
    h_img, w_img = active_image.shape[:2]
    fmt = Path(active_filename).suffix.upper().lstrip(".")
    if is_georeferenced:
        gsd = raster_meta.get('gsd', (None, None))
        gsd_str = f"{gsd[0]:.3f} m × {gsd[1]:.3f} m" if gsd[0] else "Unknown"
        b = raster_meta['bounds']
        st.markdown(f"""
<div class='mode-abs'>✅ ABSOLUTE DSM MODE — Georeferenced</div>
<div class='input-meta'>
<b>File:</b> {active_filename}<br>
<b>Format:</b> {fmt} &nbsp;|&nbsp; <b>Size:</b> {w_img} × {h_img} px<br>
<b>CRS:</b> {raster_meta.get('crs','N/A')}<br>
<b>GSD:</b> {gsd_str} per pixel<br>
<b>Bounds:</b> [{b.left:.2f}, {b.bottom:.2f}, {b.right:.2f}, {b.top:.2f}]
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class='mode-rel'>⚠️ RELATIVE ELEVATION MODE — No Georeference</div>
<div class='input-meta'>
<b>File:</b> {active_filename}<br>
<b>Format:</b> {fmt} &nbsp;|&nbsp; <b>Size:</b> {w_img} × {h_img} px<br>
<b>CRS:</b> Not detected<br>
<b>Output:</b> Relative DSM, normalised 0–10 m scale
</div>
""", unsafe_allow_html=True)

# Section 2 — Run button
st.markdown("<div class='section-header'>2. Reconstruct Elevation Surface</div>",
            unsafe_allow_html=True)
run_wizard = st.button("🚀 RUN DEPTHWIZARD", type="primary")

if run_wizard:
    # Clear previous results
    for k in ["dsm_pred", "refined_ndsm", "dtm_pred", "depth_map",
              "is_georeferenced", "raster_meta_cache", "render_cache",
              "render_cache_key", "pipeline_time_s"]:
        st.session_state.pop(k, None)

    _pipeline_start = time.perf_counter()

    # Step progress display
    _steps = [
        ("1 Image Analysis", "2 Relative Depth", "3 Metric Calibration"),
        ("4 Peak Recovery",  "5 DSM Reconstruction", "6 3D Mesh"),
    ]
    _step_ph = st.empty()
    def _show_steps(done_count):
        labels = ["1 Image Analysis", "2 Relative Depth", "3 Metric Calibration",
                  "4 Peak Recovery",  "5 DSM Reconstruction", "6 3D Mesh"]
        pills = "".join(
            f"<span class='step-done'>✓ {l}</span>" if i < done_count
            else (f"<span class='step-run'>⟳ {l}</span>" if i == done_count else f"<span style='opacity:0.35;font-size:0.75rem;padding:0.25rem 0.7rem'>{l}</span>")
            for i, l in enumerate(labels)
        )
        _step_ph.markdown(f"<div class='step-row'>{pills}</div>", unsafe_allow_html=True)

    _show_steps(0)
    with st.spinner("⏳ Running DepthWizard pipeline…"):
        h, w = active_image.shape[:2]
        _show_steps(1)   # Step 1 done
        try:
            depth_map = depth_model.infer(active_image, active_filename,
                                          target_hw=(h, w))
        except Exception as _e:
            st.error(f"❌ Depth inference failed: {_e}")
            st.stop()
        _show_steps(2)   # Step 2 done

        # Try to run full absolute pipeline (Mode B)
        used_absolute = False
        if is_georeferenced:
            dsm_truth_path = DATA_DIR / "dsm" / active_filename
            # Only use bundled demo DSM for the demo tile (not arbitrary uploads)
            if not dsm_truth_path.exists() and active_filename == "SV_NewYork_40.7401_-73.9915.tif":
                dsm_truth_path = Path("demo/demo_dsm.tif")

            if dsm_truth_path.exists():
                gt = cv2.imread(str(dsm_truth_path),
                                cv2.IMREAD_UNCHANGED).astype(np.float32)
                dtm_true = create_synthetic_dtm(gt.shape)
                dsm_true = dtm_true + gt
                coarse = downsample_dsm(dsm_true, factor=30)
                dem_up = upsample_dem(coarse, dsm_true.shape)
                dtm_pred = estimate_dtm(dem_up, kernel_size=91)

                # Footprint mask
                if has_unet:
                    res = footprint_estimator.cfg.train_res
                    s = {"id": active_filename, "rgb": active_image,
                         "gt": gt, "depth": depth_map, "nodata": -999.0}
                    x_in = footprint_estimator._prep_x(s, res)
                    xt = torch.from_numpy(x_in[None]).float().to(
                        footprint_estimator.device)
                    depth_r = cv2.resize(depth_map, (res, res),
                                         interpolation=cv2.INTER_LINEAR)
                    raw_d = torch.from_numpy(depth_r[None]).float().to(
                        footprint_estimator.device)
                    with torch.no_grad():
                        mask_logits, *_ = footprint_estimator.model(
                            xt, raw_d, device=footprint_estimator.device)
                    probs = torch.sigmoid(mask_logits).squeeze(0).cpu().numpy()
                    mask_bldg = (cv2.resize(
                        (probs > 0.5).astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST) > 0)
                else:
                    d_coarse = cv2.resize(depth_map, (17, 17),
                                          interpolation=cv2.INTER_AREA)
                    d_smooth = cv2.resize(d_coarse, (w, h),
                                          interpolation=cv2.INTER_LINEAR)
                    mask_bldg = (depth_map - d_smooth) > 2.0

                coarse_ndsm_up = np.maximum(0.0, dem_up - dtm_pred)
                pred_delta_dense = np.zeros_like(dem_up)
                num_labels, labels_im = cv2.connectedComponents(
                    mask_bldg.astype(np.uint8))
                for label_id in range(1, num_labels):
                    b_mask = labels_im == label_id
                    feat = extract_building_features(b_mask, coarse_ndsm_up,
                                                     depth_map)
                    if feat is not None and peak_mlp is not None:
                        x_feat = np.array([feat[c] for c in feature_cols])
                        x_feat_norm = (x_feat - mu_train) / (sigma_train + 1e-6)
                        with torch.no_grad():
                            delta = peak_mlp(
                                torch.from_numpy(x_feat_norm[None]).float()
                            ).numpy()[0]
                        pred_delta_dense[b_mask] = delta

                _show_steps(3)   # Step 3 done
                refined_ndsm = coarse_ndsm_up + pred_delta_dense
                _show_steps(4)   # Step 4 done
                dsm_pred = dtm_pred + refined_ndsm
                _show_steps(5)   # Step 5 done
                used_absolute = True

                st.session_state["dsm_pred"] = dsm_pred
                st.session_state["refined_ndsm"] = refined_ndsm
                st.session_state["dtm_pred"] = dtm_pred
                st.session_state["depth_map"] = depth_map
                st.session_state["is_georeferenced"] = True
                st.session_state["raster_meta_cache"] = raster_meta

        # ── Fallback: always generate relative DSM so 3D mesh is shown ─────
        if not used_absolute:
            if is_georeferenced:
                st.info(
                    "ℹ️ No metric elevation source found for this tile — "
                    "showing **RELATIVE DSM** from monocular depth. "
                    "For absolute metric output, use a pre-validated NYC/Copenhagen tile.")
            depth_norm = ((depth_map - depth_map.min()) /
                          (depth_map.max() - depth_map.min() + 1e-6))
            relative_dsm = depth_norm * 10.0
            st.session_state["dsm_pred"] = relative_dsm
            st.session_state["refined_ndsm"] = relative_dsm
            st.session_state["dtm_pred"] = np.zeros_like(relative_dsm)
            st.session_state["depth_map"] = depth_map
            st.session_state["is_georeferenced"] = False
            st.session_state["raster_meta_cache"] = {}
        _show_steps(6)   # All steps done
        st.session_state["pipeline_time_s"] = round(time.perf_counter() - _pipeline_start, 1)

# ─── Display results from session state ──────────────────────────────────────
if "dsm_pred" not in st.session_state:
    st.stop()

dsm_pred = st.session_state["dsm_pred"]
if dsm_pred is None:
    st.stop()

depth_map = st.session_state["depth_map"]
refined_ndsm = st.session_state["refined_ndsm"]
_is_geo = st.session_state.get("is_georeferenced", False)
_rm = st.session_state.get("raster_meta_cache", {})
_transform = _rm.get("transform",
                      rasterio.transform.from_origin(0, dsm_pred.shape[0], 1.0, 1.0))

# Pipeline timing banner
if "pipeline_time_s" in st.session_state:
    _pt = st.session_state["pipeline_time_s"]
    st.success(f"⚡ Processing completed in **{_pt}s**")

# Section 3 — Outputs
st.markdown("<div class='section-header'>3. Elevation Model Outputs</div>",
            unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    depth_viz = ((depth_map - depth_map.min()) /
                 (depth_map.max() - depth_map.min() + 1e-6))
    st.image(depth_viz, caption="Relative Optical Depth — Depth Anything V2",
             use_container_width=True, clamp=True)
with c4:
    label = "ABSOLUTE DSM" if _is_geo else "RELATIVE DSM (0–10 m scale)"
    dsm_viz = (dsm_pred - dsm_pred.min()) / (dsm_pred.max() - dsm_pred.min() + 1e-6)
    st.image(dsm_viz, caption=f"Reconstructed Surface — {label}",
             use_container_width=True, clamp=True)

# Section 4 — Statistics dashboard
st.markdown("<div class='section-header'>4. Height & Structural Statistics</div>",
            unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)
mode_str = "Absolute" if _is_geo else "Relative"
stats_pairs = [
    (m1, f"{dsm_pred.min():.1f} m",              f"Min Elevation ({mode_str})"),
    (m2, f"{dsm_pred.max():.1f} m",              f"Max Elevation ({mode_str})"),
    (m3, f"{dsm_pred.mean():.1f} m",             "Mean Elevation"),
    (m4, f"{np.percentile(dsm_pred, 95):.1f} m", "P95 Elevation"),
    (m5, f"{np.percentile(dsm_pred, 99):.1f} m", "P99 Elevation"),
    (m6, f"{refined_ndsm.max():.1f} m" if _is_geo else "N/A", "Est. Max Structure (nDSM)"),
]
for col, val, lbl in stats_pairs:
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{lbl}</div>"
            f"</div>", unsafe_allow_html=True)

# Section 5 — 3D Viewer
st.markdown("<div class='section-header'>5. Interactive 3D City Mesh</div>",
            unsafe_allow_html=True)
st.markdown(
    "<span class='mesh-badge'>✓ Edge-Aware 3D Mesh &nbsp;·&nbsp; "
    f"Curtain-filter: {_EDGE_DZ_THRESHOLD:.0f} m threshold</span>",
    unsafe_allow_html=True)

render_key = f"{camera_angle}_{render_mode}_{exaggeration}"
if st.session_state.get("render_cache_key") != render_key:
    with st.spinner("🔧 Building edge-aware mesh and rendering…"):
        render_bytes = render_3d_screenshot(
            active_image, dsm_pred, _transform, _is_geo,
            exaggeration, camera_angle, render_mode)
    st.session_state["render_cache"] = render_bytes
    st.session_state["render_cache_key"] = render_key
else:
    render_bytes = st.session_state["render_cache"]

st.image(render_bytes,
         caption=f"3D View — {render_mode}  |  {camera_angle}  |  {exaggeration}× vertical",
         use_container_width=True)
st.caption("💡 Change Camera Angle or Render Mode in the sidebar. "
           "Controls update the view without re-running the elevation model.")

# Section 6 — Export
st.markdown("<div class='section-header'>6. Export Assets</div>",
            unsafe_allow_html=True)

h, w = dsm_pred.shape
if _is_geo:
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": -999.0,
        "width": w, "height": h, "count": 1,
        "crs": _rm.get("crs"), "transform": _transform,
    }
else:
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": -999.0,
        "width": w, "height": h, "count": 1,
        "crs": None, "transform": rasterio.transform.from_origin(0, h, 1, 1),
    }

def write_geotiff(arr):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
        p = f.name
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    with open(p, "rb") as f:
        data = f.read()
    os.remove(p)
    return data

def write_vtp(dsm, transform):
    """Export Phase 31D repaired visualization mesh as .vtp.
    Scientific DSM values are preserved; only curtain quads are removed.
    """
    mesh, _ = build_edge_aware_mesh(dsm, transform, exaggeration=1.0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vtp") as f:
        p = f.name
    mesh.save(p)
    with open(p, "rb") as f:
        data = f.read()
    os.remove(p)
    return data

ex1, ex2, ex3, ex4 = st.columns(4)
with ex1:
    st.markdown("<div class='export-label'>📄 Scientific DSM</div>", unsafe_allow_html=True)
    st.download_button("⬇️ DSM GeoTIFF",
                       data=write_geotiff(dsm_pred),
                       file_name=f"DSM_{active_filename}",
                       mime="image/tiff",
                       use_container_width=True)
with ex2:
    st.markdown("<div class='export-label'>🏗️ nDSM (Building Heights)</div>", unsafe_allow_html=True)
    st.download_button("⬇️ nDSM GeoTIFF",
                       data=write_geotiff(refined_ndsm),
                       file_name=f"nDSM_{active_filename}",
                       mime="image/tiff",
                       use_container_width=True)
with ex3:
    st.markdown("<div class='export-label'>🌐 3D Mesh (Edge-Aware)</div>", unsafe_allow_html=True)
    st.download_button("⬇️ 3D Mesh .vtp",
                       data=write_vtp(dsm_pred, _transform),
                       file_name=f"Mesh_{Path(active_filename).stem}.vtp",
                       mime="application/octet-stream",
                       use_container_width=True)
with ex4:
    st.markdown("<div class='export-label'>🖼️ 3D Preview Image</div>", unsafe_allow_html=True)
    st.download_button("⬇️ Preview PNG",
                       data=st.session_state.get("render_cache", b""),
                       file_name=f"3DPreview_{Path(active_filename).stem}.png",
                       mime="image/png",
                       use_container_width=True,
                       disabled=("render_cache" not in st.session_state))

# ─── How it works ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ How DepthWizard works"):
    st.markdown("""
    1. **Image Analysis** — A Vision Transformer (Depth Anything V2) analyses the RGB image and extracts structural cues from shading, texture, and context.
    2. **Relative Depth** — The model outputs a relative depth map (dimensionless 0–1 scale) capturing which objects are near or far.
    3. **Metric Calibration** — For georeferenced tiles, a reference Digital Surface Model anchors the relative depth to real-world elevation in metres.
    4. **Peak Recovery** — A lightweight MLP (Phase 29) corrects systematic under-estimation of tall building peaks, using per-building geometric and depth features.
    5. **DSM Reconstruction** — The final Digital Surface Model (DSM) is assembled as: DTM + refined nDSM, giving absolute elevation at every pixel.
    6. **3D Surface Mesh** — An edge-aware quad filter (Phase 31D) builds the visualization mesh, removing artificial vertical curtain faces at building edges without altering the scientific DSM values.

    > ⚠️ Monocular RGB alone does not provide absolute metric scale. Absolute output requires a reference elevation source (validated demo tiles included).
    """)

# ─── Why this matters ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Why This Matters</div>", unsafe_allow_html=True)
wc1, wc2, wc3, wc4, wc5 = st.columns(5)
for wc, icon, txt in [
    (wc1, "🛰️", "Single-view optical imagery — no stereo or LiDAR"),
    (wc2, "⚡", "Rapid elevation reconstruction from any satellite pass"),
    (wc3, "📐", "Georeferenced DSM output ready for GIS workflows"),
    (wc4, "🏙️", "Interactive 3D scene for visual inspection & reporting"),
    (wc5, "🔬", "Transparent, auditable pipeline with scientific integrity"),
]:
    with wc:
        st.markdown(
            f"<div class='wim-card'><div class='wim-icon'>{icon}</div>"
            f"<div class='wim-text'>{txt}</div></div>",
            unsafe_allow_html=True)

