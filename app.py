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
    page_title="DepthWizard — Absolute 3D Elevation Model MVP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500&display=swap');
body { background: #0D1117; }
.main-title {
    font-family: 'Outfit', sans-serif;
    color: #FF4B4B;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.subtitle {
    font-family: 'Inter', sans-serif;
    color: #8B949E;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}
.section-header {
    font-family: 'Outfit', sans-serif;
    color: #E6EDF3;
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1.8rem;
    margin-bottom: 0.8rem;
    border-bottom: 2px solid #21262D;
    padding-bottom: 0.3rem;
}
.metric-card {
    background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #3FB950;
}
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #8B949E;
    margin-top: 0.2rem;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF4B4B, #C62828);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 2.5rem;
    border-radius: 8px;
    border: none;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.85;
    transform: translateY(-1px);
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
    st.sidebar.markdown(f"**{name}**  \n{status}")

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


def render_3d_screenshot(active_image, dsm_pred, transform, is_georeferenced,
                          exaggeration, camera_angle, render_mode):
    """Build PyVista mesh and render headless screenshot → returns bytes."""
    h, w = dsm_pred.shape
    x_g, y_g = geo_coords_vectorised(transform, h, w)
    z_ex = dsm_pred * exaggeration

    points = np.stack([x_g.ravel(), y_g.ravel(), z_ex.ravel()], axis=1)
    grid = pv.StructuredGrid()
    grid.points = points.astype(np.float64)
    grid.dimensions = (w, h, 1)
    mesh = grid.extract_surface(algorithm='dataset_surface')

    # UV coords
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, w), np.linspace(1, 0, h))
    mesh.active_texture_coordinates = np.stack(
        [u_grid.ravel(), v_grid.ravel()], axis=1)
    mesh['Elevation'] = dsm_pred.ravel().astype(np.float32)
    mesh.set_active_scalars('Elevation')

    x_mid, y_mid, z_mid = float(x_g.mean()), float(y_g.mean()), float(z_ex.mean())
    cameras = {
        "Overhead":    [(x_mid, y_mid, z_mid + 400), (x_mid, y_mid, z_mid), (0, 1, 0)],
        "Oblique":     [(x_mid - 250, y_mid - 250, z_mid + 200), (x_mid, y_mid, z_mid), (0, 0, 1)],
        "Perspective": [(x_mid, y_mid - 300, z_mid + 150), (x_mid, y_mid, z_mid), (0, 0, 1)],
    }

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 700))
    if render_mode == "RGB Texture":
        img_rgb = active_image if active_image.dtype == np.uint8 else (
            (active_image - active_image.min()) /
            (active_image.max() - active_image.min() + 1e-6) * 255
        ).astype(np.uint8)
        # Always resize texture to exactly match DSM grid dimensions
        # (prevents torn/jagged render when image ≠ DSM resolution)
        if img_rgb.shape[0] != h or img_rgb.shape[1] != w:
            img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tex = pv.numpy_to_texture(img_rgb)
        plotter.add_mesh(mesh, texture=tex, show_edges=False)
    elif render_mode == "Elevation-Colored":
        plotter.add_mesh(mesh, scalars='Elevation', cmap="plasma", show_edges=False)
        plotter.add_scalar_bar("Elevation (m)", title_font_size=14)
    else:
        contours = mesh.contour(isosurfaces=15, scalars='Elevation')
        plotter.add_mesh(mesh, color="#2C3E50", opacity=0.7, show_edges=False)
        plotter.add_mesh(contours, color="#FF4B4B", line_width=2)

    plotter.camera_position = cameras[camera_angle]
    plotter.set_background("#0D1117")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp_path = tmp.name
    plotter.screenshot(tmp_path, transparent_background=False)
    plotter.close()

    with open(tmp_path, "rb") as f:
        img_bytes = f.read()
    os.remove(tmp_path)
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
exaggeration = st.sidebar.slider("Vertical Exaggeration", 1.0, 3.0, 1.0, 0.5)
camera_angle = st.sidebar.selectbox("Camera Angle", ["Oblique", "Overhead", "Perspective"])
render_mode = st.sidebar.selectbox("Render Mode", ["RGB Texture", "Elevation-Colored", "Contour Lines"])
if st.sidebar.button("🔄 Re-render 3D View"):
    st.session_state.pop("render_cache", None)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🌐 DepthWizard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Absolute 3D Elevation Reconstruction — SIH Demonstration MVP</div>",
    unsafe_allow_html=True)

if not models_ready:
    st.warning("⚠️ Locked model resources unavailable. Ensure all checkpoint files are present.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Input routing
# ──────────────────────────────────────────────────────────────────────────────
active_image = None
active_filename = None
is_georeferenced = False
raster_meta = {}
file_bytes_cache = None  # Keep raw bytes in case rasterio fails

if demo_scene:
    active_filename = "SV_NewYork_40.7401_-73.9915.tif"

    # Look in full data dir first, then fall back to the bundled demo/ folder
    rgb_demo_path = DATA_DIR / "rgb" / active_filename
    if not rgb_demo_path.exists():
        rgb_demo_path = Path("demo/demo_rgb.tif")

    if rgb_demo_path.exists():
        with rasterio.open(rgb_demo_path) as src:
            bands = src.read()
            bands = bands[:3] if src.count >= 3 else np.stack([bands[0]] * 3)
            bands_u8 = np.clip(bands, 0, 255).astype(np.uint8)
            active_image = np.transpose(bands_u8, (1, 2, 0))
            is_georeferenced = src.crs is not None
            raster_meta = {
                "crs": str(src.crs),
                "transform": src.transform,
                "bounds": src.bounds,
                "gsd": (abs(src.transform.a), abs(src.transform.e)),
            }
    else:
        st.error("Demo file not found even in `demo/` folder. Please re-clone the repository.")

elif uploaded_file is not None:
    active_filename = uploaded_file.name
    file_bytes_cache = uploaded_file.read()  # Read once, store in memory

    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=Path(active_filename).suffix) as tmp:
        tmp.write(file_bytes_cache)
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path) as src:
            if src.count >= 3:
                bands = src.read([1, 2, 3])
            else:
                b = src.read(1)
                bands = np.stack([b, b, b])
            # Normalise to uint8 for display
            def to_u8(arr):
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
                return np.zeros_like(arr, dtype=np.uint8)
            active_image = np.transpose(
                np.stack([to_u8(bands[i]) for i in range(3)]), (1, 2, 0))
            is_georeferenced = src.crs is not None
            if is_georeferenced:
                raster_meta = {
                    "crs": str(src.crs),
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "gsd": (abs(src.transform.a), abs(src.transform.e)),
                }
    except Exception:
        # Fallback: treat as standard image
        arr = np.frombuffer(file_bytes_cache, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            st.error("❌ Cannot read uploaded file. Please upload a valid PNG, JPEG, or GeoTIFF.")
        else:
            active_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            is_georeferenced = False
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main dashboard content
# ──────────────────────────────────────────────────────────────────────────────
if active_image is None:
    st.info("👈 Upload a satellite image in the sidebar, or click **Load Demo Scene (NYC)** to begin.")
    st.stop()

# Section 1 — Input preview
st.markdown("<div class='section-header'>1. Input Specifications</div>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])
with c1:
    st.image(active_image, caption=active_filename, use_container_width=True)
with c2:
    st.markdown(f"**Filename:** `{active_filename}`")
    st.markdown(f"**Dimensions:** `{active_image.shape[1]} × {active_image.shape[0]} px`")
    if is_georeferenced:
        st.success("✅ GEOREFERENCED — Absolute Metric Mode")
        st.markdown(f"**CRS:** `{raster_meta['crs']}`")
        st.markdown(f"**GSD:** `{raster_meta['gsd'][0]:.3f} m × {raster_meta['gsd'][1]:.3f} m per pixel`")
        b = raster_meta['bounds']
        st.markdown(
            f"**Bounds:** Left `{b.left:.1f}`, Bottom `{b.bottom:.1f}`, "
            f"Right `{b.right:.1f}`, Top `{b.top:.1f}`")
    else:
        st.warning("⚠️ NON-GEOREFERENCED — Relative Scale Only")
        st.info("No CRS detected. Output will be labelled RELATIVE DSM (normalised 0–10 m).")

# Section 2 — Run button
st.markdown("<div class='section-header'>2. Reconstruct Elevation Surface</div>",
            unsafe_allow_html=True)
run_wizard = st.button("🚀 RUN DEPTHWIZARD", type="primary")

if run_wizard:
    # Clear previous results
    for k in ["dsm_pred", "refined_ndsm", "dtm_pred", "depth_map",
              "is_georeferenced", "raster_meta_cache", "render_cache"]:
        st.session_state.pop(k, None)

    with st.spinner("⏳ Running DepthWizard pipeline…"):
        h, w = active_image.shape[:2]
        depth_map = depth_model.infer(active_image, active_filename,
                                      target_hw=(h, w))

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

                refined_ndsm = coarse_ndsm_up + pred_delta_dense
                dsm_pred = dtm_pred + refined_ndsm
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

# Section 3 — Outputs
st.markdown("<div class='section-header'>3. Elevation Model Outputs</div>",
            unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    depth_viz = ((depth_map - depth_map.min()) /
                 (depth_map.max() - depth_map.min() + 1e-6))
    st.image(depth_viz, caption="Relative Optical Depth (Depth Anything V2)",
             use_container_width=True, clamp=True)
with c4:
    label = "ABSOLUTE DSM" if _is_geo else "RELATIVE DSM"
    dsm_viz = (dsm_pred - dsm_pred.min()) / (dsm_pred.max() - dsm_pred.min() + 1e-6)
    st.image(dsm_viz, caption=f"Reconstructed Surface — {label}",
             use_container_width=True, clamp=True)

# Section 4 — Statistics
st.markdown("<div class='section-header'>4. Height & Structural Statistics</div>",
            unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)
stats_pairs = [
    (m1, f"{dsm_pred.min():.1f} m", "Min Elevation"),
    (m2, f"{dsm_pred.max():.1f} m", "Max Elevation"),
    (m3, f"{dsm_pred.mean():.1f} m", "Mean Elevation"),
    (m4, f"{np.percentile(dsm_pred, 95):.1f} m", "P95 Elevation"),
    (m5, f"{np.percentile(dsm_pred, 99):.1f} m", "P99 Elevation"),
    (m6, f"{refined_ndsm.max():.1f} m" if _is_geo else "N/A", "Est. Max Structure"),
]
for col, val, lbl in stats_pairs:
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{lbl}</div>"
            f"</div>", unsafe_allow_html=True)

# Section 5 — 3D Viewer
st.markdown("<div class='section-header'>5. 3D Terrain Mesh Viewer</div>",
            unsafe_allow_html=True)

render_key = f"{camera_angle}_{render_mode}_{exaggeration}"
if st.session_state.get("render_cache_key") != render_key:
    with st.spinner("🔧 Rendering 3D mesh…"):
        render_bytes = render_3d_screenshot(
            active_image, dsm_pred, _transform, _is_geo,
            exaggeration, camera_angle, render_mode)
    st.session_state["render_cache"] = render_bytes
    st.session_state["render_cache_key"] = render_key
else:
    render_bytes = st.session_state["render_cache"]

st.image(render_bytes,
         caption=f"3D View — {render_mode} | {camera_angle} | {exaggeration}×",
         use_container_width=True)

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
    h2, w2 = dsm.shape
    x_g, y_g = geo_coords_vectorised(transform, h2, w2)
    pts = np.stack([x_g.ravel(), y_g.ravel(), dsm.ravel()], axis=1)
    grid = pv.StructuredGrid()
    grid.points = pts.astype(np.float64)
    grid.dimensions = (w2, h2, 1)
    mesh = grid.extract_surface(algorithm='dataset_surface')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vtp") as f:
        p = f.name
    mesh.save(p)
    with open(p, "rb") as f:
        data = f.read()
    os.remove(p)
    return data

ex1, ex2, ex3 = st.columns(3)
with ex1:
    st.download_button("⬇️ Download DSM GeoTIFF",
                       data=write_geotiff(dsm_pred),
                       file_name=f"DSM_{active_filename}",
                       mime="image/tiff")
with ex2:
    st.download_button("⬇️ Download nDSM GeoTIFF",
                       data=write_geotiff(refined_ndsm),
                       file_name=f"nDSM_{active_filename}",
                       mime="image/tiff")
with ex3:
    st.download_button("⬇️ Download 3D Mesh (.vtp)",
                       data=write_vtp(dsm_pred, _transform),
                       file_name=f"Mesh_{Path(active_filename).stem}.vtp",
                       mime="application/octet-stream")
