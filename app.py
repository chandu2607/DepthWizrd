import os
import sys
import json
import time
import tempfile
import numpy as np
import cv2
import streamlit as st
import rasterio
import pyvista as pv
from pathlib import Path

# Setup system path to import workspace modules
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from depthwizard.config import TrainConfig
from scripts.run_phase29_peak_recovery import PeakRecoveryMLP

pv.OFF_SCREEN = True
DATA_DIR = Path("data/dfc2023_multicity")

# Page configuration
st.set_page_config(
    page_title="DepthWizard — Absolute 3D Elevation Model MVP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme premium styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #FF4B4B;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #A0AEC0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-family: 'Outfit', sans-serif;
        color: #E2E8F0;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3F444E;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38A169;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #A0AEC0;
    }
</style>
""", unsafe_allow_html=True)

# 1. Caching locked model resources
@st.cache_resource
def load_locked_models():
    # Cache Depth Anything V2
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)
    
    # Cache U-Net Footprint model
    from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
    tcfg = TrainConfig(arch="unet3", target_transform="none", epochs=1, batch_size=8, lr=1e-3, amp=True)
    estimator = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=0)
    p24_ckpt = Path("runs/phase24_moe/seed_0/model.pt")
    has_unet = False
    if p24_ckpt.exists():
        try:
            state = torch.load(p24_ckpt, map_location=estimator.device)
            estimator.model.load_state_dict(state)
            estimator.model.eval()
            has_unet = True
        except Exception as e:
            pass
            
    # Cache PeakRecoveryMLP model
    p29_dir = Path("runs/phase29_peak_recovery")
    ckpt_path = p29_dir / "seed_0/model.pt"
    stats_path = p29_dir / "normalization_stats.json"
    
    mlp_model = None
    mu_train, sigma_train, feature_cols = None, None, None
    if ckpt_path.exists() and stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        mu_train = np.array(stats["mean"])
        sigma_train = np.array(stats["std"])
        feature_cols = stats["features"]
        
        import torch
        mlp_model = PeakRecoveryMLP(input_dim=18, hidden_dim=64)
        mlp_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        mlp_model.eval()
        
    return depth_model, estimator, mlp_model, mu_train, sigma_train, feature_cols, has_unet

try:
    import torch
    depth_model, footprint_estimator, peak_mlp, mu_train, sigma_train, feature_cols, has_unet = load_locked_models()
    models_ready = True
except Exception as e:
    st.error(f"Failed to load locked models: {e}")
    models_ready = False

# Sidebar configuration
st.sidebar.markdown("### DepthWizard Config")
demo_scene = st.sidebar.button("Load Demo Scene (NYC)")

uploaded_file = st.sidebar.file_uploader(
    "Upload RGB Satellite Image / GeoTIFF",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

# Render Controls sidebar
st.sidebar.markdown("### 3D Render controls")
exaggeration = st.sidebar.slider("Vertical Exaggeration", 1.0, 3.0, 1.0, 0.5)
camera_angle = st.sidebar.selectbox("Camera View Angle", ["Oblique", "Overhead", "Perspective"])
render_mode = st.sidebar.selectbox("Render Mode", ["RGB Texture", "Elevation-Colored", "Contour Lines"])

# Dashboard Title
st.markdown("<div class='main-title'>DepthWizard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Absolute Elevation Reconstruction & 3D Interactive Viewer Prototype</div>", unsafe_allow_html=True)

# Helper function to create synthetic terrain
def create_synthetic_dtm(shape):
    h, w = shape
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    return 50.0 + 10.0 * xv / w + 15.0 * yv / h

def downsample_dsm(dsm, factor=30):
    h, w = dsm.shape
    th, tw = max(1, h // factor), max(1, w // factor)
    coarse = np.zeros((th, tw), dtype=np.float32)
    for r in range(th):
        for c in range(tw):
            r_start = r * factor
            r_end = min((r + 1) * factor, h)
            c_start = c * factor
            c_end = min((c + 1) * factor, w)
            coarse[r, c] = np.mean(dsm[r_start:r_end, c_start:c_end])
    return coarse

def upsample_dem(coarse, target_shape):
    return cv2.resize(coarse, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def estimate_dtm(dem_up, kernel_size=91):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded = cv2.erode(dem_up, kernel)
    dtm_pred = cv2.GaussianBlur(eroded, (21, 21), 0)
    return dtm_pred

def extract_building_features(s, b_mask, dem_up, d_rel):
    area = float(b_mask.sum())
    if area < 10: return None
    dem_b = dem_up[b_mask]
    dem_mean = float(np.mean(dem_b))
    dem_median = float(np.median(dem_b))
    dem_p95 = float(np.percentile(dem_b, 95))
    dem_range = float(np.max(dem_b) - np.min(dem_b))
    dem_std = float(np.std(dem_b))
    d_b = d_rel[b_mask]
    d_mean = float(np.mean(d_b))
    d_median = float(np.median(d_b))
    d_p90 = float(np.percentile(d_b, 90))
    d_p95 = float(np.percentile(d_b, 95))
    d_p99 = float(np.percentile(d_b, 99))
    d_std = float(np.std(d_b))
    d_range = float(np.max(d_b) - np.min(d_b))
    ys, xs = np.where(b_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    w_box = float(x_max - x_min + 1)
    h_box = float(y_max - y_min + 1)
    aspect_ratio = w_box / (h_box + 1e-6)
    contours, _ = cv2.findContours(b_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    compactness = (perimeter ** 2) / (4.0 * np.pi * area + 1e-6)
    return {
        "dem_mean": dem_mean, "dem_median": dem_median, "dem_p95": dem_p95, "dem_range": dem_range, "dem_std": dem_std,
        "d_mean": d_mean, "d_median": d_median, "d_p90": d_p90, "d_p95": d_p95, "d_p99": d_p99, "d_std": d_std, "d_range": d_range,
        "area": area, "w_box": w_box, "h_box": h_box, "aspect_ratio": aspect_ratio, "perimeter": perimeter, "compactness": compactness
    }

# Main container
if not models_ready:
    st.warning("Locked model resources not available. Check that runs/ checkpoints exist.")
else:
    # State flags
    is_demo = False
    active_image = None
    active_filename = None
    is_georeferenced = False
    raster_meta = {}
    
    # Check for Demo button trigger
    if demo_scene:
        is_demo = True
        active_filename = "SV_NewYork_40.7401_-73.9915.tif"
        rgb_demo_path = Path("data/dfc2023_multicity/rgb") / active_filename
        
        if rgb_demo_path.exists():
            with rasterio.open(rgb_demo_path) as src:
                active_image = src.read()
                active_image = np.transpose(active_image, (1, 2, 0)) # H, W, C
                is_georeferenced = True
                raster_meta = {
                    "crs": str(src.crs),
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "gsd": (abs(src.transform.a), abs(src.transform.e)),
                    "shape": (src.height, src.width)
                }
        else:
            st.error("Demo RGB file missing in workspace.")
            
    # Check for Upload trigger
    elif uploaded_file is not None:
        active_filename = uploaded_file.name
        # Write file temporarily to read with rasterio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(active_filename).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
            
        try:
            with rasterio.open(tmp_path) as src:
                # Read image
                if src.count >= 3:
                    active_image = src.read([1, 2, 3])
                    active_image = np.transpose(active_image, (1, 2, 0))
                else:
                    active_image = src.read(1)
                    active_image = np.stack([active_image]*3, axis=-1)
                
                # Check georeference status
                is_georeferenced = src.crs is not None
                if is_georeferenced:
                    raster_meta = {
                        "crs": str(src.crs),
                        "transform": src.transform,
                        "bounds": src.bounds,
                        "gsd": (abs(src.transform.a), abs(src.transform.e)),
                        "shape": (src.height, src.width)
                    }
        except Exception as e:
            # Fallback to OpenCV for standard images
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                active_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                active_image = cv2.cvtColor(active_image, cv2.COLOR_BGR2RGB)
                is_georeferenced = False
            except Exception as e2:
                st.error("Failed to decode uploaded image. Invalid format.")
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass

    # Dashboard Content
    if active_image is not None:
        # 1. Metadata and Input preview section
        st.markdown("<div class='section-header'>1. Input Specifications</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(active_image, caption=active_filename, use_container_width=True)
            
        with col2:
            st.markdown(f"**Filename:** `{active_filename}`")
            st.markdown(f"**Dimensions:** `{active_image.shape[1]} x {active_image.shape[0]} px`")
            
            if is_georeferenced:
                st.success("Status: GEOREFERENCED (Metric Mode Enabled)")
                st.markdown(f"**CRS:** `{raster_meta['crs']}`")
                st.markdown(f"**Resolution (GSD):** `{raster_meta['gsd'][0]:.2f}m x {raster_meta['gsd'][1]:.2f}m per pixel`")
                st.markdown(f"**Geospatial Bounds:**  \nLeft: `{raster_meta['bounds'].left:.1f}`  \nBottom: `{raster_meta['bounds'].bottom:.1f}`  \nRight: `{raster_meta['bounds'].right:.1f}`  \nTop: `{raster_meta['bounds'].top:.1f}`")
            else:
                st.warning("Status: NON-GEOREFERENCED (Relative Mode Only)")
                st.info("No spatial tags or coordinate reference systems detected. Outputs will be normalized relative scales.")

        # 2. Trigger run button
        st.markdown("<div class='section-header'>2. Reconstruct Elevation Surface</div>", unsafe_allow_html=True)
        run_wizard = st.button("RUN DEPTHWIZARD", type="primary")
        
        if run_wizard or 'dsm_pred' in st.session_state:
            # Lazy execution or session state recall
            if run_wizard:
                with st.spinner("Processing optical inference and elevation mapping..."):
                    # Process Relative Depth
                    h, w = active_image.shape[:2]
                    depth_map = depth_model.infer(active_image, active_filename, target_hw=(h, w))
                    
                    if is_georeferenced:
                        # Georeferenced Pipeline (Mode B)
                        # Load Ground Truth DSM from file to simulate coarse DEM (controlled proxy experiment)
                        dsm_truth_path = DATA_DIR / "dsm" / active_filename
                        if dsm_truth_path.exists():
                            gt = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                            dtm_true = create_synthetic_dtm(gt.shape)
                            dsm_true = dtm_true + gt
                            
                            coarse = downsample_dsm(dsm_true, factor=30)
                            dem_up = upsample_dem(coarse, dsm_true.shape)
                            dtm_pred = estimate_dtm(dem_up, kernel_size=91)
                            
                            # Footprint mask
                            s = {"id": active_filename, "rgb": active_image, "gt": gt, "depth": depth_map, "nodata": -999.0}
                            if has_unet:
                                res = footprint_estimator.cfg.train_res
                                x = footprint_estimator._prep_x(s, res)
                                xt = torch.from_numpy(x[None]).float().to(footprint_estimator.device)
                                depth_r = cv2.resize(depth_map, (res, res), interpolation=cv2.INTER_LINEAR)
                                raw_d = torch.from_numpy(depth_r[None]).float().to(footprint_estimator.device)
                                with torch.no_grad():
                                    mask_logits, _, _, _, _ = footprint_estimator.model(xt, raw_d, device=footprint_estimator.device)
                                probs = torch.sigmoid(mask_logits).squeeze(0).cpu().numpy()
                                mask_bldg = cv2.resize((probs > 0.5).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0.5
                            else:
                                d_coarse = cv2.resize(depth_map, (17, 17), interpolation=cv2.INTER_AREA)
                                d_smooth = cv2.resize(d_coarse, (512, 512), interpolation=cv2.INTER_LINEAR)
                                mask_bldg = (depth_map - d_smooth) > 2.0
                                
                            pred_delta_dense = np.zeros_like(dem_up)
                            num_labels, labels_im = cv2.connectedComponents(mask_bldg.astype(np.uint8))
                            coarse_ndsm_up = np.maximum(0.0, dem_up - dtm_pred)
                            for label in range(1, num_labels):
                                b_mask = labels_im == label
                                feat = extract_building_features(s, b_mask, coarse_ndsm_up, depth_map)
                                if feat is not None:
                                    x_feat = np.array([feat[c] for c in feature_cols])
                                    x_feat_norm = (x_feat - mu_train) / (sigma_train + 1e-6)
                                    with torch.no_grad():
                                        pred_delta = peak_mlp(torch.from_numpy(x_feat_norm[None]).float()).numpy()[0]
                                    pred_delta_dense[b_mask] = pred_delta
                                    
                            refined_ndsm = coarse_ndsm_up + pred_delta_dense
                            dsm_pred = dtm_pred + refined_ndsm
                            
                            st.session_state.dsm_pred = dsm_pred
                            st.session_state.refined_ndsm = refined_ndsm
                            st.session_state.dtm_pred = dtm_pred
                            st.session_state.depth_map = depth_map
                        else:
                            st.warning("Metric-scale auxiliary elevation source required for this coordinates area.")
                            st.session_state.dsm_pred = None
                    else:
                        # Non-Georeferenced Pipeline (Mode A)
                        # Normalize depth to arbitrary relative scale [0.0, 10.0]
                        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
                        relative_dsm = depth_norm * 10.0
                        
                        st.session_state.dsm_pred = relative_dsm
                        st.session_state.refined_ndsm = relative_dsm
                        st.session_state.dtm_pred = np.zeros_like(relative_dsm)
                        st.session_state.depth_map = depth_map
                        
            # Display processing results if populated in session state
            dsm_pred = st.session_state.dsm_pred
            
            if dsm_pred is not None:
                st.markdown("<div class='section-header'>3. Elevation Model Outputs</div>", unsafe_allow_html=True)
                col3, col4 = st.columns([1, 1])
                
                with col3:
                    # Normalize depth for visualization
                    depth_viz = (st.session_state.depth_map - st.session_state.depth_map.min()) / (st.session_state.depth_map.max() - st.session_state.depth_map.min() + 1e-6)
                    st.image(depth_viz, caption="Relative Optical Depth (Depth Anything V2)", use_container_width=True, clamp=True)
                    
                with col4:
                    label_title = "ABSOLUTE DSM" if is_georeferenced else "RELATIVE DSM"
                    st.image(dsm_pred / (dsm_pred.max() + 1e-6), caption=f"Reconstructed Surface ({label_title})", use_container_width=True, clamp=True)
                    
                # Height Statistics section
                st.markdown("<div class='section-header'>4. Height & Structural Statistics</div>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{dsm_pred.min():.1f}m</div><div class='metric-label'>Min Elevation</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{dsm_pred.max():.1f}m</div><div class='metric-label'>Max Elevation</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{np.mean(dsm_pred):.1f}m</div><div class='metric-label'>Mean Elevation</div></div>", unsafe_allow_html=True)
                with m4:
                    if is_georeferenced:
                        st.markdown(f"<div class='metric-card'><div class='metric-value'>{st.session_state.refined_ndsm.max():.1f}m</div><div class='metric-label'>Estimated Max Structure Height</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='metric-card'><div class='metric-value'>N/A</div><div class='metric-label'>Estimated Max Structure Height</div></div>", unsafe_allow_html=True)
                        
                # 3D Visualizer
                st.markdown("<div class='section-header'>5. 3D Terrain mesh Viewer (Interactive Render)</div>", unsafe_allow_html=True)
                
                # Compute mesh coordinates
                h, w = dsm_pred.shape
                x_g = np.zeros((h, w), dtype=np.float64)
                y_g = np.zeros((h, w), dtype=np.float64)
                
                if is_georeferenced:
                    transform = raster_meta["transform"]
                else:
                    # Mock transform for standard images
                    transform = rasterio.transform.from_origin(0, 512, 1.0, 1.0)
                    
                for r in range(h):
                    for c in range(w):
                        x_g[r, c], y_g[r, c] = transform * (c, r)
                        
                # Render exaggerated Z coordinates
                z_ex = dsm_pred * exaggeration
                points = np.stack([x_g, y_g, z_ex], axis=-1).reshape(-1, 3)
                
                grid = pv.StructuredGrid()
                grid.points = points
                grid.dimensions = (w, h, 1)
                mesh = grid.extract_surface(algorithm='dataset_surface')
                
                # UV texture coordinate mapping
                u = np.linspace(0.0, 1.0, w)
                v = np.linspace(1.0, 0.0, h)
                u_grid, v_grid = np.meshgrid(u, v)
                mesh.active_texture_coordinates = np.stack([u_grid, v_grid], axis=-1).reshape(-1, 2)
                mesh['Elevation'] = dsm_pred.ravel()
                mesh.set_active_scalars('Elevation')
                
                # Choose camera angle position vector
                x_mid = float(np.mean(x_g))
                y_mid = float(np.mean(y_g))
                z_mid = float(np.mean(z_ex))
                
                cameras = {
                    "Overhead": [(x_mid, y_mid, z_mid + 400), (x_mid, y_mid, z_mid), (0, 1, 0)],
                    "Oblique": [(x_mid - 250, y_mid - 250, z_mid + 200), (x_mid, y_mid, z_mid), (0, 0, 1)],
                    "Perspective": [(x_mid, y_mid - 300, z_mid + 150), (x_mid, y_mid, z_mid), (0, 0, 1)]
                }
                cam_pos = cameras[camera_angle]
                
                # Headless PyVista plotter rendering
                plotter = pv.Plotter(off_screen=True)
                
                if render_mode == "RGB Texture":
                    tex = pv.numpy_to_texture(active_image)
                    plotter.add_mesh(mesh, texture=tex, show_edges=False)
                elif render_mode == "Elevation-Colored":
                    plotter.add_mesh(mesh, scalars='Elevation', cmap="jet", show_edges=False)
                else:
                    contours = mesh.contour(isosurfaces=15, scalars='Elevation')
                    plotter.add_mesh(mesh, color="gray", opacity=0.5, show_edges=False)
                    plotter.add_mesh(contours, color="red", line_width=2)
                    
                plotter.camera_position = cam_pos
                
                # Capture rendering image bytes to display
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    tmp_img_path = tmp_img.name
                plotter.screenshot(tmp_img_path)
                plotter.close()
                
                # Display screenshot
                st.image(tmp_img_path, caption=f"3D Render View: {render_mode} ({camera_angle})", use_container_width=True)
                
                # Provide downloads/export section
                st.markdown("<div class='section-header'>6. Export Asset Downloads</div>", unsafe_allow_html=True)
                
                # Export GeoTIFF files
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_dsm:
                    tmp_dsm_path = tmp_dsm.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_ndsm:
                    tmp_ndsm_path = tmp_ndsm.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".vtp") as tmp_vtp:
                    tmp_vtp_path = tmp_vtp.name
                    
                # Write DSM GeoTIFF
                if is_georeferenced:
                    profile = {
                        "driver": "GTiff", "dtype": "float32", "nodata": -999.0,
                        "width": w, "height": h, "count": 1,
                        "crs": raster_meta["crs"],
                        "transform": raster_meta["transform"]
                    }
                else:
                    profile = {
                        "driver": "GTiff", "dtype": "float32", "nodata": -999.0,
                        "width": w, "height": h, "count": 1,
                        "crs": None, "transform": rasterio.transform.from_origin(0, h, 1, 1)
                    }
                    
                with rasterio.open(tmp_dsm_path, "w", **profile) as dst:
                    dst.write(dsm_pred.astype(np.float32), 1)
                with rasterio.open(tmp_ndsm_path, "w", **profile) as dst:
                    dst.write(st.session_state.refined_ndsm.astype(np.float32), 1)
                    
                # Write polydata mesh
                mesh.save(tmp_vtp_path)
                
                with open(tmp_dsm_path, "rb") as f:
                    dsm_bytes = f.read()
                with open(tmp_ndsm_path, "rb") as f:
                    ndsm_bytes = f.read()
                with open(tmp_vtp_path, "rb") as f:
                    vtp_bytes = f.read()
                    
                # Clean temp files
                for p in [tmp_dsm_path, tmp_ndsm_path, tmp_vtp_path, tmp_img_path]:
                    try:
                        os.remove(p)
                    except:
                        pass
                        
                # Download buttons
                ex1, ex2, ex3 = st.columns(3)
                with ex1:
                    st.download_button(
                        label="Download DSM GeoTIFF",
                        data=dsm_bytes,
                        file_name=f"Reconstructed_DSM_{active_filename}",
                        mime="image/tiff"
                    )
                with ex2:
                    st.download_button(
                        label="Download refined nDSM",
                        data=ndsm_bytes,
                        file_name=f"Refined_nDSM_{active_filename}",
                        mime="image/tiff"
                    )
                with ex3:
                    st.download_button(
                        label="Download 3D Mesh (.vtp)",
                        data=vtp_bytes,
                        file_name=f"Mesh_3D_{active_filename.replace('.tif', '')}.vtp",
                        mime="application/octet-stream"
                    )
                    
    else:
        st.info("Upload an image in the sidebar or click 'Load Demo Scene' to begin.")
