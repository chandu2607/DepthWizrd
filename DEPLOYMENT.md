# DepthWizard — Deployment & User Operation Manual
**Problem Statement ID**: 26175 | **Project**: DepthWizard — Single-View Height Estimation & 3D Flythrough

---

## 1. Overview & System Requirements
DepthWizard is a lightweight, unified software platform that transforms single-view optical satellite or drone imagery into verified 3D Digital Surface Models (DSM) and an interactive 3D WebGL flythrough environment with structural height, slope, and validation analytics.

### Minimum System Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: 3.10 or 3.11
- **RAM**: 8 GB minimum (16 GB recommended for high-res urban tiles)
- **GPU**: Optional (CUDA supported for accelerated inference; CPU fallback fully functional)
- **Browser**: Chrome, Edge, Firefox, or Safari with WebGL enabled

---

## 2. Installation & Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/chandu2607/DepthWizard.git
cd DepthWizard

# 2. Create and activate a clean virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt
```

### Essential Dependencies Included:
- `streamlit>=1.35.0`: Interactive Web UI and dashboard
- `torch>=2.0.0`, `torchvision`: Neural network inference
- `rasterio>=1.3.0`: Geospatial TIFF, CRS, and transform parsing
- `pyvista>=0.43.0`: 3D mesh processing and VTP export
- `opencv-python-headless`: Fast morphological operations and DTM filtering
- `scipy`, `pandas`, `numpy`, `matplotlib`: Scientific analytics and validation

---

## 3. Pretrained Model Checkpoints
All required model weights are packaged within the repository:
1. **Depth Anything V2 (ViT-Small)**: Automatically cached in `data/dfc2023_multicity/depth_cache/`
2. **Phase 24 Building Footprint U-Net**: Stored at `runs/phase24_moe/seed_0/model.pt`
3. **Phase 29 PeakRecoveryMLP (Ensemble)**:
   - `runs/phase29_peak_recovery/seed_0/model.pt`
   - `runs/phase29_peak_recovery/seed_1/model.pt`
   - `runs/phase29_peak_recovery/normalization_stats.json`

---

## 4. Single-Command Application Launch

To start the local web application:
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 5. End-to-End User Workflow Guide

### Step 1: Input Ingestion
- **Demo Mode**: Click `🏙️ Load Demo Scene (NYC)` in the sidebar for an immediate validated Manhattan urban scene.
- **Custom Upload**: Upload any PNG, JPG, or GeoTIFF image. The system automatically inspects geospatial headers:
  - **PNG / JPG** $\to$ `⚠️ RELATIVE ELEVATION MODE (rDSM)`
  - **GeoTIFF** $\to$ `✅ ABSOLUTE DSM MODE (EPSG CRS & GSD in meters)`

### Step 2: Calibration Mode Selection
Choose an appropriate calibration strategy in the sidebar:
- `Auto (Best Validated)`: Automatically applies Phase 29 structural prior when georeferencing is detected, or relative rDSM for non-georeferenced inputs.
- `Structural Prior (Phase 29 PeakRecovery MLP)`: Corrects skyscraper peak heights using building geometry and learned priors.
- `DEM / SRTM Anchored`: Anchors relative depth to coarse reference terrain.
- `Ground Plane Referenced`: Subtracts local DTM ground plane ($H = Z_{\text{roof}} - Z_{\text{ground}}$).
- `Ground Control Points (GCP)`: Calibrates using user-supplied reference points.
- `Monocular Relative (rDSM)`: Pure optical relative elevation (0–10 scale).

### Step 3: Run Pipeline
Click **`🚀 EXECUTE DEPTHWIZARD PIPELINE`**. Execution takes ~0.5–1.5 seconds per tile.

### Step 4: Interactive 3D WebGL Flythrough
- **Mouse Controls**: Left-click to orbit, right-click to pan, scroll wheel to zoom.
- **First-Person Navigation**: Use **`W`**, **`A`**, **`S`**, **`D`** or arrow keys to navigate the ground level.
- **Cinematic Flythrough**: Click **`✈️ Cinematic Flythrough`** for an automated orbiting camera tour.
- **Camera Presets**: Switch between `City Overview`, `Urban Oblique`, `Inspection`, and `Top-Down`.

### Step 5: Structural Height & Slope Inspection
- **Building Massing Table**: Inspect individual structures sorted by height ($H$), roof elevation ($Z_{\text{roof}}$), ground base ($Z_{\text{ground}}$), and footprint area ($m^2$).
- **Point Elevation Probe**: Enter any pixel $(X, Y)$ coordinate to probe surface elevation, terrain ground elevation, and structural height.
- **Slope Analytics**: View mean terrain slope in degrees, 95th percentile slope, steep slope area percentage ($>25^\circ$), and maximum vertical facade gradient.

### Step 6: Quantitative Validation & Export
- **Validation**: When reference data is available, review MAE, RMSE, Pearson $R$, Spearman $\rho$, bias, P90 error, and height-binned accuracy.
- **Export Assets**:
  - `⬇️ Download DSM GeoTIFF`
  - `⬇️ Download nDSM GeoTIFF`
  - `⬇️ Download Building Massing (CSV)`
  - `⬇️ Download Validation Report (JSON)`

---

## 6. Automated Verification & Smoke Testing
To verify system integrity on a target machine:
```bash
python scripts/test_deployment_smoke.py
```
This tests all 6 core subsystems headlessly and confirms operational readiness.

---

## 7. Scientific Disclaimers & Known Limitations
1. **Monocular Scale Ambiguity**: Optical RGB imagery alone cannot provide absolute elevation in meters without external reference anchors (coarse DEM, SRTM, GCPs, or calibrated structural priors). Non-georeferenced images are honestly reported as relative rDSM (0–10 scale).
2. **Building Footprint Resolution**: Footprint precision depends on optical resolution ($0.5\text{m}$ GSD recommended for dense urban centers).
3. **Extreme Skyscrapers ($>100\text{m}$)**: Monocular shading cues saturate on supertall towers; Phase 29 PeakRecoveryMLP significantly reduces this gap but physical ground truth anchors remain recommended for ultra-tall structures.
