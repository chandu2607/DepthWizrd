# DepthWizard Complete Codebase & Execution Path Audit
**Audit Date**: 2026-08-30  
**Target**: SIH Problem Statement ID 26175 ("DepthWizard — Single-View Height Estimation and 3D Flythrough")

---

## 1. Active Execution Path Trace

```
[User Action: Run Pipeline / Load Demo]
         │
         ▼
[app.py : Section 1 — Input Ingestion]
         │  Calls: depthwizard.data.raster_loader.load_raster_input()
         │  Returns: RasterInput (RGB, is_georeferenced, CRS, transform, GSD, shape)
         │
         ▼
[app.py : Section 2 — Monocular Depth Inference]
         │  Calls: depthwizard.depth.depth_anything.DepthAnythingV2.infer()
         │  Returns: depth_raw (2D relative depth in [0, 1])
         │
         ▼
[app.py : Section 2 — Metric Elevation Calibration Engine]
         │  Calls: depthwizard.calibration.engine.CalibrationEngine.calibrate()
         │  Executes:
         │    - DTM ground-plane decomposition (Phase 30 morphology)
         │    - nDSM computation: max(0, DSM_coarse - DTM)
         │    - PeakRecoveryMLP ensemble (Phase 29: seed_0 + seed_1) for tall structures
         │  Returns: CalibrationResult (DSM, DTM, nDSM, mask_bldg, units, is_metric)
         │
         ▼
[app.py : Section 3 — Interactive 3D WebGL Flythrough & Orbit Viewer]
         │  Calls: depthwizard.viz.interactive_viewer.generate_interactive_webgl_html()
         │  Executes:
         │    - build_city_geometry()
         │    - Layer 1: Subsampled DTM PlaneGrid (in meters)
         │    - Layer 2: DSM Building Roofs with Ear-Clipping triangulation
         │    - Layer 3: Vertical Architectural Facade Walls
         │    - Pre-encodes Satellite Orthophoto texture (base64 JPEG)
         │    - Generates self-contained Three.js WebGL payload
         │  Renders: Streamlit components.html(webgl_html, height=720)
         │
         ▼
[app.py : Sections 4–7 — Analytics, Validation & Probing]
         │  - compute_slope() -> slope_deg, slope_pct, aspect
         │  - analyze_building_massing() -> BuildingRecord DataFrame
         │  - probe_point_elevation() -> interactive point inspection
         │  - run_validation() -> MAE, RMSE, Pearson R, binned height table
```

---

## 2. Module Traceability & Integrity Status

| Module | File Location | Responsibility | Scientific Integrity Status |
|:---|:---|:---|:---:|
| **Raster Ingestion** | `depthwizard/data/raster_loader.py` | Universal PNG/JPG/GeoTIFF ingestion & CRS parsing | **LOCKED & VERIFIED** |
| **Depth Backbone** | `depthwizard/depth/depth_anything.py` | Depth Anything V2 ViT-Small inference with cache | **LOCKED & VERIFIED** |
| **Calibration Engine** | `depthwizard/calibration/engine.py` | 5-mode strategy (DEM, GCP, Structural MLP, DTM) | **LOCKED & VERIFIED** |
| **Peak Recovery MLP** | `runs/phase29_peak_recovery/` | 18-feature MLP ensemble for skyscraper recovery | **BYTE-IDENTICAL LOCK** |
| **Interactive Viewer** | `depthwizard/viz/interactive_viewer.py` | 3-Layer WebGL Scene Graph (DTM + Roofs + Walls) | **ACTIVE & ENHANCED** |
| **Height Analytics** | `depthwizard/analysis/height.py` | Building massing & point elevation probe | **VERIFIED** |
| **Slope Analytics** | `depthwizard/analysis/slope.py` | Sobel gradient slope with facade masking | **VERIFIED** |
| **Validation Engine** | `depthwizard/metrics/validation.py` | Metric accuracy & binned statistics | **VERIFIED** |
| **Main Web Application**| `app.py` | Unified Streamlit multi-tab production UI | **ACTIVE & LIVE** |

---

## 3. Ambiguity & Stale Function Audit
- **Old PyVista Static Viewer**: Completely decoupled from the live interactive viewer; retained strictly as an offline benchmarking script (`scripts/run_phase31g_building_aware_3d.py`).
- **Old BoxGeometry Heightfield**: Fully replaced by the 3-Layer Ear-Clipped Building-Aware Three.js WebGL engine in `depthwizard/viz/interactive_viewer.py`.
- **Session State & Iframe Isolation**: Mode switching (RGB, Elevation, Height, Slope) and camera presets execute entirely client-side inside the Three.js canvas context at 60fps without triggering Streamlit reruns.
