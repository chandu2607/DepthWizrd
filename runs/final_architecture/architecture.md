# Final System Architecture — DepthWizard (SIH Problem Statement 26175)

## 1. System Vision
DepthWizard converts single optical satellite imagery (PNG, JPG, TIFF, GeoTIFF) into scientifically validated Digital Surface Models (DSM / rDSM) and an interactive 3D WebGL flythrough environment with structural height, slope, and validation analytics.

---

## 2. Multi-Tier System Architecture

```
                                  [ User / Client Browser ]
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │                                                 │
          [ Streamlit UI Orchestration ]                  [ Three.js WebGL 3D Canvas ]
                     │                                    (60 FPS Orbit, Pan, Zoom,
                     │                                    WASD First-Person, Flythrough)
                     ▼                                                 ▲
 ┌──────────────────────────────────────────────────────────┐          │
 │                      Core Pipeline                       │          │
 │                                                          │          │
 │  1. Ingestion Engine (depthwizard.data.raster_loader)    │          │
 │     • Auto-detects PNG/JPG (Non-Geo) vs GeoTIFF (Geo)    │          │
 │     • Extracts CRS, GSD, Affine Transform, Bounds        │          │
 │                                                          │          │
 │  2. Monocular Depth Backbone (depthwizard.depth)         │          │
 │     • Pretrained Depth Anything V2                       │          │
 │     • MD5 Cache for instantaneous repeat inference       │          │
 │                                                          │          │
 │  3. Modular Calibration Engine (depthwizard.calibration) │          │
 │     • Mode A: Monocular Relative (rDSM)                  │          │
 │     • Mode B: DEM Anchored (Low-res DEM / SRTM)          │          │
 │     • Mode C: Ground Referenced (DTM Plane)              │          │
 │     • Mode D: GCP Anchored (Sparse Control Points)       │          │
 │     • Mode E: Structural Prior (Phase 29 PeakRecovery)   │          │
 │                                                          │          │
 │  4. Reconstruction & Mesh Engine (depthwizard.viz)       │──────────┘
 │     • Phase 30 DTM / nDSM Decomposition                  │
 │     • Phase 33D Architectural Wall Extrusion             │
 │     • Solid DSM Roof Surface Mesh                        │
 │     • Satellite RGB UV Texture Mapping                   │
 │                                                          │
 │  5. Analysis & Measurement Suite (depthwizard.analysis)  │
 │     • Structural Height Measurement (Point & Building)   │
 │     • Terrain vs Facade Slope Analysis (deg, %, aspect)  │
 │                                                          │
 │  6. Validation & Quality Suite (depthwizard.metrics)     │
 │     • MAE, RMSE, Pearson R, Spearman Rho, Bias, P90      │
 │     • Binned Height Accuracy (<10m to >=40m)             │
 │     • 2D Error Map & Cross-Sectional Elevation Profiles  │
 └──────────────────────────────────────────────────────────┘
```

---

## 3. Data Ingestion & Routing Logic

### Mode A: Non-Georeferenced (PNG / JPG)
- **Input**: Optical RGB image without embedded coordinates.
- **Backbone**: Monocular depth inference $\to d_{\text{norm}} \in [0, 1]$.
- **Calibration**: Monocular Relative mode.
- **Output**: **Relative DSM (rDSM)** normalized on a relative structural scale.
- **3D Visualization**: Relative 3D terrain with original RGB texture projection.
- **UI Flag**: `⚠️ RELATIVE ELEVATION MODE — No Georeference`.

### Mode B: Georeferenced (GeoTIFF)
- **Input**: GeoTIFF raster with CRS, Affine Transform, and GSD.
- **Backbone**: Monocular depth inference $\to d_{\text{norm}} \in [0, 1]$.
- **Calibration**: DEM Anchored / GCP / Phase 29 PeakRecoveryMLP.
- **Output**: **Absolute DSM** with metric elevations in meters above sea level.
- **3D Visualization**: Georeferenced metric 3D city scene with separate DTM terrain, architectural building side walls, and textured roofs.
- **UI Flag**: `✅ ABSOLUTE DSM MODE — Georeferenced (EPSG:xxxx)`.

---

## 4. Calibration Engine Mathematical Formulations

1. **Relative Normalization**:
   $$Z_{\text{rel}}(x, y) = 10.0 \cdot \frac{d(x, y) - d_{\min}}{d_{\max} - d_{\min} + \epsilon}$$

2. **DEM Anchoring & DTM Decomposition**:
   $$\text{DTM}(x, y) = \mathcal{M}_{\text{ground}}(\text{DEM}_{\text{coarse}}(x, y))$$
   $$\text{nDSM}_{\text{coarse}}(x, y) = \max(0, \text{DEM}_{\text{coarse}}(x, y) - \text{DTM}(x, y))$$

3. **Phase 29 Structural Prior (PeakRecoveryMLP)**:
   $$\mathbf{f}_i = \left[\text{dem}_{\text{mean}}, \dots, \text{depth}_{\text{p95}}, \text{area}, \text{aspect\_ratio}, \text{compactness}\right]$$
   $$\Delta H_i = \text{MLP}_{\text{ensemble}}\left(\frac{\mathbf{f}_i - \boldsymbol{\mu}}{\boldsymbol{\sigma}}\right)$$
   $$\text{nDSM}_{\text{refined}}(x, y) = \text{nDSM}_{\text{coarse}}(x, y) + \sum_i \Delta H_i \cdot \mathbb{I}_{i}(x, y)$$
   $$\text{DSM}_{\text{pred}}(x, y) = \text{DTM}(x, y) + \text{nDSM}_{\text{refined}}(x, y)$$

4. **GCP Transformation Fit**:
   $$Z_{\text{metric}} = a \cdot Z_{\text{relative}} + b + \mathbf{w}^T \mathbf{x}_{\text{geo}}$$
   Fitted via Huber robust loss on user/reference control points.

---

## 5. Interactive 3D Geometry Specifications
- **Terrain Surface**: Planar quad mesh with DTM elevation and RGB texture UVs.
- **Rooftops**: DSM-derived surface with bilateral edge-preserving smoothing (`d=5`, `sigma=3.0`).
- **Walls**: Vertical extruded quads between ground footprint boundary ($\text{DTM}$) and roof peak ($\text{DSM}_{p95}$).
- **Material Properties**:
  - Terrain & Roofs: Textured with 1:1 satellite RGB.
  - Walls: Flat slate-gray architectural shading (`#1E293B`) to eliminate texture stretching artifacts.
- **Viewer Controls**: Three.js WebGL canvas with OrbitControls, WASD first-person camera, and automated flythrough spline.
