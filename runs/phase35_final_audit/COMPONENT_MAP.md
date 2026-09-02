# DepthWizard Component Hierarchy & Module Map

## 1. System Architecture Diagram

```
DepthWizard Root Application (app.py)
 ├── Ingestion Tier (depthwizard/data/)
 │    └── raster_loader.py (RasterInput, rasterio EPSG parser, PNG/JPG relative fallback)
 │
 ├── AI Inference Tier (depthwizard/depth/)
 │    └── depth_anything.py (DepthAnythingV2 ViT-Small, MD5 caching engine)
 │
 ├── Calibration & Scientific Reconstruction Tier (depthwizard/calibration/)
 │    ├── engine.py (CalibrationEngine: 5 Modes)
 │    │    ├── CalibrationMode.MONOCULAR_RELATIVE
 │    │    ├── CalibrationMode.DEM_ANCHORED
 │    │    ├── CalibrationMode.GCP_ANCHORED
 │    │    ├── CalibrationMode.GROUND_REFERENCED (Phase 30 DTM)
 │    │    └── CalibrationMode.STRUCTURAL_PRIOR (Phase 29 PeakRecoveryMLP)
 │    └── checkpoints/
 │         ├── phase29_peak_recovery_seed0.pt
 │         └── phase29_peak_recovery_seed1.pt
 │
 ├── 3D Visualization & Flythrough Tier (depthwizard/viz/)
 │    └── interactive_viewer.py (generate_interactive_webgl_html)
 │         ├── Ear-Clipping Polygon Triangulator (triangulate_polygon_earcut)
 │         ├── Layer 1: Subsampled DTM PlaneGrid (in physical meters)
 │         ├── Layer 2: DSM Building Roofs (Orthophoto UVs + Bilateral DSM heights)
 │         ├── Layer 3: Vertical Slate Architectural Facade Walls
 │         ├── Live Client-Side Mode Switcher (RGB, Elevation, Height, Slope)
 │         ├── Raycaster Building Selector & Live Inspector HUD
 │         ├── Camera Presets & Bounding-Box Autocenter
 │         └── WASD/Arrows Free-Fly & Cinematic Flythrough Trajectory
 │
 ├── Geospatial Analytics Tier (depthwizard/analysis/)
 │    ├── height.py (analyze_building_massing, probe_point_elevation)
 │    └── slope.py (compute_slope: slope_deg, slope_pct, aspect, facade mask)
 │
 └── Scientific Validation Tier (depthwizard/metrics/)
      └── validation.py (run_validation: MAE, RMSE, Pearson R, binned height stats)
```

---

## 2. Data Flow Contract

| Interface Step | Input Data | Output Data | Coordinate System |
|:---|:---|:---|:---|
| **Loader** | Raw File Path (GeoTIFF/PNG/JPG) | `RasterInput` (uint8 RGB, GSD, Affine, CRS) | Native Image / Projected CRS |
| **Backbone** | RGB array $(H \times W \times 3)$ | `depth_raw` $(H \times W \in [0, 1])$ | Normalized Relative |
| **Calibration** | `depth_raw` + `RasterInput` | `CalibrationResult` (DSM, DTM, nDSM, mask) | True Metric Elevation ($Z$ in meters) |
| **3D Geometry** | DSM, DTM, mask, RGB | `build_city_geometry()` JSON dictionary | Local Metric ($X \in [-W/2, W/2]$, $Z \in [-H/2, H/2]$) |
| **3D WebGL** | `build_city_geometry()` JSON | Standalone Three.js HTML Payload | 60 FPS WebGL Scene Context |
