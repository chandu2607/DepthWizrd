# Architecture Gap Analysis — DepthWizard

## Overview
This document evaluates the software architecture of DepthWizard, contrasting the existing monolithic script structure with the target modular production architecture required for SIH 2025.

---

## 1. Existing Architecture vs Target Architecture

```
Existing Monolithic Architecture:
┌────────────────────────────────────────────────────────┐
│                        app.py                          │
│ ┌────────────────┐ ┌───────────────┐ ┌───────────────┐ │
│ │ Hardcoded load │ │ Monolithic ML │ │ Static PyVista│ │
│ │ & Rasterio     │ │ inference     │ │ Render        │ │
│ └────────────────┘ └───────────────┘ └───────────────┘ │
└────────────────────────────────────────────────────────┘

Target Modular Architecture:
┌──────────────────────────────────────────────────────────────────────────┐
│                             Streamlit UI                                 │
│ ┌──────────────┬──────────────┬──────────────┬─────────────┬───────────┐ │
│ │ Input & Meta │ 3D WebGL     │ Height &     │ Validation  │ Export &  │ │
│ │ Manager      │ Flythrough   │ Slope Tools  │ Dashboard   │ Assets    │ │
│ └──────▲───────┴──────▲───────┴──────▲───────┴──────▲──────┴─────▲─────┘ │
└────────┼──────────────┼──────────────┼──────────────┼────────────┼───────┘
         │              │              │              │            │
┌────────┴──────────────┴──────────────┴──────────────┴────────────┴───────┐
│                        depthwizard Core Package                          │
│                                                                          │
│ ├── data/ (raster_loader.py, geotiff_meta.py)                            │
│ ├── depth/ (depth_anything.py, cache.py)                                 │
│ ├── calibration/ (engine.py, dem_anchored.py, gcp_anchored.py)           │
│ ├── models/ (building_conditioned_net.py, peak_recovery.py)             │
│ ├── analysis/ (height.py, slope.py, massing.py)                          │
│ ├── metrics/ (height_metrics.py, validation.py)                          │
│ └── viz/ (interactive_viewer.py, mesh_generator.py, colormaps.py)        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Identified Architectural Gaps

| Component | Current Implementation | Architectural Defect | Required Remediation |
|:--|:--|:--|:--|
| **Input Ingestion** | Inlined inside `app.py` | Couples UI logic directly to rasterio file handles | Abstract into `depthwizard.data.raster_loader` |
| **Calibration** | Inlined inside `app.py` execution block | Hardcoded to coarse DEM + Phase 29; cannot select GCP or relative mode dynamically | Create `depthwizard.calibration.engine` with strategy pattern |
| **3D Rendering** | Headless PyVista PNG screenshots | Non-interactive static rendering; prevents true flythrough | Implement `depthwizard.viz.interactive_viewer` with Three.js HTML WebGL bridge |
| **Slope Analysis** | None | Feature completely missing from codebase | Create `depthwizard.analysis.slope` computing gradient degrees and aspect |
| **Height Analysis** | Inlined table in `app.py` | Building massing metrics not structured or sortable | Create `depthwizard.analysis.height` returning structured DataFrames |
| **Validation** | Standalone script only | UI lacks real-time validation comparison against GT | Embed dynamic validation dashboard into `app.py` |

---

## 3. Module Boundaries & Data Flow

1. **`RasterLoader`** $\to$ Output: `(RGB, is_geo, metadata, bounds, crs, gsd)`
2. **`DepthEstimator`** $\to$ Output: `d_relative (0..1 normalized)`
3. **`CalibrationEngine`** $\to$ Inputs: `(d_relative, is_geo, meta, mode, reference_dem, gcps)` $\to$ Output: `(DSM_pred, DTM_pred, nDSM_pred, is_metric)`
4. **`MeshGenerator`** $\to$ Inputs: `(DSM_pred, DTM_pred, RGB, exaggeration)` $\to$ Output: `(vertices, faces, uvs, building_walls, building_stats)`
5. **`InteractiveViewer`** $\to$ Bridge: Converts mesh topology to Three.js BufferGeometry payload $\to$ Renders in browser.
6. **`SlopeAnalyzer`** $\to$ Inputs: `(DSM_pred, transform)` $\to$ Output: `(slope_deg, slope_pct, aspect_deg, terrain_slope_deg)`
7. **`ValidationEngine`** $\to$ Inputs: `(DSM_pred, DSM_gt, mask_bldg)` $\to$ Output: `(metrics_dict, error_map, binned_stats)`
