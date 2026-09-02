# Requirements Gap Report — DepthWizard SIH Problem Statement 26175

## Executive Summary
This audit evaluates the current DepthWizard codebase against all 15 explicit technical specifications mandated by the Smart India Hackathon Problem Statement 26175 ("DepthWizard - Single-View Height Estimation and 3D Flythrough").

The audit classifies:
- **PASS**: 6 / 15 requirements fully supported and validated.
- **PARTIAL**: 7 / 15 requirements partially implemented but requiring UI integration, interactive capability, or modularization.
- **MISSING**: 2 / 15 requirements not yet implemented (Slope Analysis Suite and Multi-View Feasibility Probe).

---

## Detailed Gap Analysis by Requirement Area

### 1. Non-Georeferenced (PNG/JPG) vs Georeferenced (GeoTIFF) Input Pipeline (REQ-01 & REQ-02)
- **Current Status**: PARTIAL / PASS
- **Current State**: `app.py` detects GeoTIFF CRS and bounds using `rasterio`. Non-georeferenced images default to a simple fallback normalized 0–10m.
- **Gap Identified**:
  - The UI needs distinct, unmistakable operational modes:
    - **Relative Mode (rDSM)** for PNG/JPG with clear relative colormaps and structural units.
    - **Absolute Mode (DSM)** for GeoTIFF with metric units (meters above sea level), GSD, and EPSG CRS metadata.
- **Action**: Formalize data loader and user banner with clear relative vs metric indicators.

### 2. Metric Calibration Engine (REQ-04 & REQ-05)
- **Current Status**: PARTIAL
- **Current State**: Phase 29 PeakRecoveryMLP + coarse DTM anchoring is embedded in `app.py` as an ad-hoc monolithic block.
- **Gap Identified**:
  - The SIH problem statement explicitly lists multiple calibration sources: scene statistics, low-resolution DEM/SRTM, minimal GCPs, and semantic priors.
  - The system lacks a clean, modular calibration engine where users can select or inspect the active calibration mode.
- **Action**: Implement `depthwizard/calibration/` with 5 modular strategies:
  1. `MonocularRelative`
  2. `DEMAnchored`
  3. `GroundReferenced`
  4. `GCPAnchored`
  5. `StructuralPrior` (Phase 29 PeakRecoveryMLP)

### 3. Interactive 3D Flythrough & Viewer (REQ-08)
- **Current Status**: PARTIAL
- **Current State**: The application uses headless PyVista to render static PNG snapshots from fixed camera angles (`City Overview`, `Urban`, `Inspection`, `Top View`).
- **Gap Identified**:
  - The problem statement explicitly requires **interactive navigation, flythrough, and first-person navigation**.
  - Static images do not allow the user to orbit, zoom, pan, or navigate through the urban canopy in real time.
- **Action**: Build and embed a lightweight **Three.js WebGL Interactive Viewer** directly in Streamlit, featuring:
  - 60fps WebGL rendering of terrain + architectural walls + textured roofs.
  - Full mouse/touch OrbitControls (orbit, rotate, pan, zoom).
  - First-person WASD / Arrow key keyboard navigation.
  - Automated Cinematic Flythrough button.
  - Presets (City Overview, Urban, Inspection, Top-Down, Street Level).
  - Retain PyVista for high-resolution `.vtp` exports and static PNG snapshots.

### 4. Structural Height Measurement Suite (REQ-09)
- **Current Status**: PARTIAL
- **Current State**: Global scalar summary cards exist in `app.py`, but individual building height inspection and point-to-point measurement tools are not interactive.
- **Gap Identified**:
  - Users and evaluators cannot click or select a specific building to inspect its ground elevation, roof elevation, and calculated height ($H = Z_{\text{roof}} - Z_{\text{ground}}$).
- **Action**: Implement interactive building massing inspector table with sortable columns, confidence ratings, and point-level coordinate/height inspection.

### 5. Terrain & Structural Slope Analysis Suite (REQ-10)
- **Current Status**: MISSING
- **Current State**: No slope computation or visualization exists in the current codebase.
- **Gap Identified**:
  - The problem statement explicitly requires **slope analysis**.
- **Action**: Implement `depthwizard/analysis/slope.py`:
  - $\text{Slope (deg)} = \arctan(\sqrt{(\partial z/\partial x)^2 + (\partial z/\partial y)^2}) \cdot \frac{180}{\pi}$.
  - Slope percentage and Aspect calculation.
  - Separate natural terrain slope from vertical building facades.
  - Colorized slope map layer in 2D and 3D.

### 6. Quantitative Validation Dashboard (REQ-11)
- **Current Status**: PARTIAL
- **Current State**: Metric calculation functions exist in `depthwizard/metrics/`, but `app.py` does not provide an interactive validation dashboard when reference data is present.
- **Gap Identified**:
  - Needs side-by-side reference vs prediction comparison, error maps, residual distribution histograms, and binned height accuracy breakdown ($<10\text{m}$, $10\text{--}20\text{m}$, $20\text{--}30\text{m}$, $30\text{--}40\text{m}$, $\ge 40\text{m}$).
- **Action**: Implement a dedicated Validation Dashboard tab in `app.py`.

### 7. Professor Advice Experiments Suite (REQ-13 & REQ-14)
- **Current Status**: PARTIAL / MISSING
- **Current State**: Phase 34 (pseudo-LiDAR) and Phase 35 (sparse metric anchors) were conducted as exploratory scripts.
- **Gap Identified**:
  - All four professor-suggested pathways (sparse depth completion, LiDAR fusion, pseudo-LiDAR, multi-view geometry) need a consolidated, well-documented home in `scripts/experiments/professor_methods/`.
- **Action**: Create clean, reproducible standalone scripts with explicit quantitative verdicts.

---

## Action Plan Priority Matrix
1. **P0 (Critical Path)**: Interactive 3D WebGL Viewer (Phase F) + Modular Calibration Engine (Phase D).
2. **P1 (Core Features)**: Height & Slope Analysis Suite (Phase G) + Validation Dashboard (Phase H).
3. **P2 (Scientific Rigor)**: Unified Input Engine (Phase C) + Professor Methods Suite (Phase E).
4. **P3 (Packaging)**: Deployment Guide & Final End-to-End Acceptance Report (Phase I & J).
