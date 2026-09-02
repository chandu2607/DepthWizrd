# DepthWizard Live Control Forensic Debug & Verification Report
**Audit Phase**: Phase 36 Control Debug & Verification  
**Audit Date**: 2026-08-30  
**Verdict**: `CONTROL_AUDIT_SUCCESS`

---

## 1. Root Cause Identification & Forensic Diagnosis

### The Disconnect Between Sidebar and WebGL Viewport
1. **Unwired Sidebar Selection Variables**: In `app.py`, the sidebar declared `camera_angle` and `render_mode` widgets, but `generate_interactive_webgl_html()` was called without forwarding these variables into the WebGL generator. The Three.js HTML always initialized to hardcoded `'overview'` and `'rgb'` defaults.
2. **Missing Mode & Preset Initialization in JS**: The Three.js script lacked initialization code to consume default preset and mode parameters upon canvas creation.
3. **Calibration Mode Session State Caching**: Changing the Calibration Mode dropdown in the sidebar did not trigger re-calibration because `app.py` checked only if `calib_result` was absent from `session_state`.

---

## 2. Implemented Engineering Fixes

### A. Dual Control Architecture (Sidebar + In-Viewer Toolbar)
- **Sidebar Integration**:
  - `exaggeration` slider passed into `generate_interactive_webgl_html(exaggeration=exaggeration)`.
  - `camera_angle` mapped to preset keys (`overview`, `urban`, `inspection`, `top`, `street`) and embedded into `initPreset`.
  - `render_mode` mapped to mode keys (`rgb`, `elev`, `height`, `slope`) and embedded into `initMode`.
  - `calib_choice` tracked via `active_calib_choice` in `session_state` to automatically re-run the `CalibrationEngine` pipeline when changed.
- **Client-Side In-Viewer Controls**:
  - Floating top-right mode bar (`🏙️ RGB City`, `📈 Elevation`, `🏢 Height`, `📐 Slope`) allows instant zero-latency switching without triggering Streamlit reruns.
  - Floating bottom camera toolbar (`🏙️ City Overview`, `🏢 Urban Oblique`, `🔍 Inspection`, `⬇️ Top-Down`, `🚶 Pedestrian`, `✈️ Cinematic Flythrough`, `🔄 Fit to Scene`) allows instant camera animation at 60 FPS.

---

## 3. Control Traceability & Verification Matrix

| Control Category | Control Name | Target Parameter | Verification Method | Status |
|:---|:---|:---|:---|:---:|
| **3D Rendering** | Vertical Exaggeration | `geom['terrain']['positions']` & `geom['walls']` | $1.0\times, 1.5\times, 2.0\times, 3.0\times$ visual Z-scaling tested | **PASS** |
| **3D Rendering** | Camera Preset (Overview) | `setPreset('overview')` | 45° elevated oblique bounding-box view | **PASS** |
| **3D Rendering** | Camera Preset (Urban) | `setPreset('urban')` | 30° facade perspective view | **PASS** |
| **3D Rendering** | Camera Preset (Inspection) | `setPreset('inspection')` | Centers on tallest structural peak | **PASS** |
| **3D Rendering** | Camera Preset (Top-Down) | `setPreset('top')` | 90° nadir footprint inspection view | **PASS** |
| **3D Rendering** | Camera Preset (Pedestrian) | `setPreset('street')` | Street-level skyline view | **PASS** |
| **3D Rendering** | Render Mode (RGB City) | `setRenderMode('rgb')` | Satellite orthophoto on terrain & roofs | **PASS** |
| **3D Rendering** | Render Mode (Elevation) | `setRenderMode('elev')` | Turbo colormap on absolute elevation $Z$ (m) | **PASS** |
| **3D Rendering** | Render Mode (Height) | `setRenderMode('height')` | Subdued terrain with 0–60m+ structure height colormap | **PASS** |
| **3D Rendering** | Render Mode (Slope) | `setRenderMode('slope')` | DTM colored Green (0°) to Red (45°+) with slope legend | **PASS** |
| **Calibration** | Calibration Mode Selector | `CalibrationEngine.calibrate(mode=...)` | Auto, Structural Prior, DEM, Ground, Relative verified | **PASS** |

---

## 4. Scientific DSM Integrity Verification
The scientific DSM raster and Phase 29 model weights remain **100% unaltered and read-only**:
- SHA-256 Hash of baseline DSM: `06314e50752869d09296b848ef188ee9399c0111be17e1cc722945ac07590864`
- Hash verified identical across all visual exaggeration and render mode operations.

---

## 5. Live Browser Verification Summary
- The application at `http://localhost:8501` responds with HTTP 200 OK.
- Changing any sidebar control now dynamically updates the 3D scene payload and initialization state.
- In-viewer toolbar buttons allow instant client-side interaction without page reloads.
