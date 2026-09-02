# Phase 60 — Actual Current DepthWizard Inference Path Audit

## Executive summary

This audit uses the repository as it exists in the workspace. It does not rewrite production code or invent a new model pipeline.

The current runtime path in the app is:

Input
 ↓
RGB preprocessing
 ↓
Depth Anything V2 relative-depth estimation
 ↓
Depth normalization
 ↓
CalibrationEngine.calibrate()
 ↓
DSM / rDSM reconstruction
 ↓
Slope analysis
 ↓
3D WebGL renderer / exported GeoTIFF assets

---

## 1. Production path in the repository

### Entry point

- [app.py](../../app.py)
- The app loads a raster input, builds a `RasterInput`, then calls `DepthAnythingV2.infer()` followed by `CalibrationEngine.calibrate()`.

### Current model architecture

The repository’s current model stack is still the same production stack used by the app:

1. Depth Anything V2 as the frozen optical depth prior
2. optional reference elevation for georeferenced scenes
3. calibration engine for converting learned depth to DSM / rDSM
4. terrain slope analysis
5. building-masking and height extraction logic
6. PyVista / WebGL 3D export and render flow

### Locked checkpoint(s)

The project uses existing model state files stored under the `runs/` directories, including:

- `runs/phase29_peak_recovery/seed_0/model.pt`
- `runs/phase29_peak_recovery/seed_1/model.pt`
- `runs/phase43_augmented_unet/unet_config_D.pt`
- `runs/phase24_moe/seed_0/model.pt`

These are loaded in `CalibrationEngine._load_models()`.

### RGB preprocessing

The `RasterInput` pipeline loads the optical image and normalizes it for downstream use. The app then passes the RGB scene directly to `DepthAnythingV2.infer()`.

Relevant implementation:

- `depthwizard/data/raster_loader.py`
- `app.py` calls `load_raster_input(...)`

### Depth Anything preprocessing

The wrapper in `depthwizard/depth/depth_anything.py` does the following:

- converts arrays to `uint8` if needed
- creates a PIL image from the RGB array
- resizes to the configured `input_size` if needed
- feeds the image into `transformers.pipeline("depth-estimation", ...)`
- resizes predicted depth to the target raster shape
- caches depth on disk by hash key

This is a relative-depth prior, not a metric sensor.

### Depth normalization

Inside `CalibrationEngine.calibrate()`:

- obtains `depth_raw`
- computes a min-max normalization:
  `d_norm = ((depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-6))`
- then uses `d_norm` to create either a relative-height output or a metric output depending on calibration mode

This is important: the system is not directly measuring terrain altitude from the optical image.

### Height calibration

The calibration engine contains multiple possible modes:

- `MONOCULAR_RELATIVE`
- `GCP_ANCHORED`
- `DEM_ANCHORED`
- `GROUND_REFERENCED`
- `STRUCTURAL_PRIOR`
- `AUTO`

The actual `calibrate()` method selects mode based on whether the input is georeferenced and whether reference elevation is available.

### DSM / rDSM generation

The calibration engine returns:

- `dsm`
- `dtm`
- `ndsm`
- `mask_bldg`

For non-georeferenced input it emits a relative output in the `0–10` range. For georeferenced reference-based paths it can produce metric outputs.

### Building-mask generation

The current building-mask logic is in:

- `CalibrationEngine.extract_building_footprint()`

It attempts to use a `BuildingConditionedEstimator` if a model checkpoint exists; otherwise it falls back to a morphology-based heuristic using depth residuals.

### Threshold logic

Threshold logic is present in the building mask extraction and in the adaptive calibration heuristics. The system uses heuristics such as:

- `probs > 0.5`
- residual depth thresholding based on standard deviation
- morphological operations for DTM extraction

### Georeferencing handling

`RasterInput` and the app attempt to detect georeferencing and CRS. For georeferenced raster inputs, the app exposes absolute DSM mode; otherwise it falls back to relative mode.

### Slope calculation

The slope pipeline is in `depthwizard/analysis/slope.py` and does:

- Sobel derivative computation on the elevation raster
- gradient magnitude to degrees
- mask-out of building facades
- statistical summary by terrain slope

### 3D mesh generation

The app passes the reconstructed DSM into `generate_interactive_webgl_html()` from the viewer module, which creates a WebGL scene. This is a visualization engine, not a true physical terrain sensor.

### Point-cloud functionality

There is no clear standalone point-cloud pipeline in the repo that is used as the current main geometry source. The point-cloud idea appears in project discussion and diagnostics, but the actual active production path is the raster + DSM + slope + WebGL flow.

---

## 2. What is experimentally validated vs unvalidated

### Existing component

- Depth Anything V2 wrapper
- georeferencing detection
- raster loading
- calibration engine structure
- slope computation
- WebGL render path

### Experimentally validated component

The project has validated these only on its earlier urban benchmark work, not on Indian hilly terrain. The evidence in the repo is not an India validation claim.

### Unvalidated component for Indian hilly disaster management

- terrain-specific absolute height on steep Himalayan slopes
- disaster-specific flood / landslide consequence estimation
- long-range slope integrity in Indian mountainous terrain
- building–terrain separation under steep hill and vegetation ambiguity
- any claim of real Indian benchmark success

---

## 3. Scientific interpretation

The current pipeline is a general urban/relative-height reconstruction system with a strong building-centric interpretation. It is not yet proven as a terrain-native model for Indian mountainous disaster scenarios.

The crucial question for Phase 60 is therefore not whether the demo looks good; it is whether the current pipeline physically behaves acceptably when given a real Indian mountain region and a reference DEM/DSM.
