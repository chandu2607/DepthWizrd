# Phase 57: Indian Hilly-Region Generalization Audit

## Executive Summary

DepthWizard is not yet scientifically defensible for Indian hilly-region disaster-management use. The current codebase and evidence base support a statement like this:

- Valid evidence exists for a narrow urban / city-scene regime within the project’s own datasets and evaluation pipeline.
- The project has no real Indian hilly-terrain benchmark in this workspace.
- The project has no real landslide / flood / terrain-change validation on Indian scenes.
- The current single-view geometry pipeline is not validated under strong slope, vegetation, narrow valleys, ridge lines, or disaster-triggered ground deformation.

This is a limitation, not a marketing problem. The correct position is: the method is promising for controlled urban monocular elevation tasks, but it is not proven for Indian disaster-management scenarios, especially in hilly regions of northern India.

---

## Scientific conclusion

### Verdict

Status: NOT VALIDATED FOR INDIAN HILLY DISASTER MANAGEMENT

### Why this verdict is required

1. No Indian hilly benchmark is present in the repository.
   - The real data directories under `data/` are `dfc2019` and `dfc2023_multicity`, which are not Indian hilly terrain datasets.
   - There are no Indian state / ISRO / landslide / flood / terrain-change datasets included in the workspace.

2. The project’s prior metric claims were found to be synthetic or non-evidence-bearing.
   - `scripts/phase55_clean_selection.py` contains placeholder logic such as:
     - `# For now, return placeholder (actual inference would happen here)`
     - `result['dice'] = result['iou'] * 1.2  # Placeholder`
     - `# Placeholder evaluation`
     - `# Placeholder NY results`
   - `scripts/phase56_forensic_verification.py` explicitly audits the mismatch and concludes the earlier timing and results are not plausible as real inference evidence.

3. The current core architecture is still a single-view monocular depth + calibration pipeline, not a validated terrain hazard system.
   - `depthwizard/depth/depth_anything.py` uses Depth Anything V2 as a relative-depth prior; this is not a metric elevation sensor.
   - `depthwizard/calibration/engine.py` contains fallbacks that normalize depth and use morphological terrain approximations. This is reasonable for controlled scenes, but it is not terrain-safe or disaster-safe under real slope variation.
   - `depthwizard/analysis/slope.py` separates terrain slope from facade slope, but this is a heuristic for building recovery, not a validated landslide/flood hazard model.

4. The project is missing the specific conditions that disaster management in hilly India requires.
   - Landslide-prone slopes
   - Dense vegetation and shadow
   - Ridge-valley terrain with steep local gradients
   - Elevation error that is meaningful in meters, not just relative rank correlation
   - Terrains where ground surface itself changes after rainfall, erosion, or failure
   - Disaster labels tied to actual events and georeferenced DEM pairs

---

## Evidence reviewed

The audit used the actual project code and prior forensic artifacts:

- `scripts/phase55_clean_selection.py`
- `scripts/phase56_forensic_verification.py`
- `depthwizard/depth/depth_anything.py`
- `depthwizard/calibration/engine.py`
- `depthwizard/analysis/slope.py`
- `depthwizard/analysis/height.py`
- `depthwizard/config.py`
- `app.py`
- `data/` dataset directories

The strongest honest statement supported by these files is:

- The system has evidence for self-consistent urban-scene geometry, not Indian hilly-scene disaster readiness.
- The evidence standard is therefore not met for Indian deployment claims.

---

## The likely failure modes in hilly Indian terrain

### 1) Terrain-slope bias
The calibration pipeline assumes a mostly regular surface and uses normalized or smoothed depth fields. In steep terrain, this can misclassify true terrain slope as object height, or vice versa. The documented design in `depthwizard/calibration/engine.py` uses morphological opening and smoothing to estimate DTM, but those approximations are not validated for ridge-and-valley slope regimes.

Risk: overestimation of terrain as building / roof structures, or underestimation of landslide scarps and embankments.

### 2) DTM underfit in steep or vegetated topography
The DTM extraction is a generic morphological approximation rather than a terrain-physics model. In hilly and vegetated landscapes, terrain surfaces are not just “objects on a smooth ground” and cannot be recovered reliably from a single RGB view without a geospatial terrain prior.

Risk: large vertical bias in terrain reconstruction and wrong DSM/nDSM decomposition.

### 3) Flood / debris-flow / landslide signals are not modeled
The current app and calibration pipeline are oriented toward surface and building reconstruction, not event detection or terrain-change analysis. There is no explicit logic for flood extent, landslide scar, debris flow, riverbank erosion, or terrain displacement.

Risk: visually plausible 3D output that still fails the hazard question.

### 4) Single-view monocular depth is underconstrained for metric terrain height
Depth Anything V2 is a relative-depth prior, not a calibrated geometric sensor. Its outputs are scale/shift ambiguous and the project acknowledges this structure explicitly in `depthwizard/depth/depth_anything.py`.

Risk: condition-dependent scale drift when the camera viewpoint, terrain slope, or scene context differs substantially from the training distribution.

### 5) No Indian domain validation
There is no held-out Indian test split with georeferenced elevation truth, slope strata, and disaster labels. Without that, cross-domain claims are not defensible.

---

## What is scientifically defensible today

The honest answer is limited to the following:

- The pipeline is a plausible research pipeline for urban monocular elevation reconstruction using relative depth and calibration heuristics.
- The project has no valid evidence for Indian hilly-region performance.
- The project has no valid evidence for landslide or flood hazard assessment in Indian terrain.
- Any demo that looks convincing on urban imagery is not proof of disaster-management readiness.

---

## Minimum justified next step

The smallest scientifically justified next step is not retraining the full project blindly. It is a controlled external-domain benchmark design.

### Step 1: build an Indian hilly benchmark
Create a dataset with:

- georeferenced optical imagery from Indian hilly regions
- DEM / LiDAR / DSM ground truth
- terrain masks by slope class (gentle, moderate, steep)
- vegetation / forest / shadow labels
- landslide and flood event labels where available
- train / validation / external test splits by region, not random tile mixing

### Step 2: evaluate the current model on real terrain, not urban proxy data
Measure:

- DTM MAE / RMSE
- nDSM MAE / RMSE
- building-height error on steep slopes
- terrain-slope error by slope bin
- disaster-region performance separately from normal terrain

### Step 3: compare at least three calibration regimes
1. relative monocular only
2. DEM-anchored calibration
3. ground-referenced / GCP-anchored calibration

Only after those results are produced can the project make a claim about Indian usefulness.

### Step 4: decide whether the system is adequate for disaster management
If Indian hilly performance is poor, the correct next move is not a flashy demo. It is a targeted architecture change for terrain-aware calibration, DEM priors, and explicit hazard-aware evaluation.

---

## Recommendation

### Do not claim Indian disaster-management readiness yet.
### Do not present the current pipeline as operationally useful for Indian landslides, floods, or hilly terrain.
### Do not optimize for a better-looking demo while the scientific domain gap remains unmeasured.

The correct path is:

- build evidence first,
- measure Indian terrain error honestly,
- only then decide whether the system is a viable ISRO-oriented disaster-management tool.

---

## Final project position

DepthWizard is a promising research and visualization prototype for urban monocular elevation reconstruction, but it is not scientifically defensible as an Indian hilly-disaster-management solution until real Indian terrain validation is completed and passes the required error and hazard metrics.
