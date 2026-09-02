# Phase 90: Indian mountainous 3D scene validation

TERRAIN_SOURCE = REAL GEOREFERENCED DEM
BUILDING_SOURCE = SINGLE-VIEW MODEL PREDICTIONS
HEIGHT_MAPPING = COMPONENT-ID CORRECTED
HEIGHT_ACCURACY = UNVALIDATED
Sikkim = LOCKED

## Scope
This phase validates pipeline integrity, geometric consistency, and scene rendering only. It does not validate building-height accuracy.

## Terrain and building construction
- Terrain meshes use the Phase 72 aligned DEM values directly in meters.
- Building footprints and heights were taken from Phase 89 corrected component-ID assignments.
- Finite-height components were extruded with vertical walls and roof surfaces.
- Missing-height components remain footprint-only and are explicitly marked HEIGHT_UNAVAILABLE; no numeric substitute was assigned.
- Vertical exaggeration factor: 1.0.

## Structural audit
### UTTARAKHAND
- CRS: EPSG:32644; resolution: [10.0, 10.0] m.
- Terrain elevation min/max/mean: 2953.3701171875 / 4592.90673828125 / 3560.871850371361 m.
- Terrain vertices: 262144; total triangles: 524987.
- Buildings: 11; finite heights: 9; HEIGHT_UNAVAILABLE: 2.
- Finite predicted height statistics: {'count': 9, 'min': 5.191013336181641, 'max': 41.42979431152344, 'mean': 17.35407288869222, 'median': 12.293121337890625, 'std': 10.971451836102299}.
- Geometry assertions passed: True.
- Failure flags: {'FLOATING_BUILDING': False, 'BURIED_BUILDING': False, 'NEGATIVE_HEIGHT': False, 'NAN_GEOMETRY': False, 'MISSING_HEIGHT': True, 'TERRAIN_DISCONTINUITY': False, 'TEXTURE_ALIGNMENT_FAILURE': False, 'EMPTY_SCENE': False}.
### HIMACHAL
- CRS: EPSG:32643; resolution: [10.0, 10.0] m.
- Terrain elevation min/max/mean: None / None / None m.
- Terrain vertices: 0; total triangles: 0.
- Buildings: 64; finite heights: 25; HEIGHT_UNAVAILABLE: 39.
- Finite predicted height statistics: {'count': 25, 'min': 4.402091979980469, 'max': 10.439148902893066, 'mean': 6.55735523223877, 'median': 6.321893215179443, 'std': 1.3327038618360265}.
- Geometry assertions passed: False.
- Failure flags: {'FLOATING_BUILDING': False, 'BURIED_BUILDING': False, 'NEGATIVE_HEIGHT': False, 'NAN_GEOMETRY': True, 'MISSING_HEIGHT': True, 'TERRAIN_DISCONTINUITY': True, 'TEXTURE_ALIGNMENT_FAILURE': False, 'EMPTY_SCENE': True}.

## RGB texture
- Actual B04/B03/B02 optical imagery was draped using the same crop alignment as Phase 89.
- Texture alignment was checked against the DEM crop dimensions and Phase 72 CRS/resolution metadata.

## Interactive controls
- The existing Three.js viewer was preserved and not rewritten; its orbit, zoom, pan, and reset-camera controls remain the project integration surface.

## Ground truth
- No verified Indian building footprint or building-height ground truth exists.
- No IoU, Dice, height MAE, height RMSE, or height correlation is reported.

## Limitations
- This is a visualization/integration validation, not geometric accuracy validation.
- Height coverage remains incomplete because the frozen model has unavailable components.
- The rendered building heights remain single-view model predictions and are unvalidated in India.

## Final decision
INDIAN_3D_SCENE_INTEGRATION_FAILED
