# Phase 95: Complete Canny overlay and visible point-cloud mode

CORE_SCENE = PRESERVED
CANNY = AUXILIARY STRUCTURAL EDGE CUE
POINT_CLOUD = XYZ REPRESENTATION
TERRAIN = REAL GEOREFERENCED DEM
BUILDINGS = PHASE 89 COMPONENT-ID CORRECTED
HEIGHT_ACCURACY = UNVALIDATED
SIKKIM = LOCKED

## Implementation
- Canny uses the same aligned RGB and deterministic OpenCV configuration: low threshold 50, high threshold 150, aperture size 3.
- Canny is rendered as a screen-aligned semi-transparent PNG overlay in the existing Three.js viewer. It never changes authoritative geometry.
- Point cloud uses authoritative terrain and finite building mesh vertices in the existing viewer coordinate system and is rendered with `THREE.Points`.
- Unavailable-height component IDs remain metadata-only and receive no fabricated roof points.

## Validation
### UTTARAKHAND
- Canny edge fraction: 0.025951385498046875.
- Point count: 263986; finite XYZ: 263986; NaN XYZ: 0.
- Authoritative geometry hash unchanged for Canny and point-cloud toggles.
- Browser captures were taken for Canny OFF/ON, point-cloud OFF/ON, and both Indian scenes.
### HIMACHAL
- Canny edge fraction: 0.05645751953125.
- Point count: 262344; finite XYZ: 262344; NaN XYZ: 0.
- Authoritative geometry hash unchanged for Canny and point-cloud toggles.
- Browser captures were taken for Canny OFF/ON, point-cloud OFF/ON, and both Indian scenes.

## Scientific limits
- No quantitative accuracy claim is made for Canny or point-cloud modes.
- Height accuracy remains UNVALIDATED because Indian building ground truth is unavailable.
- Sikkim remains LOCKED.

## Browser validation

- Fresh Streamlit server started at `http://localhost:8511`.
- Default page loaded with existing upload path and viewer controls.
- Uttarakhand loaded with 11 structures; Canny ON exposed the `Canny auxiliary structural edge cue` layer.
- Point-cloud ON loaded with the authoritative point layer; the earlier NaN bounding-sphere warning was eliminated by finite XYZ filtering.
- Himachal loaded with one structure and EPSG:32643 metadata.
- Fit to Scene/reset was exercised in the existing iframe.
- Actual screenshots are stored in `VISUALS/`; details are in `BROWSER_QA.json`.

## Final decision
CANNY_AND_POINTCLOUD_VISUALLY_VALIDATED
