# Phase 58: Architecture Audit for Indian Hilly Terrain Feasibility

## Scope

This is a read-only audit of the current DepthWizard implementation. The objective is to determine which parts are already useful for terrain-aware disaster applications, which require modification, and which are fundamentally building-specific.

No model retraining, no checkpoint changes, and no production code changes are performed in this phase.

---

## Executive Summary

DepthWizard is currently strongest as a building-aware urban elevation reconstruction pipeline. Its main components are designed around RGB input, monocular depth prior, calibration, building masks, and 3D city visualization. These remain useful for terrain visualization when the terrain is gentle and the surface is close to the assumptions used in the current pipeline.

They are not yet validated for Indian hilly disaster scenarios because:

- the terrain branch is weakly modeled compared with the building branch,
- the system has not been benchmarked on Indian hilly ground truth,
- landslide, flood, and slope-distribution tasks are not modeled as first-class objectives,
- the calibration logic assumes surface regularity and does not explicitly represent steep terrain physics.

---

## Component Audit

| Component | Current Purpose | Building-specific? | Terrain usable? | Required modification |
|---|---|---:|---:|---|
| RGB input pipeline | Loads optical imagery, geospatial raster metadata, and prepares normalized RGB for downstream processing | No | Yes, but only as visual context and appearance prior | Keep as-is; the main challenge is terrain-aware calibration, not image loading |
| Depth Anything V2 integration | Produces a relative-depth prior from RGB input | No | Partially usable | Keep as a prior, but add explicit terrain-aware calibration and metric grounding |
| Corrected U-Net | Learns building mask / height-conditioned outputs from RGB and depth features | Mostly yes | Not in current form | Needs terrain branch or separate terrain head for continuous elevation modeling |
| Building-conditioned segmentation | Produces building mask and structural height evidence | Yes | Limited, not terrain semantics | Separate natural terrain / bare-earth modeling is required |
| Depth calibration | Converts relative depth to metric or relative DSM-like output | Partially | Partially | Needs DEM / DTM anchoring and slope-aware correction for hilly terrain |
| DSM / rDSM generation | Produces scene surface elevation or relative surface model | Partially | Yes for overall surface, but not bare-earth ground | Add explicit DTM decomposition and terrain-aware smoothing |
| nDSM / DTM processing | Distinguishes structure above ground from terrain | Partially | Yes in principle | Current morphological approximation is not robust for steep ridges / valleys / vegetation |
| Slope calculation | Computes local gradient magnitude and aspect | not inherently | Yes, this is a terrain-native capability | Needs terrain masking and hazard-focused analysis thresholds |
| 3D mesh generation | Creates explicit roofs, walls, and terrain surfaces for 3D viewer | Partially | Yes, but only as a surface visualization | Must add true terrain geometry and hazard overlays |
| Point-cloud / 3D rendering | Offers interactive geometry representation | Partially | Yes, as geometric representation | Potentially useful as an intermediate representation, but not sufficient by itself |
| Georeferencing support | Recognizes CRS and raster transform when provided | No | Yes, essential for real-world terrain analysis | Keep as-is; absolute elevation validation depends on it |
| Preprocessing / normalization | Normalizes depth and image statistics for model input | No | Yes, but with terrain-specific scaling risks | Must be risk-aware and region-aware, not only fixed global normalization |
| Inference pipeline | End-to-end stack from image to depth and geometry | Partially | Partially | Needs separate terrain inference path and evaluated disaster metrics |

---

## What is already applicable to terrain

The following pieces are relevant to terrain and disaster workflows:

1. RGB + georeferencing pipeline
   - If the image is georeferenced, the system can preserve spatial alignment and metric interpretation.
   - This is useful for terrain visualization and scenario overlays.

2. Depth Anything V2 as a visual prior
   - It captures relative scene structure and can be useful for broad surface organization.
   - It is not a metric terrain sensor, but as a visual prior it is still useful if properly calibrated.

3. DSM and DTM decomposition logic
   - The project explicitly distinguishes DSM, DTM, and nDSM concepts.
   - That is scientifically valuable because terrain analysis depends on bare-earth modeling, not only object height.

4. Slope analysis module
   - This is one of the most terrain-relevant components in the codebase.
   - It can produce slope magnitude and aspect, which are directly relevant to landslide-prone terrain.

5. 3D visualization pipeline
   - The app and WebGL viewer can show surfaces and height variation in a geographically interpretable way.
   - This is useful for rapid terrain visualization and event communication.

---

## What is only partially applicable

The following are useful but not sufficient for Indian hilly disaster work:

- Global normalization of depth values
- Morphological terrain extraction heuristics
- Building mask based structural priors
- City-centric 3D reconstruction logic
- Single-view relative depth under steep terrain

These components were designed around building height under urban assumptions, not terrain-physics modeling.

---

## What is fundamentally building-specific

The following components are not terrain-native and should not be treated as disaster-terrain models:

- building-conditioned U-Net outputs trained for building mask / urban structure localization,
- footprint extraction for building instance geometry,
- façade / roofing reconstruction logic,
- interpreting scene variation mainly as building massing rather than terrain morphology,
- building-oriented 3D mesh generation tuned to rooftops and walls.

These are valuable building features, but they are not the same as a terrain or landslide model.

---

## Key design concern

The project currently contains a clear split between:

- building height / urban structure
- terrain roughness / slope / DTM heuristics

But the terrain branch is not a first-class learned model. It is effectively a geometric post-processing layer on top of a building-centric system.

This is acceptable for urban city-scene reconstruction, but not for natural hilly terrain where bare-earth elevation, slope, drainage, and landslide susceptibility are the primary hazards.

---

## Final assessment

DepthWizard is not a terrain-native disaster model today.

It is a promising urban single-view elevation pipeline with some terrain-concept components. For Indian hilly areas, the current architecture can be reused only as a partial visual and geometric foundation, not as a validated disaster-analysis system.

The minimum scientific path is to add a terrain branch and evaluate it on Indian terrain ground truth before claiming operational usefulness.
