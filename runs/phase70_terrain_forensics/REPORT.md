# Phase 70 Terrain Forensics Report

PHASE 70 STATUS: FORENSIC_AUDIT_COMPLETE
METRIC_RECONCILIATION: MIXED_RAW_AND_NORMALIZED_TARGETS
TARGET_UNITS: DEM raw values are in meters; the pilot target was normalized to [0,1] and reported without inverse transform
NORMALIZATION: train-only normalization is not yet implemented consistently; phase69 compared normalized predictions to normalized targets without inverse scaling
RGB_ELEVATION_ALIGNMENT: raw optical and raw DEM are not in a common geospatial frame; DEM is EPSG:4326 while optical bands are EPSG:32643/44/45 and were center-cropped without reprojection
CURRENT_MODEL_TERRAIN_COMPATIBILITY: NOT YET PROVEN; current model is building-conditioned and not a dense terrain regressor
TERRAIN_HEAD_REQUIRED: YES
ONE_EPOCH_PILOT: RAN; pipeline sanity only
SMALL_PILOT: NOT YET RUN
SIKKIM_LOCKED: YES
SIKKIM_EVALUATED: NO
CANNY_INCLUDED: NO
POINT_CLOUD_INCLUDED: NO
BUILDING_TRAINING: NO
PRODUCTION_CHANGED: NO

## Primary finding
The largest false signal is not a hidden architecture failure. The current Phase 69 workflow mixed two different target spaces:

1. raw DEM values in meters from `region_dir / f'{region_name}_dem.tif'`
2. normalized DEM target in [0,1] produced by `dem_norm = (dem - min) / (max - min)`

The baseline pipeline applied a linear affine fit from relative depth output to raw DEM, while the one-epoch pilot compared the model output raw tensor against the normalized DEM target without inverse scaling. These are not the same metric space, so the reported MAEs are not directly comparable.

## Geospatial finding
The raw DEM rasters are in EPSG:4326, while the optical Sentinel-2 tiles are in EPSG:32643/44/45. The phase69 code center-cropped raw arrays without reprojecting the DEM to the same CRS and transform. That means the arrays may have identical shape but not the same physical footprint or matched pixel coordinates.

## Model semantics
The current architecture is building-conditioned: mask logits, connected components, object pooling, expert heads, and height bins are for building/object height estimation. It is not a dense terrain-elevation estimator and must not be called a terrain DTM model without a dedicated terrain head.

FIRST_CONFIRMED_FAILURE: Target construction and evaluation used mixed normalizations, and RGB/DEM were not aligned in a common geospatial frame.
MINIMUM_REQUIRED_FIX: Reproject DEM to the optical CRS before alignment, build one explicit terrain target/unit definition, apply train-only normalization with inverse-transform before metrics, and then add a dense terrain-regression head.
NEXT_PHASE: Implement the minimal terrain regression head and lock the validation/test split before any new training run.
