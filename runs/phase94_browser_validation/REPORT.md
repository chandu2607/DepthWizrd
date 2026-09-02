# Phase 94: DepthWizard browser validation

PRODUCTION_INTEGRATION = YES
TERRAIN_SOURCE = REAL GEOREFERENCED DEM
BUILDING_SOURCE = SINGLE-VIEW MODEL
HEIGHT_MAPPING = COMPONENT-ID CORRECTED
CANNY = OPTIONAL
POINT_CLOUD = OPTIONAL
HEIGHT_ACCURACY = UNVALIDATED
SIKKIM = LOCKED

## Browser startup

The real Streamlit application was started at `http://localhost:8511`. The initial browser shell showed `Running...` while the application initialized. It then reached the interactive DepthWizard UI. The initial browser capture is in `VISUALS/default_mode.png`.

A real browser defect was found during this phase: the new Indian `RasterInput` loader supplied tuple bounds, while the existing summary code requires a rasterio `BoundingBox`. The adapter was corrected and the application was restarted. The corrected run reached the integrated UI without that exception.

## Default mode regression

The default `Off` mode loaded the existing DepthWizard shell with the original upload/demo path and existing viewer controls. The existing viewer iframe exposed RGB, elevation, height, slope, camera presets, cinematic flythrough, and Fit to Scene/reset controls. Orbit, pan, and zoom instructions remained visible.

## Uttarakhand

The browser selected `Uttarakhand` and loaded the real georeferenced crop. The page displayed:

- Terrain: REAL GEOREFERENCED DEM
- Buildings: SINGLE-VIEW MODEL PREDICTIONS
- Height accuracy: UNVALIDATED
- CRS: EPSG:32644
- GSD: 10 m x 10 m
- Viewer structure count: 11

The integrated iframe rendered and exposed the existing camera controls. The authoritative Phase 89 component-ID scene remained the source.

## Himachal Pradesh

The browser selected `Himachal Pradesh` and loaded the Phase 91 valid crop. The page displayed:

- CRS: EPSG:32643
- GSD: 10 m x 10 m
- Viewer structure count: 1
- Existing camera presets and reset control

The scene loaded through the same existing viewer iframe.

## Missing heights

Uttarakhand's two unavailable heights remain null in the Phase 89 scene contract. They are not converted to zero, interpolated, or fabricated. The viewer scene retains the records and the inspector has `HEIGHT_UNAVAILABLE` handling.

## Canny

Canny OFF and ON were exercised on the same Uttarakhand scene. Both Streamlit reruns completed and the iframe retained 11 authoritative structures. The toggle does not replace the mask or change Phase 89 component IDs/heights.

Current limitation: Canny does not yet render a visible edge overlay in the viewer. It remains an optional auxiliary adapter cue, so this feature is only partially browser-validated.

## Point cloud

Point-cloud OFF and ON were exercised. The ON path is generated from authoritative terrain/building geometry by the adapter and does not infer missing height. The viewer structure count remained unchanged.

Current limitation: point cloud is representation/export-only and no viewport point-cloud layer is currently rendered.

## Interaction

The browser exposed orbit/pan/zoom guidance, camera presets, cinematic flythrough, and Fit to Scene/reset. The reset control was clicked successfully in the integrated iframe. The viewer also exposes click-to-inspect building interaction.

## Performance

Measured reference values are in `PERFORMANCE.json`:

- Uttarakhand existing pipeline processing: 11.83 s
- Himachal existing pipeline processing: 1.0 s
- Uttarakhand adapter: 1.2558 s; viewer HTML generation: 1.1242 s
- Himachal adapter: 1.3331 s; viewer HTML generation: 1.1491 s

## Failure flags

- `APP_NOT_LOADED`: false after restart
- `MODEL_INITIALIZATION_TIMEOUT`: false
- `INDIAN_SELECTOR_FAILURE`: false
- `TERRAIN_NOT_VISIBLE`: false for browser-loaded scenes
- `BUILDINGS_NOT_VISIBLE`: false; iframe structure counts were visible
- `MISSING_HEIGHT_BROKEN`: false
- `CANNY_CHANGES_AUTHORITATIVE_GEOMETRY`: false
- `POINT_CLOUD_INVALID`: false at adapter contract level
- `DEFAULT_PATH_BROKEN`: false after the bounds fix
- Optional feature limitations: visible Canny overlay and point-cloud viewport are not implemented

## Scientific boundary

This system demonstrates REAL INDIAN TERRAIN VISUALIZATION + SINGLE-VIEW BUILDING STRUCTURE OUTPUT + 3D SCENE INTEGRATION. It does not demonstrate validated Indian building-height accuracy, validated DSM accuracy, hazard prediction, landslide prediction, or flood prediction.

The result is therefore classified as partially validated: the core UI and both Indian scene paths work, while optional Canny and point-cloud presentation remain non-critical limitations.

## Final decision

DEPTHWIZARD_UI_PARTIALLY_VALIDATED
