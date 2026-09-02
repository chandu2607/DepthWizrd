# Phase 92 minimal integration plan

## Boundary

Insert the Phase 89/91 path in `app.py` after raster/DEM selection and before `generate_interactive_webgl_html(...)`. Keep the current default upload/demo path unchanged until the adapter is validated.

## Reuse

- `app.py`: Streamlit shell, upload flow, session state, `components.html` embedding.
- `depthwizard/data/raster_loader.py`: `RasterInput` metadata and RGB ingestion where its radiometric behavior is appropriate.
- `depthwizard/depth/depth_anything.py`: frozen relative-depth inference when the selected experiment requires it.
- `depthwizard/models/building_conditioned_net.py`: frozen model/checkpoint definition.
- `depthwizard/viz/interactive_viewer.py`: Three.js scene, terrain/roof/wall buffers, textures, controls, camera presets, reset, and picking.
- Phase 72 aligned-grid conventions and Phase 89 component-ID assignments as experiment inputs, not copied blindly into production.

## Adapter needed

Create a new, isolated module such as `depthwizard/integration/phase89_scene_adapter.py` in a later implementation phase. It should:

1. Accept a validated common-grid DEM, RGB crop, Phase 89 component records, footprint masks/polygons, CRS, affine transform, and resolution.
2. Reject non-finite terrain needed for a mesh; preserve DEM nodata as invalid faces.
3. Convert raster pixels to projected coordinates, then to the viewer's explicit local meter coordinates.
4. Emit `predicted_height_by_component_id` and assert every finite height is assigned by component ID.
5. Emit `height_available=false` and `height=null` for unavailable components; never zero-fill or delete them.
6. Build terrain, finite building roofs/walls, footprint-only unavailable records, and provenance metadata.
7. Validate JSON-safe values before handing geometry to the viewer.

## Minimal viewer change

Add an optional prebuilt geometry/scene-data argument to `generate_interactive_webgl_html` or its nearest geometry boundary. The current `build_city_geometry` path remains the default. The Phase 89 adapter supplies the existing viewer schema plus `height_available`, `component_id`, CRS, transform, and bounds metadata.

Do not create a second viewer. Do not modify the current renderer behavior until adapter fixtures and real Indian scene checks pass.

## Application wiring

- Add an explicit scene-source selector or internal feature flag, defaulting to the current path.
- Store the adapter result in a separate session-state key, for example `phase89_scene`.
- Pass the adapter geometry to the existing HTML generator.
- Display provenance: real DEM source, optical source, checkpoint, threshold, component cap, and UNVALIDATED height status.
- Keep Phase 87-91 artifacts immutable.

## Validation gates before enabling by default

- Same CRS/transform/shape checks for DEM and RGB.
- Finite terrain coverage threshold and nodata-face handling.
- Component-ID mapping assertions and unavailable-height preservation.
- Base and roof consistency assertions.
- JSON serialization with NaN/Inf rejection.
- Existing viewer smoke test: orbit, pan, zoom, reset, camera presets, render modes, and building picking.
- No Indian accuracy claim without verified Indian building ground truth.
