# Phase 91: Valid Himachal scene window

TERRAIN_SOURCE = REAL GEOREFERENCED DEM
BUILDING_SOURCE = SINGLE-VIEW MODEL PREDICTIONS
HEIGHT_MAPPING = COMPONENT-ID CORRECTED
HEIGHT_ACCURACY = UNVALIDATED
Sikkim = LOCKED

## Window search
- Selected first row-major candidate: row_offset=7168, column_offset=9728, width=512, height=512.
- DEM finite fraction: 1.0; RGB finite fraction: 1.0; valid-mask fraction: 1.0.
- Selection required DEM >= 0.50, RGB >= 0.50, and at least one detected component.

## Himachal valid scene
- Terrain vertices: 262144; triangles: 522242.
- Buildings: 1; finite heights: 1; height-unavailable: 0.
- Terrain elevation statistics: {'count': 262144, 'min': 241.96485900878906, 'max': 263.89581298828125, 'mean': 246.30031192512251, 'median': 246.05883026123047, 'std': 1.5648916102893211, 'p95': 248.98579330444335, 'p99': 251.9276937866211}.
- Finite predicted height statistics: {'count': 1, 'min': 9.011548042297363, 'max': 9.011548042297363, 'mean': 9.011548042297363, 'median': 9.011548042297363, 'std': 0.0, 'p95': 9.011548042297363, 'p99': 9.011548042297363}.
- Roof elevation statistics: {'count': 1, 'min': 255.30532836914062, 'max': 255.30532836914062, 'mean': 255.30532836914062, 'median': 255.30532836914062, 'std': 0.0, 'p95': 255.30532836914062, 'p99': 255.30532836914062}.
- Mapping mismatches: 0.
- Failure flags: {'FLOATING_BUILDING': False, 'BURIED_BUILDING': False, 'NEGATIVE_HEIGHT': False, 'NAN_GEOMETRY': False, 'MISSING_HEIGHT': False, 'TERRAIN_DISCONTINUITY': False, 'TEXTURE_ALIGNMENT_FAILURE': False, 'EMPTY_SCENE': False}.

## Uttarakhand comparison
- Phase 90 stored scene: {'terrain_vertices': 262144, 'terrain_bounds': {'x_m': [0.0, 5110.0], 'y_m': [0.0, 5110.0], 'z_m': [2953.3701171875, 4634.33642578125]}, 'building_count': 11, 'finite_heights': 9, 'height_unavailable': 2, 'geometry_failures': {'MISSING_HEIGHT': True}}.

## Integrity limits
- Missing heights remain unavailable; no zero or interpolated heights were assigned.
- No Indian building ground truth exists, so no accuracy metrics are reported.
- The scene validates window overlap and pipeline integration, not building-height accuracy.

## Final decision
VALID_HIMALAYAN_SCENE_WINDOW_FOUND
