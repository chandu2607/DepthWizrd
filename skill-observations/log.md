# Skill Observations

### Observation 1: Strict JSON exposed real empty-overlap cases
Status: OPEN
The Phase 87 audit initially attempted to serialize NaN raster/component summaries. Normalizing non-finite metadata to null preserved the evidence and exposed that some predicted components had no finite DEM overlap; strict artifact validation was useful as a scientific integrity check.

### Observation 2: Separate model ordering from record ordering
Status: OPEN
The Phase 88 audit showed that a model can return finite per-object predictions while downstream records still contain missing or misassigned values when connected-component ordering changes between model forward and post-processing. Auditing component IDs and order is necessary before calling this a capacity-only failure.

### Observation 3: Identity-based assignment can be validated without retraining
Status: OPEN
The Phase 89 correction used spatial overlap to translate model-side component IDs into final raster component IDs, then asserted every finite assignment by ID. This fixed the mapping while preserving the model cap and unavailable components, demonstrating a reusable surgical repair pattern for frozen inference pipelines.

### Observation 4: Integration validation must audit terrain coverage separately
Status: OPEN
Phase 90 showed that correct component-height mapping does not guarantee a valid scene: the exact Himachal inference crop had zero finite DEM pixels. A scene audit must validate terrain coverage and geometry independently from model output availability.

### Observation 5: Architecture audits should verify behavior at boundaries
Status: OPEN
Phase 92 found the application had a clear viewer boundary, but the viewer rebuilt building evidence internally and did not preserve geospatial metadata or unavailable-height semantics. Auditing the actual data handoff is more informative than matching module names alone.

