# Phase 58: Indian Data Requirement for Hilly Terrain Feasibility

## Goal

To scientifically test whether DepthWizard can extend from urban single-view elevation toward Indian hilly disaster-management use, we need a small but representative benchmark. The benchmark must include terrain, vegetation, slope, and disaster-relevant labels. It should be built from open or low-friction public sources rather than a broad all-India download.

---

## High-priority requirement

We do not need all of India. We need a small, representative Indian hilly region with:

- steep and moderate terrain,
- vegetation and shadow,
- roads and settlements,
- local elevation variation,
- optionally a recent landslide or flood event,
- valid ground truth elevation and slope reference.

A benchmark should ideally contain at least one region from a north-Indian hilly state such as Uttarakhand, Himachal Pradesh, Jammu & Kashmir, Sikkim, or Northeast India.

---

## Proposed data categories

| Dataset | Region | Spatial resolution | Sensor / type | Available labels | Elevation availability | License / access | Potential use | Limitations |
|---|---|---:|---|---|---|---|---|---|
| Sentinel-2 | India, region-specific | 10-20 m | Optical multispectral | Land cover, vegetation, water, urban areas | No direct metric elevation | Open via Copernicus / EO Data Access | Visual context, land cover, vegetation / shadow analysis | Not enough for height truth |
| Landsat 8 / 9 | India, region-specific | 15-30 m | Optical | Land cover, broad surface classes | No direct elevation | Open USGS / EarthExplorer | Broad terrain and land-use context | Coarse for fine urban / slope structure |
| NASA SRTM | India-wide | 30 m | DEM | Elevation | Yes, DEM | Open | Baseline terrain reference, slope mask | Too coarse for fine urban and disaster-scale detailing |
| Copernicus DEM (GLO-30) | India and many global regions | ~30 m | DEM | Terrain height | Yes | Open | Terrain, slope, drainage context | Coarse for local building and landslide segmentation |
| ALOS/PALSAR or similar SAR DEM | Hilly regions | 12.5-30 m | SAR / DEM | Terrain structure | Yes, depending on product | Variable access | Terrain and slope evaluation | Not always directly aligned with optical imagery |
| Cartosat-1 / Cartosat-2 | India | 0.5-2.5 m or 1 m class | Optical stereo / DEM-capable | Ortho imagery, structures | Sometimes yes | Restricted / project-based | High-value local 3D context | Access and licensing may be limited |
| Bhuvan / ISRO open geospatial layers | India | Varies | National geospatial platform | Roads, land use, settlements, terrain products | Sometimes yes | Open / government | Regional terrain context and selection | Coverage varies by region |
| OpenStreetMap | India | Varies | Vector base map | Roads, buildings, land use | No terrain height | Open | Building exposure, road / settlement masks | Not elevation truth |
| Landslide inventory datasets | Indian hills | Varies | Event labels / GIS polygons | Landslide scars, hazard zones | Sometimes DEM-derived | Mixed open / project-based | Hazard-specific evaluation | Often sparse and not standardized |
| Flood extent / inundation datasets | India | Varies | GIS / remote sensing | Flood map polygons and water masks | Often partly via DEM | Mixed | Flood-relevant elevation assessment | Discrete maps not exact height truth |
| LiDAR where available | Selected Indian regions | <1 m | Active 3D sensing | Bare-earth, canopy, building, road surface | Yes | Limited / project-specific | Best benchmark for terrain and slope | Not broadly available in open datasets |

---

## Minimal benchmark specification

The smallest defensible benchmark should include:

1. A small region of Indian hilly terrain
2. High-resolution optical image coverage
3. Co-registered DEM or DSM for terrain truth
4. Slope grid and terrain mask by slope class
5. Building polygons or a subset with building height labels where available
6. At least one disaster-relevant terrain mask, such as landslide scarp or flood extent

A realistic minimal design is:

- 1-3 representative tiles from a single region
- 3 classes of slope: gentle, moderate, steep
- vegetation and shadow annotations
- roads and buildings as separate map layers
- validation split by region, not random tile mixing

---

## Priority order of evidence sources

1. Open DEM + local high-resolution optical imagery
2. Indian hilly subregions with known slope and landslide activity
3. Local building and road masks if available
4. Event-specific landslide / flood maps only as additional, not primary, labels
5. LiDAR or local elevation truth when accessible

This is the order that gives the strongest scientific value without broad, expensive data collection.

---

## Recommended initial region strategy

A small benchmark should be selected only after confirming actual data access. Good first targets include:

- Uttarakhand
- Himachal Pradesh
- Sikkim
- Jammu & Kashmir / Ladakh where appropriate
- Arunachal Pradesh / Northeast India where open DEM coverage is available

The benchmark should not attempt all of India at once. The first benchmark should be a focused region with known topographic variation and at least some public elevation / imagery access.

---

## What data is absolutely required before credible India claims

The following are the minimum required inputs for an Indian feasibility test:

- georeferenced optical imagery,
- DEM or DSM ground truth,
- slope ground truth or derived slope map,
- a small but representative terrain sample,
- a held-out test region not used for model selection,
- a disaster-relevant label set for at least one class of terrain hazard.

Without these, the project cannot honestly claim Indian readiness.

---

## Final conclusion on data requirement

The current repository does not have the required Indian hilly benchmark. A real feasibility study requires a focused public-domain benchmark, not synthetic or urban proxy data. The first benchmark should be intentionally small and terrain-specific, with region-level test splits and separate terrain and building metrics.
